import numpy as np
import torch
import torch_geometric
from cpp_torch.modules.custom_gat_conv import GATv2Conv


class ResidualBlock(torch.nn.Module):

    def __init__(self, layer, input_dim, output_dim, dropout_rate):
        super().__init__()
        self.layer = layer
        self.projection = torch.nn.Linear(input_dim, output_dim) if input_dim != output_dim else torch.nn.Identity()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x_in):
        x_out = self.layer(x_in)
        residual = self.projection(x_in)
        residual = self.dropout(residual)
        return x_out + residual


class ExtendedGATLayer(torch.nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            edge_index,
            heads,
            dropout_rate_gat,
            dropout_rate_feature,
            use_residuals,
    ):
        super().__init__()
        if out_features % heads != 0:
            raise ValueError('heads must divide out_features')
        gat_out_per_head = out_features // heads
        self.edge_index = edge_index
        self.gat = GATv2Conv(
            in_features,
            gat_out_per_head,
            heads=heads,
            concat=True,
            dropout=dropout_rate_gat,
            add_self_loops=True,
            bias=True,
            residual=use_residuals,
        )
        self.layers = torch.nn.Sequential(
            torch.nn.LayerNorm(out_features),
            torch.nn.GELU(approximate='tanh'),
            torch.nn.Dropout(dropout_rate_feature),
        )

    def forward(self, x_in):
        x_out = self.gat(x_in, self.edge_index)
        return self.layers(x_out)



class AveragePooling(torch_geometric.nn.MessagePassing):

    def __init__(self, edge_index, num_output_nodes):
        super().__init__(aggr='mean')
        self.edge_index = edge_index
        self.num_output_nodes = num_output_nodes

    def forward(self, x_in):
        return self.propagate(
            self.edge_index,
            x=x_in,
            size=(x_in.shape[0], self.num_output_nodes)
        )

    def message(self, x_j):
        return x_j


def compute_target_connectivity(num_nodes, edge_index):
    connectivity = np.zeros(num_nodes, dtype=int)
    for (source, target) in edge_index.T:
        connectivity[target] += 1
    return connectivity


def assert_target_connectivity(connectivity, expected):
    unique_connectivity = np.unique(connectivity)
    if set(unique_connectivity) != set(expected):
        raise ValueError('unexpected connectivity')


def select_source_with_connectivity(
    num_output_nodes,
    target_connectivity,
    selected_connectivity,
    edge_index,
):
    # list of targets with the given number of sources
    targets = [i for i in range(num_output_nodes) if target_connectivity[i] == selected_connectivity]

    # list of sources for each targets
    indices = np.zeros(shape=(len(targets), selected_connectivity), dtype=int)
    for (i, target) in enumerate(targets):
        sources = [source for (source, t) in edge_index.T if t == target]
        indices[i, :] = sources

    return targets, indices


class ParametrisedPooling(torch.nn.Module):

    def __init__(self, graph, source_graph_level, target_graph_level):
        super().__init__()
        # select the appropriate graph
        graph_in = f'hidden_{source_graph_level}'
        graph_out = f'hidden_{target_graph_level}'
        edge_index = graph[graph_in, 'to', graph_out]['edge_index'].numpy()
        num_output_nodes = graph[graph_out]['x'].shape[0]

        # compute target connectivity
        target_connectivity = compute_target_connectivity(num_output_nodes, edge_index)

        # check that all targets have either 6 or 7 sources
        assert_target_connectivity(target_connectivity, expected=[6, 7])

        # list of target with 6/7 sources and their associated sources
        target_6 = [i for i in range(num_output_nodes) if target_connectivity[i] == 6]
        target_7 = [i for i in range(num_output_nodes) if target_connectivity[i] == 7]

        # list of sources for each targets
        indices_6 = np.zeros(shape=(len(target_6), 6), dtype=int)
        indices_7 = np.zeros(shape=(len(target_7), 7), dtype=int)
        for (i, target) in enumerate(target_6):
            sources = [source for (source, t) in edge_index.T if t == target]
            indices_6[i, :] = sources
        for (i, target) in enumerate(target_7):
            sources = [source for (source, t) in edge_index.T if t == target]
            indices_7[i, :] = sources

        # indices for the final permutation
        indices_target = np.zeros(num_output_nodes, dtype=int)
        for (i, j) in enumerate(target_6):
            indices_target[j] = i
        for (i, j) in enumerate(target_7):
            indices_target[j] = i + len(target_6)

        # register buffers and weights
        self.shape_6 = (*indices_6.shape, -1)
        self.shape_7 = (*indices_7.shape, -1)
        self.register_buffer('indices_6', torch.from_numpy(indices_6.flatten()))
        self.register_buffer('indices_7', torch.from_numpy(indices_7.flatten()))
        self.register_buffer('indices_target', torch.from_numpy(indices_target))
        self.weights_6 = torch.nn.Parameter(torch.ones(len(target_6), 6))
        self.weights_7 = torch.nn.Parameter(torch.ones(len(target_7), 7))

    def forward(self, x_in):
        # select the sources of targets with 6 sources
        x_6 = torch.index_select(x_in, dim=0, index=self.indices_6)
        # multiply by the associated weights
        x_6 = x_6.reshape(self.shape_6)
        x_6 = torch.einsum(
            '...ij,...i->...j',
            x_6,
            self.weights_6,
        )
        # select the sources of targets with 7 sources
        x_7 = torch.index_select(x_in, dim=0, index=self.indices_7)
        # multiply by the associated weights
        x_7 = x_7.reshape(self.shape_7)
        x_7 = torch.einsum(
            '...ij,...i->...j',
            x_7,
            self.weights_7,
        )
        # concatenate the output
        x = torch.cat((x_6, x_7), axis=0)
        # final permutation to recover the order of output nodes
        x = torch.index_select(x, dim=0, index=self.indices_target)
        return x


class ParametrisedUnPooling(torch.nn.Module):

    def __init__(self, graph, source_graph_level, target_graph_level):
        super().__init__()
        # select the appropriate graph
        graph_in = f'hidden_{source_graph_level}'
        graph_out = f'hidden_{target_graph_level}'
        edge_index = graph[graph_in, 'to', graph_out]['edge_index'].numpy()
        num_output_nodes = graph[graph_out]['x'].shape[0]

        # compute target connectivity
        target_connectivity = compute_target_connectivity(num_output_nodes, edge_index)

        # check that all targets have either 4 or 7 sources
        assert_target_connectivity(target_connectivity, expected=[4, 6, 7])

        # list of target with 4/6/7 sources and their associated sources
        target_4 = [i for i in range(num_output_nodes) if target_connectivity[i] == 4]
        target_6 = [i for i in range(num_output_nodes) if target_connectivity[i] == 6]
        target_7 = [i for i in range(num_output_nodes) if target_connectivity[i] == 7]

        # list of sources for each targets
        indices_4 = np.zeros(shape=(len(target_4), 4), dtype=int)
        indices_6 = np.zeros(shape=(len(target_6), 6), dtype=int)
        indices_7 = np.zeros(shape=(len(target_7), 7), dtype=int)
        for (i, target) in enumerate(target_4):
            sources = [source for (source, t) in edge_index.T if t == target]
            indices_4[i, :] = sources
        for (i, target) in enumerate(target_6):
            sources = [source for (source, t) in edge_index.T if t == target]
            indices_6[i, :] = sources
        for (i, target) in enumerate(target_7):
            sources = [source for (source, t) in edge_index.T if t == target]
            indices_7[i, :] = sources

        # indices for the final permutation
        indices_target = np.zeros(num_output_nodes, dtype=int)
        for (i, j) in enumerate(target_4):
            indices_target[j] = i
        for (i, j) in enumerate(target_6):
            indices_target[j] = i + len(target_4)
        for (i, j) in enumerate(target_7):
            indices_target[j] = i + len(target_4) + len(target_6)

        # register buffers and weights
        self.shape_4 = (*indices_4.shape, -1)
        self.shape_6 = (*indices_6.shape, -1)
        self.shape_7 = (*indices_7.shape, -1)
        self.register_buffer('indices_4', torch.from_numpy(indices_4.flatten()))
        self.register_buffer('indices_6', torch.from_numpy(indices_6.flatten()))
        self.register_buffer('indices_7', torch.from_numpy(indices_7.flatten()))
        self.register_buffer('indices_target', torch.from_numpy(indices_target))
        self.weights_4 = torch.nn.Parameter(torch.ones(len(target_4), 4))
        self.weights_6 = torch.nn.Parameter(torch.ones(len(target_6), 6))
        self.weights_7 = torch.nn.Parameter(torch.ones(len(target_7), 7))

    def forward(self, x_in):
        # select the sources of targets with 4 sources
        x_4 = torch.index_select(x_in, dim=0, index=self.indices_4)
        # multiply by the associated weights
        x_4 = x_4.reshape(self.shape_4)
        x_4 = torch.einsum(
            '...ij,...i->...j',
            x_4,
            self.weights_4,
        )
        # select the sources of targets with 6 sources
        x_6 = torch.index_select(x_in, dim=0, index=self.indices_6)
        # multiply by the associated weights
        x_6 = x_6.reshape(self.shape_6)
        x_6 = torch.einsum(
            '...ij,...i->...j',
            x_6,
            self.weights_6,
        )
        # select the sources of targets with 7 sources
        x_7 = torch.index_select(x_in, dim=0, index=self.indices_7)
        # multiply by the associated weights
        x_7 = x_7.reshape(self.shape_7)
        x_7 = torch.einsum(
            '...ij,...i->...j',
            x_7,
            self.weights_7,
        )
        # concatenate the output
        x = torch.cat((x_4, x_6, x_7), axis=0)
        # final permutation to recover the order of output nodes
        x = torch.index_select(x, dim=0, index=self.indices_target)
        return x


class AddNoise(torch.nn.Module):

    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x_in):
        if self.training:
            noise = torch.randn_like(x_in) * self.std
            x_in = x_in + noise
        return x_in


class ProgressiveEncoder(torch.nn.Sequential):

    def __init__(
            self,
            input_dim,
            input_graph_level,
            hidden_dims,
            hidden_graph_levels,
            latent_dim,
            graph,
            heads,
            gat_dropout,
            feature_dropout,
            latent_dropout,
            latent_noise_std,
            use_residuals,
            use_residuals_io,
    ):
        super().__init__()
        self.append_input_layers(
            graph=graph,
            graph_level=input_graph_level,
            input_dim=input_dim,
            output_dim=hidden_dims[0],
            dropout_rate=feature_dropout,
            residual=use_residuals_io,
        )
        self.append_internal_layers(
            graph=graph,
            current_graph_level=input_graph_level,
            hidden_dims=hidden_dims,
            hidden_graph_levels=hidden_graph_levels,
            heads=heads,
            dropout_rate_gat=gat_dropout,
            dropout_rate_feature=feature_dropout,
            use_residuals=use_residuals,
        )
        self.append_output_layers(
            input_dim=hidden_dims[-1],
            output_dim=latent_dim,
            dropout_rate_feature=feature_dropout,
            dropout_rate_latent=latent_dropout,
            latent_noise_std=latent_noise_std,
        )

    def append_input_layers(
            self,
            graph,
            graph_level,
            input_dim,
            output_dim,
            dropout_rate,
            residual,
    ):
        hidden_graph = f'hidden_{graph_level}'
        self.append(AveragePooling(
            edge_index=graph['data', 'to', hidden_graph]['edge_index'],
            num_output_nodes=graph[hidden_graph]['x'].shape[0],
        ))
        layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.LayerNorm(output_dim),
            torch.nn.GELU(approximate='tanh'),
            torch.nn.Dropout(dropout_rate),
        )
        if residual:
            self.append(ResidualBlock(
                layer=layers,
                input_dim=input_dim,
                output_dim=output_dim,
                dropout_rate=dropout_rate/2,
            ))
        else:
            self.extend(layers)

    def append_internal_layers(
            self,
            graph,
            current_graph_level,
            hidden_dims,
            hidden_graph_levels,
            heads,
            dropout_rate_gat,
            dropout_rate_feature,
            use_residuals,
    ):
        for (in_dim, out_dim, graph_level) in zip(hidden_dims, hidden_dims[1:], hidden_graph_levels):
            hidden_graph = f'hidden_{current_graph_level}'
            self.append(ExtendedGATLayer(
                in_features=in_dim,
                out_features=out_dim,
                edge_index=graph[hidden_graph, 'to', hidden_graph]['edge_index'],
                heads=heads,
                dropout_rate_gat=dropout_rate_gat,
                dropout_rate_feature=dropout_rate_feature,
                use_residuals=use_residuals,
            ))
            if graph_level == current_graph_level:
                pass
            elif graph_level == current_graph_level - 1:
                current_graph_level = graph_level
                self.append(ParametrisedPooling(
                    graph=graph,
                    source_graph_level=current_graph_level+1,
                    target_graph_level=current_graph_level,
                ))
            else:
                raise ValueError('incompatible graph levels')

    def append_output_layers(
            self,
            input_dim,
            output_dim,
            dropout_rate_feature,
            dropout_rate_latent,
            latent_noise_std,
    ):
        hidden_dim = 2 * output_dim
        self.extend((
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(approximate='tanh'),
            torch.nn.Dropout(dropout_rate_feature),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Dropout(dropout_rate_latent),
        ))
        if latent_noise_std > 0:
            self.append(AddNoise(latent_noise_std))


class ProgressiveDecoder(torch.nn.Sequential):

    def __init__(
            self,
            latent_dim,
            input_graph_level,
            hidden_dims,
            hidden_graph_levels,
            output_dim,
            graph,
            heads,
            gat_dropout,
            feature_dropout,
            use_residuals,
            use_residuals_io,
    ):
        super().__init__()
        self.append_input_layers(
            input_dim=latent_dim,
            output_dim=hidden_dims[0],
            dropout_rate=feature_dropout,
        )
        self.append_internal_layers(
            graph=graph,
            current_graph_level=input_graph_level,
            hidden_dims=hidden_dims,
            hidden_graph_levels=hidden_graph_levels,
            heads=heads,
            dropout_rate_gat=gat_dropout,
            dropout_rate_feature=feature_dropout,
            use_residuals=use_residuals,
        )
        self.append_output_layers(
            graph=graph,
            graph_level=hidden_graph_levels[-1],
            input_dim=hidden_dims[-1],
            output_dim=output_dim,
            residual=use_residuals_io,
            dropout_rate=feature_dropout,
        )

    def append_input_layers(
            self,
            input_dim,
            output_dim,
            dropout_rate,
    ):
        hidden_dim = max(input_dim * 2, output_dim)
        self.extend((
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(approximate='tanh'),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.LayerNorm(output_dim),
            torch.nn.GELU(approximate='tanh'),
            torch.nn.Dropout(dropout_rate),
        ))

    def append_internal_layers(
            self,
            graph,
            current_graph_level,
            hidden_dims,
            hidden_graph_levels,
            heads,
            dropout_rate_gat,
            dropout_rate_feature,
            use_residuals,
    ):
        for (in_dim, out_dim, graph_level) in zip(hidden_dims, hidden_dims[1:], hidden_graph_levels):
            if graph_level == current_graph_level:
                pass
            elif graph_level == current_graph_level + 1:
                current_graph_level = graph_level
                self.append(ParametrisedUnPooling(
                    graph=graph,
                    source_graph_level=current_graph_level-1,
                    target_graph_level=current_graph_level,
                ))
            else:
                raise ValueError('incompatible graph levels')
            hidden_graph = f'hidden_{current_graph_level}'
            self.append(ExtendedGATLayer(
                in_features=in_dim,
                out_features=out_dim,
                edge_index=graph[hidden_graph, 'to', hidden_graph]['edge_index'],
                heads=heads,
                dropout_rate_gat=dropout_rate_gat,
                dropout_rate_feature=dropout_rate_feature,
                use_residuals=use_residuals,
            ))

    def append_output_layers(
            self,
            graph,
            graph_level,
            input_dim,
            output_dim,
            dropout_rate,
            residual,
    ):
        layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.GELU(approximate='tanh'),
            torch.nn.Linear(input_dim, output_dim),
        )
        if residual:
            self.append(ResidualBlock(
                layer=layers,
                input_dim=input_dim,
                output_dim=output_dim,
                dropout_rate=dropout_rate / 2,
            ))
        else:
            self.extend(layers)
        hidden_graph = f'hidden_{graph_level}'
        self.append(AveragePooling(
            edge_index=graph[hidden_graph, 'to', 'data']['edge_index'],
            num_output_nodes=graph['data']['x'].shape[0],
        ))
