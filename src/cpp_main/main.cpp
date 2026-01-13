#include <torch/script.h>
#include <torch/autograd.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// dump tensor to binary
void dump(const std::string& name, const torch::Tensor& t) {
    std::ofstream f(name, std::ios::binary);
    auto cpu = t.detach().cpu().contiguous();
    std::cout << "writing " << name << std::endl;
    f.write((char*)cpu.data_ptr(), cpu.numel() * sizeof(float));
}

// flatten tensor
torch::Tensor flatten(const std::vector<torch::Tensor> t_in) {
    std::vector<torch::Tensor> components;
    for (const auto & t : t_in) {
        components.push_back(t.flatten());
    }
    return torch::cat(components);
}

// read shape from text file
std::vector<std::int64_t> read_shape(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    std::string line;

    // read number of dimensions
    int ndim;
    file >> ndim;

    // read each dimension size
    std::vector<std::int64_t> sizes;
    sizes.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
        std::int64_t dim;
        file >> dim;
        sizes.push_back(dim);
    }
    return sizes;
}

// wrapper around scripted module
class ScriptedModule {
    public:
        using Shape = std::vector<std::int64_t>;
        ScriptedModule(std::string filename, Shape input_shape, Shape output_shape) :
            m_model(torch::jit::load(filename)),
            m_input_shape(std::move(input_shape)),
            m_output_shape(std::move(output_shape)),
            m_num_parameters(0) {
                for (const auto& p : m_model.parameters())
                {
                    m_parameters.push_back(p);
                    m_num_parameters += p.numel();
                }
            }
        const Shape & get_input_shape() const { return m_input_shape; }
        const Shape & get_output_shape() const { return m_output_shape; }
        std::int64_t get_num_parameters() const { return m_num_parameters; }
        std::vector<torch::Tensor> unflatten(torch::Tensor flat_p) const {
            std::vector<torch::Tensor> p;
            std::int64_t start = 0;
            for (const auto& t : m_parameters) {
                std::int64_t len = t.numel();
                p.push_back(flat_p.slice(0, start, start + len).view(t.sizes()));
                start += len;
            }
            return p;
        }
        void reset_parameters(torch::Tensor flat_p) {
            std::vector<torch::Tensor> p(unflatten(flat_p));
            torch::autograd::GradMode::set_enabled(false);
            for (int i = 0; i < m_parameters.size(); ++i) {
                m_parameters[i].copy_(p[i]);
            }
            torch::autograd::GradMode::set_enabled(true);
        }
        torch::Tensor forward(torch::Tensor x) {
            m_x = x.requires_grad_(true);
            m_y = m_model.forward({m_x}).toTensor();
            return m_y;
        }
        torch::Tensor apply_ad_state(torch::Tensor dy) {
            return torch::autograd::grad(
                {m_y},
                {m_x},
                {dy},
                true // retain_graph
            )[0];
        }
        torch::Tensor apply_ad_parameters(torch::Tensor dy) {
            return flatten(torch::autograd::grad(
                {m_y},
                m_parameters,
                {dy},
                true // retain_graph
            ));
        }
        torch::Tensor apply_tl_state(torch::Tensor dx) {
            torch::Tensor dummy = torch::randn(m_output_shape).requires_grad_(true);
            torch::Tensor Fx_dummy = torch::autograd::grad(
                {m_y},
                {m_x},
                {dummy},
                true, // retain_graph
                true // create_graph
            )[0];
            return torch::autograd::grad(
                {Fx_dummy},
                {dummy},
                {dx}
            )[0];
        }
        torch::Tensor apply_tl_parameters(torch::Tensor dp) {
            std::vector<torch::Tensor> dp_unflatten(unflatten(dp));
            torch::Tensor dummy = torch::randn(m_output_shape).requires_grad_(true);
            auto Fp_dummy = torch::autograd::grad(
                {m_y},
                m_parameters,
                {dummy},
                true, // retain_graph
                true // create_graph
            );
            torch::Tensor Fp_dp = torch::zeros(m_output_shape);
            for (int i = 0; i < m_parameters.size(); ++i) {
                Fp_dp += torch::autograd::grad(
                    {Fp_dummy[i]},
                    {dummy},
                    {dp_unflatten[i]},
                    true // retain_graph
                )[0];
            }
            return Fp_dp;
        }
        torch::Tensor apply_tl_parameters_v2(torch::Tensor dp) {
            std::vector<torch::Tensor> dp_unflatten(unflatten(dp));
            torch::Tensor dummy = torch::randn(m_output_shape).requires_grad_(true);
            auto Fp_dummy = torch::autograd::grad(
                {m_y},
                m_parameters,
                {dummy},
                true, // retain_graph
                true // create_graph
            );
            auto Fp_dp_components = torch::autograd::grad(
                Fp_dummy,
                {dummy},
                dp_unflatten,
                true // retain_graph
            );
            torch::Tensor Fp_dp = torch::zeros(m_output_shape);
            for (int i = 0; i < m_parameters.size(); ++i) {
                Fp_dp += Fp_dp_components[i];
            }
            return Fp_dp;
        }

    private:
        torch::jit::Module m_model;
        Shape m_input_shape;
        Shape m_output_shape;
        std::int64_t m_num_parameters;
        std::vector<torch::Tensor> m_parameters;
        torch::Tensor m_x;
        torch::Tensor m_y;
};

int main() {
    torch::manual_seed(0);
    ScriptedModule model(
        "wdir/scripted_model.pt", // filename
        read_shape("wdir/input_shape.txt"), // input_shape
        read_shape("wdir/output_shape.txt") // output_shape
    );
    std::cout << "created scripted model" << std::endl;
    std::cout << "input_shape: " << model.get_input_shape() << std::endl;
    std::cout << "output_shape: " << model.get_output_shape() << std::endl;
    std::cout << "num_parameters: " << model.get_num_parameters() << std::endl;

    // --------------------------------------------------
    // create tensors
    torch::Tensor p = torch::randn(model.get_num_parameters());
    torch::Tensor x = torch::randn(model.get_input_shape(), torch::requires_grad());
    torch::Tensor dx = torch::randn(model.get_input_shape());
    torch::Tensor dy = torch::randn(model.get_output_shape());
    torch::Tensor dp = torch::randn(model.get_num_parameters());

    dump("wdir/p.bin", p);
    dump("wdir/x.bin", x);
    dump("wdir/dx.bin", dx);
    dump("wdir/dy.bin", dy);
    dump("wdir/dp.bin", dp);

    // --------------------------------------------------
    // reset model parameters and apply forward
    model.reset_parameters(p);
    torch::Tensor y = model.forward(x);
    dump("wdir/y.bin", y);

    // --------------------------------------------------
    // apply adjoint
    torch::Tensor FxT_dy = model.apply_ad_state(dy);
    dump("wdir/FxT_dy.bin", FxT_dy);

    torch::Tensor FpT_dy = model.apply_ad_parameters(dy);
    dump("wdir/FpT_dy.bin", FpT_dy);

    // --------------------------------------------------
    // apply tangent linear
    torch::Tensor Fx_dx = model.apply_tl_state(dx);
    dump("wdir/Fx_dx.bin", Fx_dx);

    torch::Tensor Fp_dp = model.apply_tl_parameters(dp);
    dump("wdir/Fp_dp.bin", Fp_dp);

    return 0;
}

