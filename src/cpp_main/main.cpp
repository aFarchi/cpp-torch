#include <torch/torch.h>
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

// wrapper around scripted module
class ScriptedModule {
    public:
        using Shape = std::vector<std::int64_t>;
        ScriptedModule(std::string filename) :
            m_model(torch::jit::load(filename)),
            m_input_shape(),
            m_output_shape(),
            m_num_parameters(0) {
                for (const auto& p : m_model.parameters())
                {
                    m_parameters.push_back(p);
                    m_num_parameters += p.numel();
                }
                for (const auto& buffer : m_model.named_buffers())
                {
                    if (buffer.name == "in_x")
                    {
                        Shape shape = buffer.value.sizes().vec();
                        m_input_shape = std::move(shape);
                    }
                    else if (buffer.name == "out_forward")
                    {
                        Shape shape = buffer.value.sizes().vec();
                        m_output_shape = std::move(shape);
                    }
                }
            }
        const Shape & get_input_shape() const { return m_input_shape; }
        const Shape & get_output_shape() const { return m_output_shape; }
        std::int64_t get_num_parameters() const { return m_num_parameters; }
        torch::Tensor get_buffer(std::string name) const {
            for (const auto& buffer : m_model.named_buffers())
            {
                if (buffer.name == name)
                {
                    return buffer.value;
                }
            }
            return torch::Tensor();
        }
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

// print first n elements of a tensor
void print_first_n(std::string name, const torch::Tensor tensor, int n_first) {
    torch::Tensor flat = tensor.detach().to(torch::kCPU).flatten();
    int64_t count = std::min<int64_t>(n_first, flat.numel());
    std::cout << name << " ";
    for (int64_t i = 0; i < count; ++i) {
        std::cout << flat[i].item<double>() << " ";
    }
    std::cout << std::endl;
}

// print the absolute or relative difference between the first n elements of two tensors
void print_diff_first_n(
    std::string name,
    const torch::Tensor predicted,
    const torch::Tensor expected,
    int n_first,
    bool relative
) {
    torch::Tensor flat_predicted = predicted.detach().to(torch::kCPU).flatten();
    torch::Tensor flat_expected = expected.detach().to(torch::kCPU).flatten();
    int64_t count_1 = std::min<int64_t>(n_first, flat_predicted.numel());
    int64_t count_2 = std::min<int64_t>(n_first, flat_expected.numel());
    int64_t count = std::min<int64_t>(count_1, count_2);
    std::cout << name << " ";
    for (int64_t i = 0; i < count; ++i) {
        double p = flat_predicted[i].item<double>();
        double e = flat_expected[i].item<double>();
        if (relative)
        {
            double error = abs(p - e)/abs(e);
            std::cout << error << " ";
        }
        else
        {
            double error = abs(p - e);
            std::cout << error << " ";
        }
    }
    std::cout << std::endl;
}

// check if two tensors are (element-wise) close given absolute and relative tolerance
bool all_close(const torch::Tensor predicted, const torch::Tensor expected, double atol, double rtol) {
    torch::Tensor flat_predicted = predicted.detach().to(torch::kCPU).flatten();
    torch::Tensor flat_expected = expected.detach().to(torch::kCPU).flatten();
    if ( flat_predicted.numel() != flat_expected.numel() )
    {
        return false;
    }
    for (int64_t i = 0; i < flat_predicted.numel(); ++i) {
        double p = flat_predicted[i].item<double>();
        double e = flat_expected[i].item<double>();
        if ( abs(p - e) > atol + rtol * abs(e) )
        {
            return false;
        }
    }
    return true;
}

// compute the maximum absolute difference between two tensors
double max_abs_diff(const torch::Tensor predicted, const torch::Tensor expected) {
    torch::Tensor flat_predicted = predicted.detach().to(torch::kCPU).flatten();
    torch::Tensor flat_expected = expected.detach().to(torch::kCPU).flatten();
    if ( flat_predicted.numel() != flat_expected.numel() )
    {
        return -1;
    }
    double maxi = 0;
    for (int64_t i = 0; i < flat_predicted.numel(); ++i) {
        double p = flat_predicted[i].item<double>();
        double e = flat_expected[i].item<double>();
        maxi = std::max<double>(maxi, abs(p-e));
    }
    return maxi;
}

// compute the maximum relative difference between two tensors
double max_rel_diff(const torch::Tensor predicted, const torch::Tensor expected) {
    torch::Tensor flat_predicted = predicted.detach().to(torch::kCPU).flatten();
    torch::Tensor flat_expected = expected.detach().to(torch::kCPU).flatten();
    if ( flat_predicted.numel() != flat_expected.numel() )
    {
        return -1;
    }
    double maxi = 0;
    for (int64_t i = 0; i < flat_predicted.numel(); ++i) {
        double p = flat_predicted[i].item<double>();
        double e = flat_expected[i].item<double>();
        maxi = std::max<double>(maxi, abs(p-e)/abs(e));
    }
    return maxi;
}

// compare tensors
void compare_tensors(
    std::string title,
    torch::Tensor predicted,
    torch::Tensor expected,
    int n_first,
    double atol,
    double rtol
) {

    std::cout << std::endl << "========================================" << std::endl;
    std::cout << std::endl << "Errors on " << title << std::endl << std::endl;
    print_first_n("predicted:", predicted, n_first);
    print_first_n("expected: ", expected, n_first);
    print_diff_first_n("abs. diff:", predicted, expected, n_first, false);
    print_diff_first_n("rel. diff:", predicted, expected, n_first, true);
    std::cout << "maximum of abs. diff. = " << max_abs_diff(predicted, expected) << std::endl;
    std::cout << "maximum of rel. diff. = " << max_rel_diff(predicted, expected) << std::endl;
    if ( all_close(predicted, expected, atol, rtol) )
    {
        std::cout << "unittest passed with atol = " << atol << " and rtol = " << rtol << std::endl;
    }
    else
    {
        std::cout << "<!> unittest failed with atol = " << atol << " and rtol = " << rtol << std::endl;
    }
}

// compute dot product (manually)
double dot_product(torch::Tensor tensor_1, torch::Tensor tensor_2) {
    torch::Tensor flat_1 = tensor_1.detach().to(torch::kCPU).flatten();
    torch::Tensor flat_2 = tensor_2.detach().to(torch::kCPU).flatten();
    if ( flat_1.numel() != flat_2.numel() )
    {
        return -1;
    }
    double dot = 0;
    for (int64_t i = 0; i < flat_1.numel(); ++i) {
        double e_1 = flat_1[i].item<double>();
        double e_2 = flat_2[i].item<double>();
        dot += e_1 * e_2;
    }
    return dot;
}

// compute adjoint test
void adjoint_test(
    torch::Tensor dp,
    torch::Tensor dx,
    torch::Tensor dy,
    torch::Tensor output_ad_p,
    torch::Tensor output_ad_x,
    torch::Tensor output_tl_p,
    torch::Tensor output_tl_x
) {
    double dot_1 = dot_product(
        output_ad_p,
        dp
    ) + dot_product(
        output_ad_x,
        dx
    );
    double dot_2 = dot_product(
        output_tl_x,
        dy
    ) + dot_product(
        output_tl_p,
        dy
    );
    double abs_diff = abs(dot_1 - dot_2);
    double rel_diff = 2 * abs_diff / (abs(dot_1) + abs(dot_2));

    std::cout << std::endl << "========================================" << std::endl;
    std::cout << std::endl << "Adjoint test:" << std::endl << std::endl;
    std::cout << "<F^T_p dy | dp > + <F^T_x dy | dx > = " << dot_1 << std::endl;
    std::cout << "<F_p dp | dy > + <F_x dx | dy >     = " << dot_2 << std::endl;
    std::cout << "abs. diff.                          = " << abs_diff << std::endl;
    std::cout << "rel. diff.                          = " << rel_diff << std::endl;
}

int main() {
    torch::manual_seed(0);

    // --------------------------------------------------
    // read scripted module
    ScriptedModule model(
        "wdir/scripted_model.pt" // filename
    );
    std::cout << "created scripted model" << std::endl;
    std::cout << "num_parameters: " << model.get_num_parameters() << std::endl;
    std::cout << "input_shape: " << model.get_input_shape() << std::endl;
    std::cout << "output_shape: " << model.get_output_shape() << std::endl;

    torch::Tensor p = model.get_buffer("in_p");
    torch::Tensor x = model.get_buffer("in_x");
    torch::Tensor dx = model.get_buffer("in_dx");
    torch::Tensor dy = model.get_buffer("in_dy");
    torch::Tensor dp = model.get_buffer("in_dp");

    // --------------------------------------------------
    // reset model parameters and apply forward
    model.reset_parameters(p);
    torch::Tensor y = model.forward(x);
    torch::Tensor y_py = model.get_buffer("out_forward");
    compare_tensors("forward", y, y_py, 5, 0, 1e-5);

    // --------------------------------------------------
    // apply adjoint
    torch::Tensor FxT_dy = model.apply_ad_state(dy);
    torch::Tensor FpT_dy = model.apply_ad_parameters(dy);
    torch::Tensor FxT_dy_py = model.get_buffer("out_ad_x");
    torch::Tensor FpT_dy_py = model.get_buffer("out_ad_p");
    compare_tensors("adjoint wrt state", FxT_dy, FxT_dy_py, 5, 0, 1e-5);
    compare_tensors("adjoint wrt parameters", FpT_dy, FpT_dy_py, 5, 0, 1e-5);

    // --------------------------------------------------
    // apply tangent linear
    torch::Tensor Fx_dx = model.apply_tl_state(dx);
    torch::Tensor Fp_dp = model.apply_tl_parameters(dp);
    torch::Tensor Fx_dx_py = model.get_buffer("out_tl_x");
    torch::Tensor Fp_dp_py = model.get_buffer("out_tl_p");
    compare_tensors("TL wrt state", Fx_dx, Fx_dx_py, 5, 0, 1e-5);
    compare_tensors("TL wrt parameters", Fp_dp, Fp_dp_py, 5, 0, 1e-5);

    // --------------------------------------------------
    // compute adjoint test
    adjoint_test(dp, dx, dy, FpT_dy_py, FxT_dy_py, Fp_dp_py, Fx_dx_py);

    return 0;
}
