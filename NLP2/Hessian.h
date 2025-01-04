/*
Copyright 2025 Harold James Krause

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the “Software”),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and /or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once
#include <Eigen/Dense>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <future>
#include <thread>

namespace NLP {

    // Base class for inheritance
    template<typename Objective>
    class BaseHessian {
    public:
        // Typedef for easier reference; identical to Objective's.
        typedef typename Objective::VectorType2 VectorType;
        typedef typename VectorType::Scalar Scalar;
        typedef Objective::MatrixType MatrixType;

        virtual MatrixType operator()(Objective& objective, const VectorType& vector) const = 0;
    protected:
        BaseHessian() = default;
    };

    template<typename Objective>
    class MachineHessian : public BaseHessian<Objective> {
    public:
        typedef typename BaseHessian<Objective>::VectorType VectorType;
        typedef typename BaseHessian<Objective>::Scalar Scalar;
        typedef typename BaseHessian<Objective>::MatrixType MatrixType;

        // Hessian methods . . . 
        // Note that these methods exploit hessian symmetry and calculate upper/lower triangular matrices, then copy 
        // into other cells to build the full hessian.
        // 
        // CENTRAL_DIFFERENCE
        // Most consistent and conventional machine derivative method.
        // Practically not much slower that forward/backward given the hessian needs f'(x +/- h) anyway.
        // f''(x) = (f'(x + h) - f'(x - h)) / (h * h)
        // 
        // BFGS (not implemented)
        //
        //

        enum Method {
            CENTRAL_DIFFERENCE
        };

        // sqrt(machine epsilon) is commonly taken as an optimal step; however it is sometimes too small.
        // 1e-06 is also commonly used; however, basing on machine epsilon is more useful for different datatypes; 
        // multiplying sqrt(machine epsilon) by 10 yields a similar number for doubles.
        // 
        // Particularly here, sqrt(machine epsilon) is much too small -- x10 fixes.
        // 
        // For a double . . . 
        // machine epsilon = 2.22045e-16
        // sqrt(machine epsilon) = 1.49012e-08
        // 10 * sqrt(machine epsilon) = 1.49012e-07

        MachineHessian(Method method = Method::CENTRAL_DIFFERENCE, Scalar epsilon = Scalar(10) * std::sqrt(std::numeric_limits<Scalar>::epsilon()))
            : BaseHessian<Objective>(), method_(method), epsilon_(epsilon) {
            if (epsilon_ <= Scalar(0)) { //Negative or zero epsilon doesn't make sense
                throw std::invalid_argument("Epsilon must be positive.");
            }
        }

        MatrixType operator()(Objective& objective, const VectorType& vector) const override {
            switch (method_) {
            case CENTRAL_DIFFERENCE:
                return centralDifference(objective, vector);
            default:
                throw std::invalid_argument("Unknown or not-implemented method."); //Throw explicit error
            }
        }

    private:
        Method method_; // Hessian method
        Scalar epsilon_; // Epsilon for finite differences

        MatrixType centralDifference(Objective& objective, const VectorType& vector) const {
            MatrixType hessian = MatrixType::Zero(vector.size(), vector.size());
            for (int i = 0; i < vector.size(); ++i) {
                // j <= i to calculate lower triangular and exploit symmetry
                for (int j = 0; j <= i; ++j) {
                    // If i == j then we are calculating symmetric derivatives and can use the symmetric second-order formula
                    // All of the function calls are reused by the gradient, making it very fast to evaluate via lookup
                    // https://en.wikipedia.org/wiki/Symmetric_derivative 
                    // f(x + h) - 2f(x) + f(x - h) / (h * h)
                    if (i == j) {
                        VectorType x_up = vector, x_down = vector;
                        x_up(i) += epsilon_;
                        x_down(i) -= epsilon_;

                        //Runtime determ whether more threads needed
                        std::future<Scalar> f_up = std::async(std::launch::async | std::launch::deferred, [&]() {return objective(x_up); });
                        std::future<Scalar> f_down = std::async(std::launch::async | std::launch::deferred, [&]() {return objective(x_down); });
                        std::future<Scalar> f_center = std::async(std::launch::async | std::launch::deferred, [&]() {return objective(vector); });

                        hessian(i, i) = (f_up.get() - 2 * f_center.get() + f_down.get()) / (epsilon_ * epsilon_);
                    }
                    else {
                        // Need to calculate full second-order partials as
                        // x up i, up j; x up i, down j; etc
                        VectorType x_up_up = vector;
                        VectorType x_down_down = vector;
                        VectorType x_up_down = vector;
                        VectorType x_down_up = vector;

                        // Perturbations
                        x_up_up(i) += epsilon_; 
                        x_up_up(j) += epsilon_;
                        x_down_down(i) -= epsilon_; 
                        x_down_down(j) -= epsilon_;
                        x_up_down(i) += epsilon_; 
                        x_up_down(j) -= epsilon_;
                        x_down_up(i) -= epsilon_; 
                        x_down_up(j) += epsilon_;

                        //Runtime determ whether more threads needed
                        // Evaluate f and compute second-order partial in upper/lower triangular
                        std::future<Scalar> f_up_up = std::async(std::launch::async | std::launch::deferred, [&]() {return objective(x_up_up); });
                        std::future<Scalar> f_down_down = std::async(std::launch::async | std::launch::deferred, [&]() {return objective(x_down_down); });
                        std::future<Scalar> f_up_down = std::async(std::launch::async | std::launch::deferred, [&]() {return objective(x_up_down); });
                        std::future<Scalar> f_down_up = std::async(std::launch::async | std::launch::deferred, [&]() {return objective(x_down_up); });

                        hessian(i, j) = (f_up_up.get() - f_up_down.get() - f_down_up.get() + f_down_down.get()) / (4 * epsilon_ * epsilon_);
                        hessian(j, i) = hessian(i, j); // Exploit symmetry
                    }
                }
            }
            return hessian;
        }
    };
}
