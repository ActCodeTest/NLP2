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
#include <memory>
#include <Eigen/dense>
#include <cmath>
#include <limits>
#include "Gradient.h"
#include "Hessian.h"
#include <iostream>

namespace NLP {
	template<typename Objective>
	class BaseOptimizer {
	public:
		typedef Objective::VectorType2 VectorType;
		typedef VectorType::Scalar Scalar;
		typedef Objective::MatrixType MatrixType;

		virtual bool minimize(Objective& f, VectorType& x) const = 0;

		Scalar evaluate(Objective& f, const VectorType& x) const {
			return f(x);
		}
		VectorType gradient(Objective& f, const VectorType& x) const {
			return (*gradient_)(f, x);
		}
		MatrixType hessian(Objective& f, const VectorType& x) const {
			return (*hessian_)(f, x);
		}
		void setDebugMode(const bool& debug_mode) {
			debug_mode_ = debug_mode;
		}
	protected:
		BaseOptimizer(
			const std::shared_ptr<BaseGradient<Objective>>& gradient, 
			const std::shared_ptr<BaseHessian<Objective>>& hessian,
			const bool& debug_mode = false) : 
			gradient_(gradient), hessian_(hessian), debug_mode_(debug_mode) {}

		template<typename... Args>
		void debugPrint(const Args&... args) const {
			if (debug_mode_) {
				(std::cout << ... << args) << std::endl; // Fold expression for printing all args
			}
		}
	private:
		std::shared_ptr<BaseGradient<Objective>> gradient_;
		std::shared_ptr<BaseHessian<Objective>> hessian_;
		bool debug_mode_;
	};

	template<typename Objective>
	class GradientDescent : public BaseOptimizer<Objective> {
	public:
		typedef typename BaseOptimizer<Objective>::VectorType VectorType;
		typedef typename BaseOptimizer<Objective>::Scalar Scalar;
		typedef typename BaseOptimizer<Objective>::MatrixType MatrixType;

		GradientDescent(
			const std::shared_ptr<BaseGradient<Objective>>& gradient,
			const std::shared_ptr<BaseHessian<Objective>>& hessian,
			const bool& debug_mode = false,
			const Scalar& learning_rate = Scalar(0.01),
			const int& max_iterations = 1000,
			const Scalar& tolerance = Scalar(100) * std::sqrt(std::numeric_limits<Scalar>::epsilon())) : 
			BaseOptimizer<Objective>(gradient, hessian, debug_mode), learning_rate_(learning_rate), max_iterations_(max_iterations), tolerance_(tolerance) {}

		bool minimize(Objective& f, VectorType& x) const override {
			for (int i = 0; i < max_iterations_; i++) {
				VectorType grad = this->gradient(f, x);

				//Debug iteration status / values
				this->debugPrint("Iteration:\t", i, "\nFunction:\t", this->evaluate(f, x), "\nVector X:\t", x.transpose(), "\nGradient:\t", grad.transpose(), "\nGrad Norm:\t", grad.norm(), "\n");

				x -= learning_rate_ * grad;

				if (grad.norm() < tolerance_) {
					return true;
				}
			}
			return false;
		}

	private:
		Scalar learning_rate_;
		int max_iterations_;
		Scalar tolerance_;
	};

    template<typename Objective>
    class TrustRegionOptimizer : public BaseOptimizer<Objective> {
    public:
        typedef typename BaseOptimizer<Objective>::VectorType VectorType;
        typedef typename BaseOptimizer<Objective>::Scalar Scalar;
        typedef typename BaseOptimizer<Objective>::MatrixType MatrixType;

        TrustRegionOptimizer(
            const std::shared_ptr<BaseGradient<Objective>>& gradient,
            const std::shared_ptr<BaseHessian<Objective>>& hessian,
            const bool& debug_mode = false,
            const Scalar& initial_tr_radius = Scalar(1.0),
            const Scalar& max_tr_radius = Scalar(100.0),
            const Scalar& tolerance = Scalar(100) * std::sqrt(std::numeric_limits<Scalar>::epsilon()))
            : BaseOptimizer<Objective>(gradient, hessian, debug_mode),
            initial_tr_radius_(initial_tr_radius),
            max_tr_radius_(max_tr_radius),
            tolerance_(tolerance) {}

        bool minimize(Objective& f, VectorType& x) const override {
            Scalar trust_radius = initial_tr_radius_;
            for (int i = 0; i < max_iterations_; ++i) {
                VectorType grad = this->gradient(f, x);
                MatrixType hess = this->hessian(f, x);

                // Debugging information
                this->debugPrint("Iteration:\t", i, "\nFunction:\t", this->evaluate(f, x),
                    "\nVector X:\t", x.transpose(), "\nGradient:\t", grad.transpose(),
                    "\nGrad Norm:\t", grad.norm(), "\nTrust Radius:\t", trust_radius);

                // Solve the trust-region subproblem
                VectorType step = computeStep(grad, hess, trust_radius);

                Scalar pred_reduction = predictedReduction(grad, hess, step);
                Scalar act_reduction = this->evaluate(f, x) - this->evaluate(f, x + step);

                Scalar rho = act_reduction / pred_reduction;

                // Update trust region radius
                if (rho < 0.25) {
                    trust_radius *= 0.25;
                }
                else if (rho > 0.75 && step.norm() == trust_radius) {
                    trust_radius = std::min(2.0 * trust_radius, max_tr_radius_);
                }

                // Accept or reject the step
                if (rho > 0.0) {
                    x += step;
                }

                if (grad.norm() < tolerance_) {
                    return true;
                }
            }
            return false;
        }

    private:
        VectorType computeStep(const VectorType& grad, const MatrixType& hess, Scalar trust_radius) const {
            // Solve the trust region subproblem approximately using the dogleg method or truncated CG
            Eigen::LLT<MatrixType> cholesky(hess);
            if (cholesky.info() == Eigen::Success) {
                VectorType step = cholesky.solve(-grad);
                if (step.norm() <= trust_radius) {
                    return step;
                }
            }

            // Fall back to truncated CG if Cholesky fails or exceeds trust region
            return truncatedCG(grad, hess, trust_radius);
        }

        VectorType truncatedCG(const VectorType& grad, const MatrixType& hess, Scalar trust_radius) const {
            VectorType p = VectorType::Zero(grad.size());
            VectorType r = -grad;
            VectorType d = r;
            Scalar delta_sq = trust_radius * trust_radius;
            Scalar rTr = r.dot(r);

            while (rTr > tolerance_) {
                Scalar alpha = rTr / (d.dot(hess * d));
                if ((p + alpha * d).squaredNorm() > delta_sq) {
                    Scalar tau = (-p.dot(d) + std::sqrt(std::pow(p.dot(d), 2) + delta_sq * d.dot(d))) / d.dot(d);
                    return p + tau * d;
                }
                p += alpha * d;
                VectorType r_next = r - alpha * hess * d;
                Scalar rTr_next = r_next.dot(r_next);
                Scalar beta = rTr_next / rTr;
                d = r_next + beta * d;
                r = r_next;
                rTr = rTr_next;
            }
            return p;
        }

        Scalar predictedReduction(const VectorType& grad, const MatrixType& hess, const VectorType& step) const {
            return -grad.dot(step) - 0.5 * step.dot(hess * step);
        }

        Scalar initial_tr_radius_;
        Scalar max_tr_radius_;
        Scalar tolerance_;
        int max_iterations_ = 1000;  // Default maximum iterations
    };

}