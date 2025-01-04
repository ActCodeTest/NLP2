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
#include <Eigen/dense>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <future>
#include <thread>

namespace NLP {

	// Base class for inheritance
	template<typename Objective>
	class BaseGradient {
	public:
		// Typedef for easier reference; identical to Objective's.
		typedef typename Objective::VectorType2 VectorType;
		typedef typename VectorType::Scalar Scalar;

		virtual VectorType operator()(Objective& objective, const VectorType& vector) const = 0;
	protected:
		BaseGradient() = default;
	};

	template<typename Objective>
	class MachineGradient : public BaseGradient<Objective> {
	public:
		// Typedef for easier reference; identical to Objective's.
		typedef typename BaseGradient<Objective>::VectorType VectorType;
		typedef typename BaseGradient<Objective>::Scalar Scalar;

		// Gradient methods . . . 
		// CENTRAL_DIFFERENCE
		// Most consistent and conventional machine derivative method.
		// Practically not much slower that forward/backward given the hessian needs f(x +/- h) anyway.
		// df/dx = (f(x + h) - f(x - h)) / 2h
		//
		// Other methods ???
		//

		enum Method {
			CENTRAL_DIFFERENCE
		};

		// sqrt(machine epsilon) is commonly taken as an optimal step; however it is sometimes too small.
		// 1e-06 is also commonly used; however, basing on machine epsilon is more useful for different datatypes; 
		// multiplying sqrt(machine epsilon) by 10 yields a similar number for doubles.
		// 
		// For a double . . . 
		// machine epsilon = 2.22045e-16
		// sqrt(machine epsilon) = 1.49012e-08
		// 10 * sqrt(machine epsilon) = 1.49012e-07

		MachineGradient(const Method& method = Method::CENTRAL_DIFFERENCE, const Scalar& epsilon = Scalar(10) * std::sqrt(std::numeric_limits<Scalar>::epsilon())) :
			BaseGradient<Objective>(), method_(method), epsilon_(epsilon) {
			if (epsilon_ <= Scalar(0)) { //Negative or zero epsilon doesn't make sense
				throw std::invalid_argument("Epsilon must be positive.");
			}
		}

		VectorType operator()(Objective& objective, const VectorType& vector) const override {
			switch (method_)
			{
			case CENTRAL_DIFFERENCE:
				return centralDifference(objective, vector);
			default:
				throw std::invalid_argument("Unknown or not-implemented method."); //Throw explicit error
				break;
			}
		}
	private:
		Method method_; //Gradient method
		Scalar epsilon_; //Epsilon for finite differences

		// CENTAL_DIFFERENCE
		VectorType centralDifference(Objective& objective, const VectorType& vector) const {
			VectorType gradient(vector.size()); //ResultsAA

			//Cache up/down x vectors and reuse in the loop 
			VectorType x_up = vector;
			VectorType x_down = vector;

			for (int i = 0; i < vector.size(); i++) {
				// Perturb x(i) up or down
				x_up(i) += epsilon_;
				x_down(i) -= epsilon_;

				//Runtime determ whether more threads needed
				std::future<Scalar> f_up = std::async(std::launch::async | std::launch::deferred, [&]() {return objective(x_up); });
				std::future<Scalar> f_down = std::async(std::launch::async | std::launch::deferred, [&]() {return objective(x_down); });

				gradient(i) = (f_up.get() - f_down.get()) / (Scalar(2) * epsilon_);

				// Reset perturbation before next loop
				x_up(i) = vector(i);
				x_down(i) = vector(i);
			}
			return gradient;
		}
	};
}