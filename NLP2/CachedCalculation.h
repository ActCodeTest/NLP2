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
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace NLP {

	// Hashing function for an Eigen::Vector.
	// Note it is oblivious to row/column major storage order and will give the same hash value for 
	// transposed matrices in different storage order; this is not an issue because the VectorType is templated between
	// all classes, but it does mean that this cannot be generalized very well.
	template<typename VectorType>
	struct eigen_vector_hash {
		std::size_t operator()(const VectorType& vector) const {
			size_t seed = 0;
			for (size_t i = 0; i < vector.size(); ++i) {
				auto elem = *(vector.data() + i);
				seed ^= std::hash<typename VectorType::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			}
			return seed;
		}
	};
	
	// Cached calculation class; basically a wrapped thread-safe std::unordered_map.
	// Used for objectives to cache results when evaluation is costly.
	//
	// Stores the actual objective function and the Objective class wraps about it.
	template<typename VectorType>
	class CachedCalculation {
	public:
		typedef typename VectorType::Scalar Scalar;

		CachedCalculation(const std::function<Scalar(const VectorType&)>& function) : function_(function) {}

		// If the function has been evaluated for some vector, return the already-calculated value.
		// Else, calculate the new vector and store the results.
		Scalar operator()(const VectorType& vector) {
			{
				// Shared lock to allow concurrent read operations on the cache
				std::shared_lock<std::shared_mutex> lock(mtx_);
				auto it = cache_.find(vector);
				if (it != cache_.end()) {
					return it->second;
				}
			}
			// No lock on function evaluation to allow multi-threading
			Scalar value = function_(vector);
			{
				// Unique lock to block all actions when writing
				std::unique_lock<std::shared_mutex> lock(mtx_);
				cache_[vector] = value;
			}
			return value;
		}


	private:
		mutable std::shared_mutex mtx_; // Shared mutex for concurrent read access
		std::function<Scalar(const VectorType&)> function_; // Function to evaluate
		std::unordered_map<VectorType, Scalar, eigen_vector_hash<VectorType>> cache_; // Cached results
	};
}

