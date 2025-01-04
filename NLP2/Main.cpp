#include <iostream>

#include "Objective.h"
#include "Gradient.h"
#include "Hessian.h"
#include "Optimizer.h"
int someFunction() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return 42;
}

int anotherFunction() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return 24;
}

double func(const Eigen::Matrix<double, 2, 1>& x0) {
    Eigen::Matrix<double, 2, 1> x = x0;
    for (auto i = 0; i < 2; i++) {
        x[0] += x.squaredNorm();
        x[1] += x.squaredNorm();
    }
    return sqrt(x.sum());
}

int main() {


    NLP::Objective<Eigen::Matrix<double, 2, 1>> obj(func);

    auto grad = std::make_shared<NLP::MachineGradient<NLP::Objective<Eigen::Matrix<double, 2, 1>>>>();
    auto hess = std::make_shared<NLP::MachineHessian<NLP::Objective<Eigen::Matrix<double, 2, 1>>>>();

    Eigen::Matrix<double, 2, 1> x;
    x << 10, 50;
    
    NLP::TrustRegionOptimizer< NLP::Objective<Eigen::Matrix<double, 2, 1>>> trm(grad, hess, true);

    std::cout << trm.minimize(obj, x);

    //std::cout << "Result: " << x + y << std::endl;

    //pool.join(); // Wait for all tasks to finish
    return 0;
}
