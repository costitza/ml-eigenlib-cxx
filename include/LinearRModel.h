#pragma once

#include "Regressor.h"
#include <Eigen/Dense>

// model for linear regression derived from Regressor
// the predict method will return the dot product of the input and the weigths
// the train method will use the normal equation to calculate the weights

// the normal equation is given by: w = (X^T * X + l2Penalty * I)^(-1) * X^T * y
// where X is the input data, y is the target values, l2Penalty is the regularization term and I is the identity matrix

// eigen vector Xd will be used to store the weights and the bias will be stored in a separate variable


class LinearRModel : public Regressor{
    Eigen :: VectorXd weights;
    double bias;

public:
    LinearRModel() : Regressor(), bias(0.0) {}
    LinearRModel(std::string modelName, const Hyperparameters& hp, double l2 = 0.0, double b = 0.0)
        : Regressor(modelName, hp, l2), bias(b) {}

    virtual ~LinearRModel() = default;

    // methods
    void train(const Dataset& data) override;
    double predict(const Eigen :: VectorXd& input) override;

    // save / load to json
    json serialize() const override;
    void deserialize(const json& j) override;

    double getMSE() const override;

    // getters / setters
    Eigen :: VectorXd getWeights() const{
        return weights;
    }

    double getBias() const{
        return bias;
    }

};