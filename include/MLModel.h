#pragma once
#include <string>
#include "Hyperparameters.h"
#include "Dataset.h"

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using json = nlohmann :: json;


class MLModel{
    std :: string name;
    bool isTrained;
    Hyperparameters params;

public:
    MLModel() {}
    MLModel(std::string modelName, const Hyperparameters& hp);

    virtual ~MLModel() = default;

    // methods
    virtual void train(const Dataset& data) = 0;
    virtual double predict(const Eigen :: VectorXd& input) = 0;

    // save / load to json
    virtual json serialize() const = 0;
    virtual void deserialize(const json& j) = 0;

    // getters / setters
    std :: string getName() const{
        return name;
    }

    bool getIsTrained() const;

    Hyperparameters getHyperparameters() const;
};