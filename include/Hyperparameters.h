#pragma once 
#include <nlohmann/json.hpp>

using json = nlohmann :: json;

// simple class to hold the hyperparameters for the models, such as the number of input features,
// the learning rate and the number of epochs for training

class Hyperparameters{
    int inputFeatures;
    double learningRate;
    int epochs;
    
public:
    Hyperparameters(int feat = 1, double lr = 0.001, int eps = 10);
    
    // getters
    int getInputFeatures() const;
    double getLearningRate() const;
    int getEpochs() const;

    // methods for json
    json serialize() const;
    void deserialize(const json& j);
};