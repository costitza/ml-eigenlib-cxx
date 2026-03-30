#include "Hyperparameters.h"

Hyperparameters :: Hyperparameters(int feat, double lr, int eps) 
    : inputFeatures(feat), learningRate(lr), epochs(eps) {}


int Hyperparameters :: getInputFeatures() const {
    return inputFeatures;
}

double Hyperparameters :: getLearningRate() const {
    return learningRate;
}

int Hyperparameters :: getEpochs() const {
    return epochs;
}


json Hyperparameters::serialize() const {
    json j;
    j["inputFeatures"] = inputFeatures;
    j["learningRate"] = learningRate;
    j["epochs"] = epochs;
    return j;
}

void Hyperparameters::deserialize(const json& j) {
    // .value() safely checks if the key exists
    inputFeatures = j.value("inputFeatures", 5);
    learningRate = j.value("learningRate", 0.01);
    epochs = j.value("epochs", 100);
}
