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
