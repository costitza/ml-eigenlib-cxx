#pragma once 

// simple class to hold the hyperparameters for the models, such as the number of input features,
// the learning rate and the number of epochs for training

class Hyperparameters{
    int inputFeatures;
    double learningRate;
    int epochs;
    
public:
    Hyperparameters(int feat = 1, double lr = 0.001, int eps = 10) 
        : inputFeatures(feat), learningRate(lr), epochs(eps) {}
};