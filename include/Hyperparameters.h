#pragma once 

class Hyperparameters{
    int inputFeatures;
    double learningRate;
    int epochs;
    
public:
    Hyperparameters(int feat = 1, double lr = 0.001, int eps = 10) 
        : inputFeatures(feat), learningRate(lr), epochs(eps) {}
};