#include "Classifier.h"

Classifier :: Classifier(std::string modelName, const Hyperparameters& hp, int classes, double threshold) 
    : MLModel(modelName, hp), numClasses(classes), decisionThreshold(threshold) {}

int Classifier :: getNumClasses() const{
    return numClasses;
}

double Classifier :: getDecisionThreshold() const{
    return decisionThreshold;
}