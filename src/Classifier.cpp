#include "Classifier.h"

Classifier :: Classifier(std::string modelName, const Hyperparameters& hp, int classes, double threshold) 
    : MLModel(modelName, hp), numClasses(classes), decisionThreshold(threshold) {}

Classifier :: etNumClasses() const{
    return numClasses;
}

double getDecisionThreshold() const{
    return decisionThreshold;
}