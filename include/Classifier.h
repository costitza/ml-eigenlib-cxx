#pragma once
#include "MLModel.h"

// model for classification derived from MLModel
// the predict method will return the class with the highest probability for classification
// the train method will use the logistic regression algorithm to calculate the weights and the bias for classification
// (logic will be implemented in the LogicRModel class, which will be derived from Classifier)

class Classifier : virtual public MLModel{
    int numClasses;
    double decisionThreshold;

public: 
    Classifier () : MLModel(), numClasses(2), decisionThreshold(0.5) {}
    Classifier(std::string modelName, const Hyperparameters& hp, int classes = 2, double threshold = 0.5);
    

    virtual ~Classifier() = default;

    // methods
    int getNumClasses() const{
        return numClasses;
    }

    double getDecisionThreshold() const{
        return decisionThreshold;
    }

    virtual void printConfusionMatrix() const = 0;

    // getters / setters but only if needed
    int getNumClasses() const;
    double getDecisionThreshold() const;
    
};