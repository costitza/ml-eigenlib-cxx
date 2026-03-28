#pragma once
#include "MLModel.h"

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