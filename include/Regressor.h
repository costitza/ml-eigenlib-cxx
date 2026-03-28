#pragma once 
#include "MLModel.h"
#include "Dataset.h"

// model for regression derived from MLModel
// the predict method will return the dot product of the input and the weights for regression
// the train method will use the normal equation to calculate the weights for regression
// (logic will be implemented in the LinearRModel class, which will be derived from Regressor)

class Regressor : virtual public MLModel{
protected:
    double l2Penalty;

public:
    Regressor () : MLModel(), l2Penalty(0.0) {}
    Regressor(std::string modelName, const Hyperparameters& hp, double l2 = 0.0) ;
    
    virtual ~Regressor() = default;

    // methods
    double getL2Penalty() const;

    virtual double getMSE(const Dataset& data) const = 0;

};