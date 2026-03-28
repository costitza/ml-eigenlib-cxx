#pragma once 
#include "MLModel.h"

class Regressor : virtual public MLModel{
    double l2Penalty;

public:
    Regressor () : MLModel(), l2Penalty(0.0) {}
    Regressor(std::string modelName, const Hyperparameters& hp, double l2 = 0.0) ;
    
    virtual ~Regressor() = default;

    // methods
    double getL2Penalty() const{
        return l2Penalty;
    }

    virtual double getMSE() const = 0;

};