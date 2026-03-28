#include "Regressor.h"


Regressor :: Regressor(std::string modelName, const Hyperparameters& hp, double l2) 
    : MLModel(modelName, hp), l2Penalty(l2) {}

double Regressor :: getL2Penalty() const{
    return l2Penalty;
}
