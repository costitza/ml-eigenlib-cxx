#include "MLModel.h"

// the other methods are pure virtual
// they will be implemented in the derived classes
// (LinearRModel, LogicRModel and KNNModel)


// constr
MLModel :: MLModel(std::string modelName, const Hyperparameters& hp)
    : name(modelName), isTrained(false), params(hp) {}


// methods
bool MLModel :: getIsTrained() const{
    return isTrained;
}

Hyperparameters MLModel :: getHyperparameters() const{
    return params;
}

std :: string MLModel :: getName() const{
    return name;
}

void MLModel :: setIsTrained(const bool a){
        this -> isTrained = a;
    }
