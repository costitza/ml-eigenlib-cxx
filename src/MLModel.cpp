#include "MLModel.h"
#include <random>
#include <iomanip>
#include <sstream>

// the other methods are pure virtual
// they will be implemented in the derived classes
// (LinearRModel, LogicRModel and KNNModel)

// static methods and static variables initialization
int MLModel :: totalModels = 0;

// generate random string for id of each model
std :: string MLModel :: generateRandomID(){

    std :: random_device rd;
    std :: mt19937 gen(rd());

    // random number (1000 - FFFF in hex)
    std :: uniform_int_distribution<> dis(4096, 65535);

    // mod-XXXX
    std :: stringstream ss;
    ss << "mod-" << std :: uppercase << std :: hex << dis(gen);
    return ss.str();
}

int MLModel :: getTotalModels(){
    return totalModels;
}


// constr
MLModel :: MLModel(std::string modelName, const Hyperparameters& hp)
    : name(modelName), isTrained(false), params(hp) {
        totalModels ++;

        modelID = MLModel :: generateRandomID();
    }





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

std :: string MLModel :: getModelID() const{
    return modelID;
}





