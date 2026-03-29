#include "Regressor.h"


Regressor :: Regressor(std::string modelName, const Hyperparameters& hp, double l2) 
    : MLModel(modelName, hp), l2Penalty(l2) {}

double Regressor :: getL2Penalty() const{
    return l2Penalty;
}

double Regressor :: getMSE(const Dataset& data) const{
    if(!this -> getIsTrained()) return -1.0;

    int n = data.getRows();
    double mse = 0.0;

    for (int i = 0;i < n; i++){
        Eigen :: VectorXd row = data.getRowsAsEigen(i);
        double y_true = data.getLabel(i);
        double y_pred = predict(row);

        mse += (y_true - y_pred) * (y_true - y_pred);
    }

    return mse / n;
}