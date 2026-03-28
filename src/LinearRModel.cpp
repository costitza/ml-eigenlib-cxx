#include "LinearRModel.h"
#include <vector>

LinearRModel :: LinearRModel(std::string modelName, const Hyperparameters& hp, double l2 = 0.0, double b = 0.0)
    : Regressor(modelName, hp, l2), bias(b) {
    weights = Eigen :: VectorXd :: Zero(hp.getInputFeatures());
}

// Normal Equation
void LinearRModel :: train(const Dataset& data){
    int n = data.getRows();
    int f = data.getCols();

    // f + 1 to calculate weigths + bias at the same time
    Eigen :: MatrixXd X(n, f + 1);
    Eigen :: VectorXd Y(n);

    for (int i = 0;i < n;i++){
        // get row as eigen vector
        Eigen :: VectorXd row = data.getRowsAsEigen(i);

        for (int j = 0; j < f; j++){
            X(i, j) = row(j);
        }
        // bias column
        X(i, f) = 1.0;

        Y(i) = data.getLabel(i);
    }

    // regularization matrix (identity * l2penalty)
    Eigen :: MatrixXd I = Eigen :: MatrixXd :: Identity(f + 1, f + 1) * this->getL2Penalty();
    I(f, f) = 0.0;


    // w = (Xt * X + I) ^ (-1) * Xt * Y
    Eigen :: MatrixXd XT = X.transpose();
    Eigen :: MatrixXd parantheses = XT * X + I;

    Eigen :: VectorXd weights_bias = parantheses.inverse() * X * Y;

    weights = weights_bias.head(f); // first f elements
    bias = weights_bias(f);
    
    this->setIsTrained(true);
}


double LinearRModel :: predict(const Eigen :: VectorXd& input) const{
    // dot product y = xw + b
    return input.dot(weights) + bias;
}


json LinearRModel :: serialize() const {
    json j;
    j["name"] = name;
    j["bias"] = bias;
    j["l2Penalty"] = l2Penalty;

    // convert to normal vector the weights
    std :: vector<double> w_vec(weights.data(), weights.data() + weights.size());
    j["weights"] = w_vec;

    return j;
}


void LinearRModel :: deserialize(const json& j){
    name = j["name"];
    bias = j["bias"];
    l2Penalty = j["l2Penalty"];

    std :: vector<double> w_vec = j["weights"];

    weights = Eigen :: Map <Eigen :: VectorXd> (w_vec.data(), w_vec.size());

    isTrained = true;
}


Eigen :: VectorXd LinearRModel :: getWeights() const{
    return weights;
}

double LinearRModel :: getBias() const{
    return bias;
}


double LinearRModel :: getMSE(const Dataset& data) const{
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
