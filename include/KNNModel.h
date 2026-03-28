#pragma once

#include "Classifier.h"
#include "Regressor.h"
#include "Dataset.h"

class KNNModel : public Classifier, public Regressor{
    int kNeighbors;
    bool isClassification;

    Dataset* savedData;

public:
    KNNModel() : Classifier(), Regressor(), kNeighbors(3), isClassification(true), savedData(nullptr) {}
    KNNModel(std::string modelName, const Hyperparameters& hp, int k = 3, bool isClass = true);

    // copy constructor + deep copy
    KNNModel(const KNNModel& other);
    KNNModel& operator=(const KNNModel& other);

    ~KNNModel() override;

    // override methods
    void train(const Dataset& data) override;
    double predict(const Eigen :: VectorXd& input) override;

    // methods specific for both classifier and regressor
    void printConfusionMatrix() const override;
    double getMSE() const override;

    // json serialization
    json serialize() const override;
    void deserialize(const json& j) override;

    // getters / setters
    int getKNeighbors() const;
    bool getIsClassification() const;
    
};