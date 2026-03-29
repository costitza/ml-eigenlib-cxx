#include "Classifier.h"
#include <iostream>

Classifier :: Classifier(std::string modelName, const Hyperparameters& hp, int classes, double threshold) 
    : MLModel(modelName, hp), numClasses(classes), decisionThreshold(threshold) {}

int Classifier :: getNumClasses() const{
    return numClasses;
}

double Classifier :: getDecisionThreshold() const{
    return decisionThreshold;
}


// method for models derived

void Classifier::printConfusionMatrix(const Dataset& data) const {
    if (!this->getIsTrained()) {
        std::cout << "Model " << this->getName() << " is not trained yet!\n";
        return;
    }

    int tp = 0, tn = 0, fp = 0, fn = 0;
    
    for (int i = 0; i < data.getRows(); ++i) {
        double y_true = data.getLabel(i);
        
        // this will route to LogicRModel::predict() OR KNNModel::predict()
        double y_pred = this->predict(data.getRowsAsEigen(i));

        if (y_true == 1.0 && y_pred == 1.0) tp++;
        else if (y_true == 0.0 && y_pred == 0.0) tn++;
        else if (y_true == 0.0 && y_pred == 1.0) fp++;
        else if (y_true == 1.0 && y_pred == 0.0) fn++;
    }

    std::cout << "Confusion Matrix for " << this->getName() << " ---\n";
    std::cout << "True Positives (TP):  " << tp << "\tFalse Positives (FP): " << fp << "\n";
    std::cout << "False Negatives (FN): " << fn << "\tTrue Negatives (TN):  " << tn << "\n";
    
    double accuracy = (double)(tp + tn) / data.getRows() * 100.0;
    std::cout << "Overall Accuracy: " << accuracy;
}