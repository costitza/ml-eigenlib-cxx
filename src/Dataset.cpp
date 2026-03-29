#include "Dataset.h"
#include "Exceptions.h"

Dataset :: Dataset(int r, int c) : rows(r), cols(c) {
    labels = new double[rows];
    features = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        features[i] = new double[cols];
    }
}

Dataset :: Dataset(const Dataset& other)
    : rows(other.rows), cols(other.cols){

    labels = new double[rows];
    features = new double*[rows];

    for (int i = 0;i < rows; i++){
        features[i] = new double[cols];
        for (int j = 0; j < cols; j++){
            features[i][j] = other.features[i][j];
        }
    }
    for (int i = 0; i < rows; i++){
        labels[i] = other.labels[i];
    }
}

// deep copy
Dataset& Dataset :: operator=(const Dataset& other){
    if (this == &other){
        return *this;
    }

    // clear old states
    for (int i = 0;i < rows; i++){
        delete[] features[i];
    }

    delete[] features;
    delete[] labels;

    rows = other.rows;
    cols = other.cols;

    labels = new double[rows];
    features = new double*[rows];

    for (int i = 0;i < rows; i++){
        features[i] = new double[cols];

        labels[i] = other.labels[i];

        for (int j = 0; j < cols; j ++){
            features[i][j] = other.features[i][j];
        }
    }

    return *this;
} 

Dataset :: ~Dataset(){
    for (int i = 0;i < rows; i++){
        delete[] features[i];
    }

    delete[] features;
    delete[] labels;
}


double Dataset :: getLabel(int index) const{
    if (index < 0 || index > rows){
        throw DataIndexException(index, rows);
    }

    return labels[index];
}

Eigen :: VectorXd Dataset :: getRowsAsEigen(int index) const{
    if (index < 0 || index > rows){
        throw DataIndexException(index, rows);
    }

    Eigen :: VectorXd vec(cols);

    for (int j = 0; j < cols; j++){
        vec(j) = features[index][j];
    }

    return vec;
}


// dummy until we will have the reader from a csv
void Dataset :: populateDummyData(){
    // simple linear relation: y = 2x0 + 3 * x1
    for(int i = 0;i < rows; i++){
        for(int j = 0;j < cols; j++){
            // random looking but deterministic values
            features[i][j] = (i + 1) * 0.5 + j;
        }

        labels[i] = features[i][0] * 2.0;
        if (cols > 1) {
            labels[i] += features[i][1] * 3.0;
        }
    }
}