#pragma once
#include <stdexcept>
#include <string>

// forgot to train error
class NotTrainedException : public std::runtime_error {
public:
    NotTrainedException(const std::string& modelName)
        : std::runtime_error("\n[ML ERROR] The model '" + modelName + "' cannot perform this action because it has not been trained yet!\n") {}
};

// math crash error
class DimensionMismatchException : public std::invalid_argument {
public:
    DimensionMismatchException(int expectedSize, int actualSize)
        : std::invalid_argument("\n[MATH ERROR] Dimension mismatch! The model expects " 
                                + std::to_string(expectedSize) + " features, but you provided " 
                                + std::to_string(actualSize) + ".\n") {}
};

// memory crash error
class DataIndexException : public std::out_of_range {
public:
    DataIndexException(int badIndex, int maxRows)
        : std::out_of_range("\n[DATA ERROR] Tried to access row " + std::to_string(badIndex) + 
                            ", but the dataset only has " + std::to_string(maxRows) + " rows.\n") {}
};