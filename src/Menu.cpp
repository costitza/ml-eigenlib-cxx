#include "Menu.h"
#include "LinearRModel.h"
#include "LogicRModel.h"
#include "KNNModel.h"
#include "Dataset.h"
#include "Hyperparameters.h"
#include <iostream>

// private constr
Menu :: Menu() : isRunning(true) {}

Menu :: ~Menu() {
    for (MLModel* model : models){
        delete model;
    }
    models.clear();
}

Menu& Menu :: getInstance(){
    static Menu instance;
    return instance;
}


void Menu::printHeader() const {
    std::cout << "\n=========================================\n";
    std::cout << "      Machine Learning Library C++       \n";
    std::cout << "          Total Models: " << MLModel::getTotalModels() << "\n";
    std::cout << "=========================================\n";
    std::cout << "1. Create a New Model\n";
    std::cout << "2. Train a Model (Dummy Data)\n";
    std::cout << "3. List Active Models\n";
    std::cout << "4. Exit\n";
    std::cout << "Choose an option: ";
}


void Menu::run() {
    int choice;
    while (isRunning) {
        printHeader();
        
        // Robust input handling
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "\n[Error] Invalid input. Please enter a number.\n";
            continue;
        }

        switch (choice) {
            case 1: createModel(); break;
            case 2: trainModel(); break;
            case 3: listModels(); break;
            case 4: 
                std::cout << "\nExiting program. Cleaning up memory...\n";
                isRunning = false; 
                break;
            default:
                std::cout << "\n[Error] Invalid choice.\n";
                break;
        }
    }
}


void Menu::createModel() {
    std::cout << "\n--- Create Model ---\n";
    std::cout << "1. Linear Regression\n";
    std::cout << "2. Logistic Regression (Classifier)\n";
    std::cout << "3. K-Nearest Neighbors\n";
    std::cout << "Select type: ";
    
    int type;
    std::cin >> type;

    std::string name;
    std::cout << "Enter a name for the model: ";
    std::cin >> name;

    // Create default hyperparameters for now (5 features, lr=0.01, 100 epochs)
    Hyperparameters hp(5, 0.01, 100);

    MLModel* newModel = nullptr;

    if (type == 1) {
        newModel = new LinearRModel(name, hp, 0.0);
    } else if (type == 2) {
        newModel = new LogicRModel(name, hp, 2, 0.5);
    } else if (type == 3) {
        newModel = new KNNModel(name, hp, 3, true);
    } else {
        std::cout << "[Error] Unknown model type.\n";
        return;
    }

    models.push_back(newModel);
    std::cout << "[Success] Created " << newModel->getName() 
              << " with ID: " << newModel->getModelID() << "\n";
}


void Menu::trainModel() {
    if (models.empty()) {
        std::cout << "\n[Error] No models available to train.\n";
        return;
    }

    listModels(); // Show the user the available IDs
    std::cout << "\nEnter the exact ID of the model to train (e.g., MOD-A3F1): ";
    
    std::string targetID;
    std::cin >> targetID;

    MLModel* modelToTrain = nullptr;

    // Search through the vector for a matching ID
    for (MLModel* model : models) {
        if (model->getModelID() == targetID) {
            modelToTrain = model;
            break; // Found it! No need to keep searching.
        }
    }

    // Check if we actually found a model
    if (modelToTrain != nullptr) {
        // Generate dummy data: 100 rows, matching the 5 features in our Hyperparameters
        Dataset dummyData(100, 5);
        dummyData.populateDummyData(); 

        std::cout << "\nTraining model " << targetID << "...\n";
        
        // POLYMORPHISM: This calls the correct train() method!
        modelToTrain->train(dummyData); 
    } else {
        std::cout << "\n[Error] Could not find a model with ID: " << targetID << "\n";
    }
}


void Menu::listModels() const {
    std::cout << "\n--- Active Models ---\n";
    if (models.empty()) {
        std::cout << "No models instantiated yet.\n";
        return;
    }

    for (size_t i = 0; i < models.size(); ++i) {
        std::cout << "[" << i << "] ID: " << models[i]->getModelID() 
                  << " | Name: " << models[i]->getName() 
                  << " | Trained: " << (models[i]->getIsTrained() ? "Yes" : "No") << "\n";
    }
}