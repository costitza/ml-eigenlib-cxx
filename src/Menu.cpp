#include "Menu.h"
#include "LinearRModel.h"
#include "LogicRModel.h"
#include "KNNModel.h"
#include "Dataset.h"
#include "Hyperparameters.h"
#include <iostream>
#include <fstream>

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
    std::cout << "4. Save a Model to File\n";
    std::cout << "5. Load a Model from File\n";
    std::cout << "6. Exit\n";
    std::cout << "Choose an option: ";
}


void Menu::run() {
    int choice;
    while (isRunning) {
        printHeader();
        
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
            case 4: saveModel(); break;
            case 5: loadModel(); break; 
            case 6: 
                std::cout << "\nExiting program. Cleaning up memory...\n";
                isRunning = false; 
                break;
            default:
                std::cout << "\n[Error] Invalid choice.\n";
                break;
        }
    }
}

MLModel* Menu :: getModelByID(std :: string targetID){
    MLModel* modelToSave = nullptr;
    for (MLModel* model : models) {
        if (model->getModelID() == targetID) {
            modelToSave = model;
            break;
        }
    }
    return modelToSave;
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

    MLModel* modelToTrain = getModelByID(targetID);

    if (modelToTrain != nullptr) {
        // Generate dummy data: 100 rows, matching the 5 features in hyperparams
        Dataset dummyData(100, 5);
        dummyData.populateDummyData(); 

        std::cout << "\nTraining model " << targetID << "...\n";
        
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


void Menu::saveModel() {
    if (models.empty()) {
        std::cout << "\n[Error] No models available to save.\n";
        return;
    }

    listModels();
    std::cout << "\nEnter the exact ID of the model to save: ";
    std::string targetID;
    std::cin >> targetID;

    MLModel* modelToSave = getModelByID(targetID);

    if (modelToSave != nullptr) {
        std::cout << "Enter filename to save to (e.g., model.json): ";
        std::string filename;
        std::cin >> filename;

        // open a file stream for writing
        std::ofstream file(filename);
        if (file.is_open()) {
            json j = modelToSave->serialize();
            file << j.dump(4);
            file.close();
            std::cout << "[Success] Model " << targetID << " saved to " << filename << "\n";
        } else {
            std::cout << "[Error] Could not open file for writing.\n";
        }
    } else {
        std::cout << "\n[Error] Could not find a model with ID: " << targetID << "\n";
    }
}


void Menu::loadModel() {
    std::cout << "\n--- Load Model ---\n";
    std::cout << "Enter filename to load from (e.g., model.json): ";
    std::string filename;
    std::cin >> filename;

    // open a file stream for reading
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "[Error] Could not open file " << filename << ". Does it exist?\n";
        return;
    }

    json j;
    try {
        file >> j;
    } catch (const json::parse_error& e) {
        std::cout << "[Error] Invalid JSON format: " << e.what() << "\n";
        return;
    }
    file.close();

    std::cout << "\nWhat type of model is inside this file?\n";
    std::cout << "1. Linear Regression\n";
    std::cout << "2. Logistic Regression\n";
    std::cout << "3. K-Nearest Neighbors\n";
    std::cout << "Select type: ";
    
    int type;
    std::cin >> type;


    std::string name = j.value("name", "LoadedModel");
    
    // create default hyperparameters (they will be overwritten if theyre serialized)
    Hyperparameters hp(5, 0.01, 100);
    if (j.contains("hyperparameters")) {
        hp.deserialize(j["hyperparameters"]);
    }

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

    // push model into blank
    try {
        newModel->deserialize(j);
        models.push_back(newModel);
        std::cout << "[Success] Model loaded and created with new ID: " << newModel->getModelID() << "\n";
    } catch (const std::exception& e) {
        std::cout << "[Error] Failed to deserialize. Did you pick the wrong model type?\n";
        std::cout << "Details: " << e.what() << "\n";
        // delete if error for memory leaks
        delete newModel;
    }
}

