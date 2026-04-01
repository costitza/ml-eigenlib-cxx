# ML mini library for predictive models

- from-scratch
- lightweight
- educational purpose Machine Learning library built entirely in modern C++. 

This project was built to explore how fundamental ML algorithms work under the hood without relying on Python libraries like scikit-learn / Pytorch. It features a fully interactive terminal menu that lets you create models, train them, and save/load their states in json files.

## Contents of project 

Currently, the library supports three foundational models:
* **Linear Regression:** For predicting continuous numerical values using matrix operations.
* **Logistic Regression:** A binary classifier built with a custom sigmoid activation function.
* **K-Nearest Neighbors (KNN):** A flexible "lazy learner" that dynamically supports *both* classification (majority vote) and regression (mean averaging).

**NOTE:** More classes of models and complexity are expected to be added.

### Key features
* **Interactive CLI:** A terminal-based menu to manage your models in real-time.
* **Save & Load:** Models serialize their learned weights, hyperparameters, and types into clean JSON files (stored in a `data/` folder). You can close the app, come back later, and load your exact model back into memory.
* **Auto-Detection:** When loading a JSON file, the library automatically figures out what kind of model it is and rebuilds it for you.
* **Safe Math:** Built-in error handling prevents the program from crashing if you try to multiply mismatched matrices or predict with an untrained model.
* **Modular OOP Architecture:** Designed with strict Object-Oriented principles like polymorphism and virtual inheritance. The codebase is highly modular, meaning you can easily drop in new models or expand the library without tearing apart the core math engine.

## Tech stack & dependencies

This project uses two header-only libraries (included in the `external/` folder):
1. **[Eigen](https://eigen.tuxfamily.org/):** Handles all the heavy lifting for linear algebra, matrix multiplication, and Euclidean distances.
2. **[nlohmann/json](https://github.com/nlohmann/json):** Used to effortlessly parse and save our C++ objects and Eigen matrices into readable text files.

## How to install project

**1. Clone the repository and navigate to the folder:**
```bash
git clone <your-repo-url>
cd ml-eigenlib-cxx
```

**2. Create a build directory:**
```bash
mkdir build
cd build
```

**3. Configure and compile the project:**
```bash
cmake ..
cmake --build .
```

## 🎮 How to Use

Launch the application by running the executable (`oop.exe` on Windows or `./oop` on Mac/Linux). You will be greeted by an interactive menu designed to manage the lifecycle of your Machine Learning models.

### 1. Create a Model
Select **Option 1** to instantiate a new model. 
* Choose between **Linear Regression**, **Logistic Regression**, or **KNN**.
* Give your model a unique name.
* The system automatically assigns a unique Hexadecimal ID (e.g., `MOD-A3F1`) and sets up default hyperparameters.

### 2. Train with Synthetic Data
Select **Option 2** to train a model. 
* The system will list all active models and their IDs.
* Enter the specific **Model ID** you wish to train.
* The library generates a synthetic `Dataset` that perfectly matches the required input features of your model and executes the training algorithm (e.g., Normal Equation for Linear or Gradient Descent for Logistic).

### 3. Monitoring & Listing
Select **Option 3** to view all models currently in memory. 
* Thanks to custom operator overloading, this will print a detailed summary of each model, including its type, training status, bias, and specific settings (like the $K$ value for KNN).

### 4. Persistent Storage (Save/Load)
* **Save (Option 4):** Enter a Model ID and a filename (e.g., `my_model.json`). The model will be serialized—saving its type, hyperparameters, and learned weights—into the `data/` directory.
* **Load (Option 5):** Enter the filename. The library will scan the JSON, **automatically detect the model type**, reconstruct the correct C++ object, and restore its trained state.

### 5. Error Handling
The library is protected by custom exceptions. If you attempt to predict with a model before training it, or if you provide data with mismatched dimensions, the program will catch the error and display a descriptive message instead of crashing.