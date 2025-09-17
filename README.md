# Foundations of Classification Algorithms

This project implements and evaluates simple machine learning classification algorithms using the **UCI Adult Income dataset** to predict whether an individual earns more than $50K/year.  

---

## 📂 Project Structure
```text
ML-CLASSIFICATION/
├─ README.md
├─ requirements.txt                # Python dependencies             
├─ LICENSE
├─ data/
│  ├─ raw/                         # project_adult.csv, project_validation_inputs.csv (read-only)
│  └─ processed/                   # cleaned & encoded datasets, train/val splits
├─ notebooks/
│  ├─ 01_preprocess.ipynb          # Part 1: missing, encode, standardize
│  ├─ 02_perceptron_scratch.ipynb  # Part 2: perceptron from scratch (+ misclass plot)
│  ├─ 03_adaline_scratch.ipynb     # Part 2: Adaline / AdalineSGD (+ MSE plot)
│  ├─ 04_sklearn_baselines.ipynb   # Part 2e: sklearn Perceptron & Adaline
│  ├─ 05_logreg_svm.ipynb          # Part 3: Logistic Regression & Linear SVM (+ decision boundaries)
│  └─ 06_reflection.ipynb          # Part 4 answers, discussion, figures
├─ outputs/
│  ├─ graphs/                      # learning curves, MSE curves
│  ├─ Group_18_Perceptron_PredictedOutputs.csv
│  ├─ Group_18_Adaline_PredictedOutputs.csv
│  ├─ Group_18_LogisticRegression_PredictedOutputs.csv
│  └─ Group_18_SVM_PredictedOutputs.csv
```

---

## Features
- **Preprocessing**
  - Handle missing values with imputations
  - Encode categorical features  
  - Standardize numerical features  

- **Implemented from Scratch**
  - Perceptron (misclassification curves)
  - Adaline (batch GD & SGD variants)  

- **Using scikit-learn**
  - Perceptron (with GridSearchCV)
  - Adaline (via SGDRegressor, custom scorer)
  - Logistic Regression
  - Support Vector Machine (Linear SVM) 

- **Visualization**
  - Learning curves (misclassifications, MSE)  

- **Reflection**
  - Feature scaling & gradient descent  
  - Batch vs. stochastic updates  
  - Regularization and overfitting control  
  - Comparison of linear classifiers

---

## Deliverables
- Python source code + Jupyter notebooks  
- Four prediction CSVs (Perceptron, Adaline, Logistic Regression, SVM)  
- Figures for learning curves and decision boundaries  
- 15-minute recorded presentation with slides  
- Written reflection on algorithm behavior  

## Data
- `project_adult.csv`  
- `project_validation_inputs.csv`  
- Source: [UCI Adult Income Dataset](https://archive.ics.uci.edu/dataset/2/adult)  

---

## Results

**Scratch Implementations**
- Perceptron (scratch): Successfully converged with clear decrease in misclassifications over epochs.
- Adaline (scratch, GD & SGD): Showed expected reduction in MSE over training epochs.

**sckit-learn Implementations**
- Perceptron (best params: `alpha=0.0001`, `eta0=0.1`, `penalty='elasticnet'`):
  - Cross-validation accuracy: ~0.82
  - Test accuracy: ~0.77

- Adaline (`SGDRegressor`, best params: `alpha=0.001`, `eta0=0.0001`):
  - Cross-validation unstable (`NaN` scores for some folds due to SGD sensitivity)
  - Test accuracy: ~0.83

- Logistic Regression:

- SVM:

---

## ⚙️ Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/ml-classification.git
   cd ml-classification
2. **Create a virtual environment & install dependencies**
    ```bash
    python -m venv venv
    source venv/bin/activate     # Mac/Linux
    venv\Scripts\activate        # Windows

    pip install -r requirements.txt
    ```
3. **Run Notebooks**
    ```bash
    jupyter notebook
    ```