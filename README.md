# Foundations of Classification Algorithms

This project implements and evaluates simple machine learning classification algorithms using the **UCI Adult Income dataset** to predict whether an individual earns more than $50K/year.  

---

## 📂 Project Structure
``` text
dasc41103-ml-classification/
├─ README.md
├─ requirements.txt                # Python dependencies             
├─ .gitignore                      # ignore data/cache/outputs, notebooks checkpoints, etc.
├─ data/
│  ├─ raw/                         # project_adult.csv, project_validation_inputs.csv (read-only)
│  └─ processed/                   # cleaned & encoded datasets, train/val splits
├─ notebooks/
│  ├─ 01_preprocess.ipynb          # Part 1: missing, encode, standardize
│  ├─ 02_perceptron_scratch.ipynb  # Part 2: perceptron from scratch (+ misclass plot)
│  ├─ 03_adaline_scratch.ipynb     # Part 2: Adaline / AdalineSGD (+ MSE plot)
│  ├─ 04_sklearn_baselines.ipynb   # Part 2e: sklearn Perceptron + (Adaline via SGDRegressor note)
│  ├─ 05_logreg_svm.ipynb          # Part 3: Logistic Regression & Linear SVM (+ decision boundaries)
│  └─ 06_reflection.ipynb          # Part 4 answers, figures, citations
curves
├─ outputs/
│  ├─ predictions/
│  │  ├─ Group_18_Perceptron_PredictedOutputs.csv
│  │  ├─ Group_18_Adaline_PredictedOutputs.csv
│  │  ├─ Group_18_LogisticRegression_PredictedOutputs.csv
│  │  └─ Group_18_SVM_PredictedOutputs.csv
│  ├─ figures/                     # misclassifications vs epochs, MSE curves, boundaries
│  └─ reports/                     # exported slides/PDFs

```

---

## Features
- **Preprocessing**
  - Handle missing values  
  - Encode categorical features  
  - Standardize numerical features  

- **Implemented from Scratch**
  - Perceptron  
  - Adaline (batch GD & SGD variants)  

- **Using scikit-learn**
  - Perceptron  
  - Logistic Regression  
  - Support Vector Machine (SVM)  

- **Visualization**
  - Learning curves (misclassifications, MSE)  
  - Decision boundaries with 2 features  

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

## ⚙️ Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/dasc41103-ml-classification.git
   cd dasc41103-ml-classification
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