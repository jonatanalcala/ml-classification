import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Preprocess the Adult Income dataset.

    This function performs several preprocessing steps to clean and prepare
    the dataset for machine learning models. Steps include stripping whitespace,
    handling missing values, dropping redundant columns, one-hot encoding
    categorical features, encoding the target variable, and returning feature
    and target matrices.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataset containing both numeric and categorical features,
        along with the 'income' target column.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix after preprocessing (numeric + one-hot encoded categorical variables).

    y : pandas.Series
        Target vector with binary-encoded 'income' values (1 for >50K, 0 for <=50K).

    Notes
    -----
    - Rows containing missing values in categorical or target columns are dropped.
    - The 'education' column is removed since 'education-num' is retained.
    - One-hot encoding is applied to categorical features (excluding 'education').
    - Scaling is **not** performed here to avoid data leakage; it should be done
      later within a training pipeline using only the training set.
    - Ensure train/test split is done before fitting scalers or models.
    """

    # 0) Peek
    print("First 5 rows before transformation:\n", df.head(), "\n*********************\n")

    # 1) Define categorical columns up front
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'native-country']

    # 2) Normalize placeholders & whitespace
    df = df.copy()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace('?', np.nan, inplace=True)

    # 3) Drop rows missing required categoricals or target
    df.dropna(subset=cat_cols + ['income'], inplace=True)

    # 4) Drop redundant 'education' (keep 'education-num')
    df.drop(columns=['education'], inplace=True)

    # 5) One-hot encode categoricals (education already removed)
    ohe_cols = [c for c in cat_cols if c != 'education']
    df = pd.get_dummies(df, columns=ohe_cols, dtype=int)

    # 6) Target encode AFTER stripping spaces
    df['income'] = df['income'].map({'>50K': 1, '<=50K': 0})

    # 7) Split features/target
    y = df['income']
    X = df.drop(columns=['income'])

    # 9. Separate features and target
    X = df.drop(columns=['income'], axis=1)
    y = df[['income']]

    # 8) Do NOT scale here to avoid leakage; scale in a Pipeline or on X_train only
    # Example (outside): scaler.fit(X_train[numeric_cols]); X_train[numeric_cols] = scaler.transform(...)

    print("\n*********************\nFirst 5 rows of X:\n", X.head(), "\n*********************")
    print("First 5 rows of y:\n", y.head(), "\n*********************\n")
    return X, y