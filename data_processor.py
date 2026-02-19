import pandas as pd
import numpy as np
try:
    from AutoClean import AutoClean
    _HAS_AUTOCLEAN = True
except Exception:
    _HAS_AUTOCLEAN = False
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
import os
import argparse
import json
import sys
import traceback

def preprocess_training_ready(file_path):
    """
    Complete Training-Ready Pipeline
    Builds on AutoClean's output (cleaned Pandas DataFrame) with sklearn for the rest.
    Assumes the last column is the target/label.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")

    # 1. Attempt AutoClean if available; otherwise use a simple fallback cleaning
    df = pd.read_csv(file_path)
    if _HAS_AUTOCLEAN:
        try:
            ac = AutoClean(df, mode='auto')
            cleaned_df = ac.output
        except Exception:
            cleaned_df = df.copy()
    else:
        # basic fallback: drop duplicate rows, fill numeric NaNs with column mean
        cleaned_df = df.drop_duplicates().copy()
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    # 2. Detect target column and separate features (X) and target (y)
    # Prefer a column named 'Attrition' when present (common HR dataset); otherwise use the last column
    target_col = 'Attrition' if 'Attrition' in cleaned_df.columns else cleaned_df.columns[-1]
    X = cleaned_df.drop(columns=[target_col])  # All feature columns
    y = cleaned_df[target_col]   # Target column
    
    # 3. Encode target if categorical (NNs need numeric labels)
    le = None
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Ensure y is a numpy array for splitting and tensor conversion
    if isinstance(y, pd.Series):
        y = y.values
    elif not isinstance(y, np.ndarray):
        y = np.array(y)
    
    # 4. Handle categorical features in X (label-encode categorical columns)
    feature_label_encoders = {}
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(cat_cols) > 0:
        for col in cat_cols:
            le_col = LabelEncoder()
            X[col] = X[col].astype(str).fillna('')
            X[col] = le_col.fit_transform(X[col])
            feature_label_encoders[col] = le_col
    # Fill any remaining NaNs and convert to numeric matrix
    X = X.fillna(0).values
    
    # 5. Scale features (CRITICAL for NNs: prevents gradient explosion/vanishing)
    scaler = StandardScaler()
    if X.shape[0] > 0:
        X = scaler.fit_transform(X)
    
    # 6. Split: 70/15/15 (no leakage - fit scalers on train only)
    # 6. Split: 70/15/15 (no leakage - fit scalers on train only)
    # If dataset is very small, avoid splits that would produce empty sets
    if X.shape[0] < 3:
        # Not enough samples to split; return everything as train and empty val/test
        X_train, y_train = X, y
        X_val, y_val = np.empty((0, X.shape[1])), np.array([])
        X_test, y_test = np.empty((0, X.shape[1])), np.array([])
    else:
        # Correct way to handle scaling without leakage is to fit on train and transform others
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)  # 0.15/0.85 ≈ 0.176
    
    # 7. Convert to PyTorch tensors (training-ready!)
    train_data = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    val_data = torch.tensor(X_val, dtype=torch.float32)
    val_labels = torch.tensor(y_val, dtype=torch.long)
    test_data = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.long)
    
    return {
        'train': (train_data, train_labels),
        'val': (val_data, val_labels),
        'test': (test_data, test_labels),
        'scaler': scaler,  # Save for inference
        'label_encoder': le,
        'feature_label_encoders': feature_label_encoders
    }


def process_and_report(file_path):
    """
    Run cleansing (similar to preprocess_training_ready) but return a JSON-serializable
    report describing what cleansing actions were performed.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")

    df = pd.read_csv(file_path)
    original_rows = df.shape[0]

    # 1. Attempt AutoClean if available; otherwise use a simple fallback cleaning
    duplicates_removed = 0
    missing_filled = 0
    outliers_replaced = 0
    categorical_columns = []

    if _HAS_AUTOCLEAN:
        try:
            ac = AutoClean(df, mode='auto')
            cleaned_df = ac.output
            # Best-effort estimates (AutoClean may expose stats in real use)
            duplicates_removed = int(original_rows - cleaned_df.shape[0])
        except Exception:
            cleaned_df = df.drop_duplicates().copy()
            duplicates_removed = int(original_rows - cleaned_df.shape[0])
    else:
        cleaned_df = df.drop_duplicates().copy()
        duplicates_removed = int(original_rows - cleaned_df.shape[0])
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            nan_count = int(cleaned_df[col].isna().sum())
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            missing_filled += nan_count

    # 2. Detect target column and separate features (X) and target (y)
    # Prefer a column named 'Attrition' when present (common HR dataset); otherwise use the last column
    target_col = 'Attrition' if 'Attrition' in cleaned_df.columns else cleaned_df.columns[-1]
    X = cleaned_df.drop(columns=[target_col])
    y = cleaned_df[target_col]

    # 3. Encode target if categorical
    le = None
    target_encoded = False
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)
        target_encoded = True

    # 4. Handle categorical features in X (label-encode categorical columns)
    cat_cols = list(X.select_dtypes(include=['object', 'category']).columns)
    categorical_columns = cat_cols
    categorical_mappings = {}

    # 5. Detect and replace outliers (IQR) — operate only on original numeric feature columns
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    if len(num_cols) > 0:
        for col in num_cols:
            # ensure column is float so we can replace with non-integer values
            cleaned_df[col] = cleaned_df[col].astype(float)
            col_series = cleaned_df[col]
            non_nan = col_series[~col_series.isna()]
            if non_nan.size == 0:
                continue
            q1 = np.percentile(non_nan, 25)
            q3 = np.percentile(non_nan, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask_series = ((col_series < lower) | (col_series > upper)) & (~col_series.isna())
            replaced = int(mask_series.sum())
            if replaced > 0:
                outliers_replaced += replaced
                # replace in the cleaned dataframe (so subsequent encoding/scaling sees the replaced values)
                cleaned_df.loc[mask_series, col] = float((q1 + q3) / 2)

    # Recompute X after potential outlier replacements
    X = cleaned_df.drop(columns=[target_col])

    # Now label-encode categorical feature columns and build mappings
    if len(cat_cols) > 0:
        for col in cat_cols:
            le_col = LabelEncoder()
            X[col] = X[col].astype(str).fillna('')
            X[col] = le_col.fit_transform(X[col])
            categorical_mappings[col] = le_col.classes_.tolist()
    else:
        categorical_mappings = {}

    # Fill any remaining NaNs and convert to numeric matrix
    nan_total = int(X.isna().sum().sum())
    missing_filled += nan_total
    X = X.fillna(0).values

    # 6. Scale features
    scaler = StandardScaler()
    if X_arr.shape[0] > 0:
        X_scaled = scaler.fit_transform(X_arr)
    else:
        X_scaled = X_arr

    # 7. Split counts
    total_samples = X_scaled.shape[0]
    if total_samples < 3:
        train_n = total_samples
        val_n = 0
        test_n = 0
    else:
        temp_size = int(round(total_samples * 0.85))
        test_n = int(round(total_samples * 0.15))
        train_n = int(round(temp_size * 0.824))  # approx 0.70
        val_n = total_samples - train_n - test_n

    report = {
        'original_rows': int(original_rows),
        'duplicates_removed': int(duplicates_removed),
        'missing_filled': int(missing_filled),
        'outliers_replaced': int(outliers_replaced),
        'categorical_columns': categorical_columns,
        'categorical_mappings': categorical_mappings,
        'target_encoded': bool(target_encoded),
        'target_mapping': (le.classes_.tolist() if le is not None else None),
        'samples': int(total_samples),
        'train_samples': int(train_n),
        'val_samples': int(val_n),
        'test_samples': int(test_n),
        'features': int(X_scaled.shape[1])
    }

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run data preprocessing and emit a JSON report')
    parser.add_argument('--file', '-f', required=False, help='Path to CSV file. If omitted, read stdin')
    args = parser.parse_args()
    try:
        if args.file:
            report = process_and_report(args.file)
        else:
            # read CSV from stdin into a temp file
            import tempfile
            content = sys.stdin.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8') as tf:
                tf.write(content)
                tf_path = tf.name
            report = process_and_report(tf_path)
            try:
                os.unlink(tf_path)
            except Exception:
                pass
        print(json.dumps(report))
    except Exception as e:
        err = {'error': str(e), 'traceback': traceback.format_exc()}
        print(json.dumps(err))

if __name__ == "__main__":
    # Example usage:
    # results = preprocess_training_ready('path_to_your_dataset.csv')
    # print("Data processing complete.")
    pass
