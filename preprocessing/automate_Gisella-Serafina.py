import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from joblib import dump

# --- 1. Custom Capper Class ---
class IQRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lower_limit_ = None
        self.upper_limit_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        Q1 = X_df.quantile(0.25)
        Q3 = X_df.quantile(0.75)
        IQR = Q3 - Q1
        self.lower_limit_ = Q1 - 1.5 * IQR
        self.upper_limit_ = Q3 + 1.5 * IQR
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].clip(lower=self.lower_limit_[col], 
                                       upper=self.upper_limit_[col])
        return X_df.values

# --- 2. Main Preprocessing Function ---
def preprocess_data(data, target_column, save_path, header_path, output_csv_path):
    # A. Drop Duplikat
    data = data.drop_duplicates()

    # B. Cleaning Dasar
    if 'customerID' in data.columns:
        data = data.drop(columns=['customerID'])
    
    if 'TotalCharges' in data.columns:
        # Ganti spasi kosong dengan NaN lalu paksa jadi float
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # C. DOWNSAMPLING: Ambil hanya 1000 baris (500 Yes, 500 No)
    df_churn_yes = data[data[target_column] == 'Yes'].head(500)
    df_churn_no = data[data[target_column] == 'No'].head(500)
    data = pd.concat([df_churn_yes, df_churn_no])
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # D. Klasifikasi Fitur (Hardcoded)
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
        'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    # E. Simpan Header untuk referensi 
    column_names = data.columns.drop(target_column)
    pd.DataFrame(columns=column_names).to_csv(header_path, index=False)

    # F. Pipeline Definition
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('capper', IQRTransformer()),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # G. Target Mapping
    y_raw = data[target_column].map({'Yes': 1, 'No': 0})
    X_raw = data.drop(columns=[target_column])

    # H. Transformasi Data
    X_transformed = preprocessor.fit_transform(X_raw)
    
    # Ambil nama kolom baru setelah OneHot
    cat_columns = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
    final_columns = numeric_features + list(cat_columns)

    # I. Create Final DataFrame & Save CSV
    df_final = pd.DataFrame(X_transformed, columns=final_columns)
    df_final[target_column] = y_raw.values # Taruh Churn di paling akhir
    df_final.to_csv(output_csv_path, index=False)
    print(f"Dataset terbaru disimpan ke: {output_csv_path} (Total: {len(df_final)} baris)")

    # J. Simpan Pipeline (.joblib)
    dump(preprocessor, save_path)
    
    # K. Split untuk keperluan modelling 
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_raw, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# --- 3. Eksekusi Automasi ---
if __name__ == "__main__":
    import os

    base_path = os.path.dirname(__file__) 
    raw_data_path = os.path.join(base_path, '..', 'TelcoCustomerChurn_raw.csv') 
    
    try:
        raw_data = pd.read_csv(raw_data_path)
        preprocess_data(
            data=raw_data,
            target_column='Churn',
            save_path=os.path.join(base_path, 'preprocessor_pipeline.joblib'),
            header_path=os.path.join(base_path, 'data_header.csv'),
            output_csv_path=os.path.join(base_path, 'TelcoCustomerChurn_preprocessing.csv')
        )
    except FileNotFoundError:
        print(f"File mentah tidak ditemukan di: {raw_data_path}")