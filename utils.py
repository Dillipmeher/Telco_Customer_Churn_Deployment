import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

def label_encoding(df, features):
    for i in features:
        if i in df.columns:
            df[i] = df[i].map({"Yes": 1, "No": 0})
    return df

def data_preprocessing(df):
    
    df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
    #telco_base_data.SeniorCitizen = pd.to_object(telco_base_data.SeniorCitizen, errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
    # Handling null values
    df.dropna(subset=['TotalCharges'], inplace=True)

    # Dropping customerID column
    df.drop(columns=['customerID'], inplace=True)

    # Label encoding
    Feature_le = ["Partner", "Dependents", "PhoneService", "Churn", "PaperlessBilling"]
    df = label_encoding(df, Feature_le)

    # Mapping gender
    df["gender"] = df["gender"].map({"Female": 1, "Male": 0})

    # One-hot encoding
    features_ohe = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]
    # df = pd.get_dummies(df, columns=features_ohe, drop_first=True)
    df = pd.get_dummies(df, columns=features_ohe)

    # Handling empty strings
    df.replace('', float('nan'), inplace=True)

    # Scaling features
    features_mms = ["tenure", "MonthlyCharges", "TotalCharges"]
    df_mms = df[features_mms]
    df_remaining = df.drop(columns=features_mms)

    # Dropping rows with missing values after replacing empty strings
    df_mms.dropna(inplace=True)

    mms = MinMaxScaler(feature_range=(0, 1))
    rescaled_feature = mms.fit_transform(df_mms)

    rescaled_feature_df = pd.DataFrame(rescaled_feature, columns=features_mms, index=df_mms.index)
    df_final = pd.concat([rescaled_feature_df, df_remaining], axis=1)
    
    return df_final
