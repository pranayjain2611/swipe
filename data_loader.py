import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(filepath='/Users/pranayjain/University of Freiburg/4th Semster/Psycology Project/python_scripts/output_directory/hurst_svm_dataset_all.csv'):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Convert 'sleep_stage' to binary for the first level classification: Wake (0) vs Sleep (1,2,3)
    df['binary_sleep_stage'] = df['sleep_stage'].apply(lambda x: 0 if (x.split()[0] == 'W') else 1)
    df['sleep_stage'] = df['sleep_stage'].apply(lambda x: 0 if (x.split()[0] == 'W') else x.split()[0])
    df.replace(["unscorable","Unscorable"],"-1",inplace=True)

    # Encode 'state' to numeric values for the second level classification
    state_encoder = LabelEncoder()
    df['encoded_state'] = state_encoder.fit_transform(df['state'])
    
    # Features and labels for first level classification
    X = df[[f'hurst_{i}' for i in range(1, 49)]]
    y_level1 = df['binary_sleep_stage']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_level1, df['encoded_state'], df['sleep_stage'], state_encoder, scaler

def split_data(X, y):
    # Split the data into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42)
