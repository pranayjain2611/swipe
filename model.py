from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data_loader import load_and_preprocess_data, split_data
import numpy as np

def train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")
    return predictions

def main():
    X_scaled, y_level1, encoded_state, sleep_stage, state_encoder, scaler = load_and_preprocess_data()
    
    # Level 1: Wake vs. Sleep
    X_train, X_test, y_train_level1, y_test_level1 = split_data(X_scaled, y_level1)
    model_level1 = train_svm(X_train, y_train_level1)
    predictions_level1 = evaluate_model(model_level1, X_test, y_test_level1)
    
    # Preparing data for Level 2 and Level 3 based on Level 1 predictions
    wake_indices = np.where(y_test_level1 == 0)[0]  # Indices where the true label is Wake
    sleep_indices = np.where(y_test_level1 > 0)[0]  # Indices where the true label indicates Sleep

    # Level 2: Sleep or Rest (only for wake cases)
    if len(wake_indices) > 0:
        X_wake = X_test[wake_indices]
        y_wake = encoded_state[wake_indices]
        model_level2 = train_svm(X_wake, y_wake)
        predictions_level2 = evaluate_model(model_level2, X_wake, y_wake)
    
    # Level 3: Identifying Sleep Stage (only for sleep cases)
    if len(sleep_indices) > 0:
        X_sleep = X_test[sleep_indices]
        y_sleep_stage = sleep_stage[sleep_indices].astype(int)  # Ensure sleep_stage is integer for training
        model_level3 = train_svm(X_sleep, y_sleep_stage)
        predictions_level3 = evaluate_model(model_level3, X_sleep, y_sleep_stage)

if __name__ == "__main__":
    main()
