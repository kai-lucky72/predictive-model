import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import numpy as np

def train():
    """
    Train a Random Forest Regressor model for student performance prediction.
    Run this from Django shell: from performance.train_model import train; train()
    """
    # Load the dataset
    df = pd.read_csv('dataset.csv')
    
    # Encode categorical variable
    df['Extracurricular Activities'] = LabelEncoder().fit_transform(
        df['Extracurricular Activities']
    )
    
    # Prepare features and target
    X = df.drop('Performance Index', axis=1)
    y = df['Performance Index']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("Model Performance Metrics:")
    print("="*50)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("="*50 + "\n")
    
    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")
    
    return model, {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

if __name__ == "__main__":
    train()
