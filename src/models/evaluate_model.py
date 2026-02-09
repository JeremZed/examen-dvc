import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os

def evaluate_model(X_test_path, y_test_path, model_path, output_dir, metrics_dir):
    """
    Permet d'évaluer le modèle.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
        
    print("Chargement des données de test...")
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    
    print(f"Forme X_test: {X_test.shape}")
    print(f"Forme y_test: {y_test.shape}")
        
    print("\nChargement du modèle entraîné...")
    model = joblib.load(model_path)
    print("Modèle chargé avec succès!")
        
    print("\nGénération des prédictions...")
    y_pred = model.predict(X_test)
    
    print("\nCalcul des métriques d'évaluation...")
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)    
    
    print("\n" + "="*50)
    print("MÉTRIQUES D'ÉVALUATION")
    print("="*50)
    print(f"MSE (Mean Squared Error):       {mse:.6f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.6f}")
    print(f"MAE (Mean Absolute Error):      {mae:.6f}")
    print(f"R² Score:                       {r2:.6f}")
    
    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2)
    }    
    
    metrics_path = os.path.join(metrics_dir, 'scores.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMétriques sauvegardées: {metrics_path}")
        
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'residual': y_test - y_pred
    })      
    
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nPrédictions sauvegardées: {predictions_path}")
    
if __name__ == "__main__":
    
    X_test_path = "data/processed_data/X_test_scaled.csv"
    y_test_path = "data/processed_data/y_test.csv"
    model_path = "models/trained_model.pkl"
    output_dir = "data"
    metrics_dir = "metrics"
    
    evaluate_model(X_test_path, y_test_path, model_path, output_dir, metrics_dir)