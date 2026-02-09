import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_model(X_train_path, y_train_path, params_path, output_dir):
    """
    Permet de lancer l'entraînement du modèle final avec les meilleurs paramètres.
    """
    
    os.makedirs(output_dir, exist_ok=True)    
    
    print("Chargement des données d'entraînement...")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    
    print(f"Forme X_train: {X_train.shape}")
    print(f"Forme y_train: {y_train.shape}")    
   
    print("\nChargement des meilleurs paramètres...")
    best_params = joblib.load(params_path)
    
    print("Paramètres utilisés pour l'entraînement:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")    
    
    print("\nEntraînement du modèle...")
    model = RandomForestRegressor(
        random_state=42,
        **best_params
    )
    
    model.fit(X_train, y_train)        
    
    model_path = os.path.join(output_dir, 'trained_model.pkl')
    joblib.dump(model, model_path)
    print(f"\nModèle entraîné sauvegardé: {model_path}")  
    
if __name__ == "__main__":
    
    X_train_path = "data/processed_data/X_train_scaled.csv"
    y_train_path = "data/processed_data/y_train.csv"
    params_path = "models/best_params.pkl"
    output_dir = "models"    
    
    train_model(X_train_path, y_train_path, params_path, output_dir)