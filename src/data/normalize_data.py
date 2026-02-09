import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def normalize_data(train_path, test_path, output_dir, scaler_output_dir):
    """
    Permet de normaliser les données en utilisant une méthode basique comme StandardScaler.
    """
   
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scaler_output_dir, exist_ok=True)
        
    print("Chargement des données...")
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    
    print(f"Forme X_train: {X_train.shape}")
    print(f"Forme X_test: {X_test.shape}")
        
    print("\nStatistiques avant normalisation (X_train):")
    print(X_train.info())
    
    print("\nNormalisation des données...")

    X_train = X_train.drop(columns=['date'])
    X_test = X_test.drop(columns=['date'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)    
    
    X_train_scaled = pd.DataFrame(
        X_train_scaled, 
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, 
        columns=X_test.columns
    )    
    
    print("\nStatistiques après normalisation (X_train):")
    print(X_train_scaled.describe())
        
    train_output = os.path.join(output_dir, 'X_train_scaled.csv')
    test_output = os.path.join(output_dir, 'X_test_scaled.csv')
    
    X_train_scaled.to_csv(train_output, index=False)
    X_test_scaled.to_csv(test_output, index=False)
    
    print(f"\nDonnées normalisées sauvegardées:")
    print(f"  - {train_output}")
    print(f"  - {test_output}")
    
    # On sauvegarde le scaler pour une utilisation future afin d'eviter les fuites de données
    scaler_path = os.path.join(scaler_output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler sauvegardé: {scaler_path}")

if __name__ == "__main__":
    
    train_path = "data/processed_data/X_train.csv"
    test_path = "data/processed_data/X_test.csv"
    output_dir = "data/processed_data"
    scaler_output_dir = "models"
    
    normalize_data(train_path, test_path, output_dir, scaler_output_dir)