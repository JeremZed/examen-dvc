import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(input_path, output_dir, target=None, test_size=0.2, random_state=123):
    """
    Permet de splitter les données en dataset d'entraînement et de test.
    """

    if target is None:
        raise ValueError("Le nom de la variable cible doit être spécifié.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Chargement des données depuis {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Forme du dataset: {df.shape}")
    print(f"Colonnes: {df.columns.tolist()}")
    
    X = df.drop(columns=[target])  
    y = df[target] 
    
    print(f"\nNombre de features: {X.shape[1]}")
    print(f"Variable cible: {y.name}")
       
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"\nTaille ensemble d'entraînement: {X_train.shape[0]}")
    print(f"Taille ensemble de test: {X_test.shape[0]}")    
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"\nDonnées sauvegardées dans {output_dir}/")
    print("Fichiers créés: X_train.csv, X_test.csv, y_train.csv, y_test.csv")

if __name__ == "__main__":
    
    input_path = "data/raw_data/raw.csv"
    output_dir = "data/processed_data"
    
    split_data(input_path, output_dir, target="silica_concentrate")