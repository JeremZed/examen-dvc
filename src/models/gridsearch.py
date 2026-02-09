import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def gridsearch_best_params(X_train_path, y_train_path, output_dir, scoring='r2', cv=5, verbose=2):
    """
    Permet d'effectuer un GridSearch pour trouver les meilleurs paramètres.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Chargement des données d'entraînement...")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    
    print(f"Forme X_train: {X_train.shape}")
    print(f"Forme y_train: {y_train.shape}")
    
    model = RandomForestRegressor(random_state=42)    
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    print(f"\nGrille de paramètres:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")    
   
    print(f"\nDébut du GridSearch avec {cv}-fold cross-validation...")
    print("Cela peut prendre plusieurs minutes...")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=verbose
    )
    
    grid_search.fit(X_train, y_train)   
    
    
    params_path = os.path.join(output_dir, 'best_params.pkl')
    joblib.dump(grid_search.best_params_, params_path)
    print(f"\nMeilleurs paramètres sauvegardés: {params_path}")
        
    best_estimator_path = os.path.join(output_dir, 'best_estimator_gridsearch.pkl')
    joblib.dump(grid_search.best_estimator_, best_estimator_path)
    print(f"Meilleur estimateur sauvegardé: {best_estimator_path}")

if __name__ == "__main__":    
    X_train_path = "data/processed_data/X_train_scaled.csv"
    y_train_path = "data/processed_data/y_train.csv"
    output_dir = "models"    
    
    gridsearch_best_params(X_train_path, y_train_path, output_dir)