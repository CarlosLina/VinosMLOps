import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

def entrenar_modelos():
    # Cargar datos limpios
    df = pd.read_csv('../data/WineQT_clean.csv')
    
    # Separar características (X) y variable objetivo (y)
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Dividir datos en entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar Modelo 1: Regresión Logística
    modelo_lr = LogisticRegression(max_iter=10000)
    modelo_lr.fit(X_train, y_train)
    preds_lr = modelo_lr.predict(X_test)
    
    # Entrenar Modelo 2: Random Forest
    modelo_rf = RandomForestClassifier(random_state=42)
    modelo_rf.fit(X_train, y_train)
    preds_rf = modelo_rf.predict(X_test)
    
    # Calcular métricas (usamos average='weighted' por ser multiclase)
    acc_lr = accuracy_score(y_test, preds_lr)
    f1_lr = f1_score(y_test, preds_lr, average='weighted')
    
    acc_rf = accuracy_score(y_test, preds_rf)
    f1_rf = f1_score(y_test, preds_rf, average='weighted')
    
    print("--- Resultados de Regresión Logística ---")
    print(f"Accuracy: {acc_lr:.4f} | F1-Score: {f1_lr:.4f}")
    
    print("\n--- Resultados de Random Forest ---")
    print(f"Accuracy: {acc_rf:.4f} | F1-Score: {f1_rf:.4f}")
    
    # Guardar el mejor modelo (Random Forest suele ganar en este dataset)
    os.makedirs('../modelos', exist_ok=True)
    if acc_rf >= acc_lr:
        print("\nRandom Forest es el mejor modelo. Guardando...")
        joblib.dump(modelo_rf, '../modelos/mejor_modelo.pkl')
    else:
        print("\nRegresión Logística es el mejor modelo. Guardando...")
        joblib.dump(modelo_lr, '../modelos/mejor_modelo.pkl')

    # Guardamos los datos de prueba para el script final
    X_test.to_csv('../data/X_test.csv', index=False)
    y_test.to_csv('../data/y_test.csv', index=False)

if __name__ == "__main__":
    print("Iniciando fase de entrenamiento...")
    entrenar_modelos()
    print("Entrenamiento finalizado.")