import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

def evaluar_modelo():
    # Cargar el modelo guardado y los datos de prueba
    modelo = joblib.load('../modelos/mejor_modelo.pkl')
    X_test = pd.read_csv('../data/X_test.csv')
    y_test = pd.read_csv('../data/y_test.csv')['quality']
    
    # Hacer predicciones
    predicciones = modelo.predict(X_test)
    
    print("--- Reporte Final de Métricas ---")
    # classification_report calcula internamente accuracy, precision, recall y f1
    print(classification_report(y_test, predicciones, zero_division=0))

    
    # Generar y guardar Matriz de Confusión
    cm = confusion_matrix(y_test, predicciones)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    
    os. makedirs('../graficos', exist_ok=True)
    
    plt.savefig('../graficos/matriz_confusion.png')
    plt.close()
    
    print("\nMatriz de confusión generada y guardada en 'graficos/matriz_confusion.png'.")

if __name__ == "__main__":
    print("Iniciando evaluación del modelo...")
    evaluar_modelo()
    print("Prueba finalizada.")