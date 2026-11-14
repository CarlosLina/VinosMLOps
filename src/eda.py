import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def cargar_y_limpiar_datos(ruta_csv):
    # Cargar los datos
    df = pd.read_csv(ruta_csv)
    print("--- Primeras filas del dataset ---")
    print(df.head())
    
    print("\n--- Estadísticas Descriptivas ---")
    print(df.describe())
    
    # Limpiar datos: La columna 'Id' no aporta valor predictivo, la eliminamos
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
        print("\nColumna 'Id' eliminada exitosamente.")
    
    # Revisión de nulos
    print("\n--- Valores nulos por columna ---")
    print(df.isnull().sum())
    
    return df

def visualizar_distribuciones(df):
    # Crear carpeta para guardar los gráficos si no existe
    os.makedirs('graficos', exist_ok=True)
    
    # 1. Histograma de la variable objetivo (Calidad del vino)
    plt.figure(figsize=(8, 5))
    sns.countplot(x='quality', data=df, palette='viridis')
    plt.title('Distribución de la Calidad del Vino')
    plt.savefig('graficos/distribucion_calidad.png')
    plt.close()
    
    # 2. Matriz de correlación
    plt.figure(figsize=(10, 8))
    correlacion = df.corr()
    sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matriz de Correlación')
    plt.savefig('graficos/matriz_correlacion.png')
    plt.close()
    print("Gráficos generados y guardados en la carpeta 'graficos/'.")

if __name__ == "__main__":
    print("Iniciando Análisis Exploratorio de Datos (EDA)...")
    # Asume que el CSV está en la raíz del repo, un nivel arriba de src/
    df_vinos = cargar_y_limpiar_datos('../WineQT.csv') 
    visualizar_distribuciones(df_vinos)
    
    # Guardamos el dataset limpio para los siguientes pasos
    os.makedirs('../data', exist_ok=True)
    df_vinos.to_csv('../data/WineQT_clean.csv', index=False)
    print("Dataset limpio guardado en 'data/WineQT_clean.csv'. EDA finalizado.")