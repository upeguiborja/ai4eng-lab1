import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

def generar_caso_de_uso_seleccionar_caracteristicas_por_permutacion():
    """
    Genera un caso de uso para la función seleccionar_caracteristicas_por_permutacion.
    """
    # Generar datos sintéticos
    n_samples = 150
    rng = np.random.RandomState(42)
    
    # 3 características informativas y 2 de ruido
    f1 = rng.randn(n_samples)
    f2 = rng.randn(n_samples)
    f3 = rng.randn(n_samples)
    f4 = rng.randn(n_samples) # Ruido
    f5 = rng.randn(n_samples) # Ruido
    
    # La etiqueta depende fuertemente de f1 y f2, un poco de f3
    y = (f1 + 2*f2 + 0.5*f3 + rng.randn(n_samples) * 0.5 > 0).astype(int)
    
    X = pd.DataFrame({
        'feature_A': f1,
        'feature_B': f2,
        'feature_C': f3,
        'noise_1': f4,
        'noise_2': f5
    })
    
    # Entrenar un modelo rápido
    modelo = RandomForestClassifier(n_estimators=10, random_state=42)
    modelo.fit(X, y)
    
    top_k = 2
    
    # Calcular Ground Truth
    result_pi = permutation_importance(modelo, X, y, n_repeats=5, random_state=42)
    # Obtener índices de mayor a menor importancia
    sorted_idx = result_pi.importances_mean.argsort()[::-1]
    expected = X.columns[sorted_idx[:top_k]].tolist()
    
    return {'modelo': modelo, 'X_val': X, 'y_val': y, 'top_k': top_k}, expected

if __name__ == "__main__":
    inputs, expected = generar_caso_de_uso_seleccionar_caracteristicas_por_permutacion()
    print("--- Caso de Uso Generado ---")
    print(f"Top {inputs['top_k']} características esperadas: {expected}")
