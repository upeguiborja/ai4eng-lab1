import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.datasets import make_classification

def generar_caso_de_uso_propagar_etiquetas_semi_supervisadas():
    """
    Genera un caso de uso para la función propagar_etiquetas_semi_supervisadas.
    """
    # Generar un dataset sintético de clasificación
    X, y_true = make_classification(
        n_samples=200, n_features=5, n_informative=3, 
        n_redundant=0, random_state=42
    )
    
    # Ocultar el 80% de las etiquetas (ponerlas en -1)
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(y_true)) < 0.8
    y_masked = np.copy(y_true)
    y_masked[random_unlabeled_points] = -1
    
    kernel = 'knn'
    
    # Calcular el resultado esperado (Ground Truth)
    ls = LabelSpreading(kernel=kernel, n_neighbors=7)
    ls.fit(X, y_masked)
    expected = ls.transduction_
    
    return {'X': X, 'y': y_masked, 'kernel': kernel}, expected

if __name__ == "__main__":
    inputs, expected = generar_caso_de_uso_propagar_etiquetas_semi_supervisadas()
    print("--- Caso de Uso Generado ---")
    print(f"Forma de X: {inputs['X'].shape}")
    print(f"Etiquetas conocidas: {np.sum(inputs['y'] != -1)} de {len(inputs['y'])}")
    print(f"Primeras 10 etiquetas propagadas esperadas: {expected[:10]}")
