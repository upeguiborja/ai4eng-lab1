import pandas as pd
import numpy as np

def generar_caso_de_uso_sincronizar_señales():
    """
    Genera un caso de uso para la función sincronizar_señales.
    Retorna un diccionario con los argumentos de entrada y el valor esperado.
    """
    # Configuración
    n_puntos = 100
    t = np.linspace(0, 10, n_puntos)

    # Crear una señal base (ej. una combinación de senos) con algo de ruido
    señal_base = np.sin(t) + 0.5 * np.sin(2.5 * t)

    # Definir un offset aleatorio entre -15 y 15
    offset_real = np.random.randint(-15, 16)

    # señal_a es la base con ruido
    señal_a = señal_base + np.random.normal(0, 0.05, n_puntos)

    # señal_b es la señal_a desplazada y con ruido adicional
    # Usamos shift de pandas para crear el desfase y luego rellenamos con ruido o ceros
    # para que la correlación sea máxima en el punto exacto.
    s_a_series = pd.Series(señal_a)
    s_b_series = s_a_series.shift(offset_real)

    # Rellenar los NaNs generados por el shift con ruido
    # Generamos un array de ruido del mismo tamaño que los nans
    s_b_series = s_b_series.fillna(pd.Series(np.random.normal(0, 0.1, n_puntos)))
    señal_b = s_b_series.values + np.random.normal(0, 0.02, n_puntos)

    df = pd.DataFrame({
        'timestamp': t,
        'señal_a': señal_a,
        'señal_b': señal_b
    })

    # El output es el offset que maximiza la correlación.
    # Debido al ruido, lo calculamos exactamente para asegurar que el Ground Truth sea correcto.
    max_corr = -1.0
    best_offset = 0
    for k in range(-20, 21):
        current_corr = df['señal_a'].corr(df['señal_b'].shift(k))
        if current_corr > max_corr:
            max_corr = current_corr
            best_offset = k

    return {'df': df}, int(best_offset)

if __name__ == "__main__":
    input_data, expected_output = generar_caso_de_uso_sincronizar_señales()
    print("--- Caso de Uso Generado ---")
    print("Primeras 5 filas del DataFrame:")
    print(input_data['df'].head())
    print(f"Offset esperado: {expected_output}")
