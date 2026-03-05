import pandas as pd
import numpy as np

def generar_caso_de_uso_agregar_datos_sensores():
    """
    Genera un caso de uso para la función agregar_datos_sensores.
    """
    inicio = pd.Timestamp('2026-03-05 08:00:00')
    n_puntos = 100
    offsets = np.random.choice(np.arange(0, 720), size=n_puntos, replace=False)
    tiempos = inicio + pd.to_timedelta(offsets, unit='m')
    
    df = pd.DataFrame({
        'timestamp': tiempos,
        'id_sensor': 'SN-001',
        'valor_sensor': np.random.uniform(20.0, 30.0, len(tiempos)),
        'energia_consumida': np.random.uniform(0.5, 2.0, len(tiempos))
    })
    
    df.loc[df.sample(frac=0.1).index, 'valor_sensor'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'energia_consumida'] = np.nan
    
    df_calc = df.copy()
    df_calc['timestamp'] = pd.to_datetime(df_calc['timestamp'])
    df_calc = df_calc.set_index('timestamp')
    
    resampled = df_calc.resample('15min').agg({
        'valor_sensor': 'mean',
        'energia_consumida': 'mean'
    })
    
    counts = df_calc.resample('15min').size()
    intervals_with_data = counts[counts > 0].index
    expected = resampled.loc[intervals_with_data]
    
    return {'df': df}, expected

if __name__ == "__main__":
    inputs, expected = generar_caso_de_uso_agregar_datos_sensores()
    print("--- Caso de Uso Generado (Muestra) ---")
    print(inputs['df'].head())
    print("\n--- Resultado Esperado (Muestra) ---")
    print(expected.head())
