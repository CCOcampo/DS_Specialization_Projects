# PruebaII_CCO_MID_LES.py
# Cristhian Camilo Ocampo - C.C. 1152220729
# Maria Isabel Duque - C.C. 1037666575
# Leidy Estefania Silva - C.C. 1020479068

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

#EJERCICIO 1.
def ejercicio_1_1():
    iris = load_iris()
    datos = iris.data
    clases = iris.target_names
    colores_mapa = ['red', 'green', 'blue']

    x = datos[:, 2]  # Longitud del pétalo
    y = datos[:, 3]  # Grosor del pétalo

    for i, clase in enumerate(clases):
        plt.scatter(x[iris.target == i], y[iris.target == i], label=clase, color=colores_mapa[i])

    plt.xlabel('Longitud del pétalo')
    plt.ylabel('Grosor del pétalo')
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)  # No bloquea la ejecución
    plt.pause(0.001)
    plt.gcf().canvas.manager.set_window_title("Ejercicio 1.1 - Gráfico de dispersión de Iris")

ejercicio_1_1()


#EJERCICIO 2.1
def ejercicio_2_1():
    df = pd.read_csv('https://github.com/tomasate/Datos_Clases/blob/main/Datos_1/weather_2016_2020_daily.csv?raw=true')
    #Los primeros 12 dias de cada mes estan mal ordenados, se debe corregir con la siguiente funcion.
    def corregir_dia_mes(date_str):
        year, day, month = date_str.split('-')
        return f"{year}-{month}-{day}"

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce') 

    filas_a_corregir = (df['Date'].dt.day <= 12)
    df.loc[filas_a_corregir, 'Date'] = df.loc[filas_a_corregir, 'Date'].astype(str).apply(corregir_dia_mes)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%Y-%m-%d')
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors='coerce')
        df.set_index("Date", inplace=True) 
        
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    return df
print(f"Resultado del ejercicio_2_1:\n {ejercicio_2_1()}")

#EJERCICIO 2.2
def ejercicio_2_2():
    df_c=ejercicio_2_1()
    df_d = df_c.resample('10D').agg({
        "Temp_max": "max",
        "Temp_avg": "mean",
        "Temp_min": "min",
    })
    return df_d
print(f"Resultado del ejercicio_2_2:\n {ejercicio_2_2()}")

#EJERCICIO 2.3
def ejercicio_2_3():
    df_c=ejercicio_2_1()
    # Descomposición Aditiva
    decomp_additive = seasonal_decompose(df_c['Temp_avg'], model='additive', period=365)

    fig, ax = plt.subplots(4, 1, figsize=(15, 8), sharex=True)

    df_c['Temp_avg'].plot(ax=ax[0], title='Serie Original')
    decomp_additive.trend.plot(ax=ax[1], title='Tendencia')
    decomp_additive.seasonal.plot(ax=ax[2], title='Estacionalidad')
    decomp_additive.resid.plot(ax=ax[3], title='Ruido')

    plt.suptitle("Descomposición de la Serie Temporal - Additive", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.gcf().canvas.manager.set_window_title("Ejercicio 2.1 - Serie Temporal - Additive")
    plt.show(block=False)  # No bloquea la ejecución
    plt.pause(0.001)
    '''Las graficas de los modelos en esta serie pueden ser similares,
    ya que, la serie no presenta diferencias claras en el comportamiento estacional en relación con la tendencia.
    Sin embargo, se elige el modelo aditivo, ya que podemos ver que la amplitud de las fluctuaciones estacionales permanecen constantes,
    Además, es recomendable optar por modelos más simples cuando ambos presentan un ajuste similar.'''

ejercicio_2_3()


#EJERCICIO 3
def ejercicio_3():
    df = pd.read_csv('https://raw.githubusercontent.com/tomasate/Datos_Clases/refs/heads/main/Datos_1/hep_exe.csv', index_col='Unnamed: 0')

    df["PTx"] = 0
    df["PTy"] = 0

    for i in range(4):
        df[f"PTy{i}"] = df[f"muon_pt{i}"] * np.sin(df[f"muon_phi{i}"])
        df[f"PTx{i}"] = df[f"muon_pt{i}"] * np.cos(df[f"muon_phi{i}"])
        
        df["PTy"] += df[f"PTy{i}"]*-1
        df["PTx"] += df[f"PTx{i}"]*-1

    df_resultado = df[["PTx", "PTy"]].copy()

    return df_resultado

print(f"Resultado del ejercicio_3:\n {ejercicio_3()}")

#EJERCICIO 2.4
def ejercicio_2_4():
    df_c=ejercicio_2_1()
    df_c["Temp_avg_MA7"] = df_c["Temp_avg"].rolling(window=7).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(df_c.index, df_c["Temp_avg"], label="Temp_avg (Diario)", color="blue", alpha=0.3)
    plt.plot(df_c.index, df_c["Temp_avg_MA7"], label="Promedio Móvil (7 días)", color="red", linewidth=1)
    plt.title("Promedio Móvil de Temp_avg (7 días)", fontsize=14, fontweight="bold")
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Temperatura Promedio (°F)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.savefig("promedio_movil_temp_avg.png", dpi=300, bbox_inches="tight")
    plt.gcf().canvas.manager.set_window_title("Ejercicio 2.4 - Promedio Móvil de Temp_avg")
    plt.show(block=True)
ejercicio_2_4()