# PruebaI_CCO_MID_LES.py
# Cristhian Camilo Ocampo - C.C. 1152220729
# Maria Isabel Duque - C.C. 1037666575
# Leidy Estefania Silva - C.C. 1020479068

#EJERCICIO 1.
import numpy as np
import pandas as pd
import math 
pd.set_option('future.no_silent_downcasting', True)

def ejercicio_1_1():
    A = np.array([
    [2, 3, -1, 4, 5],
    [1, -2, 4, -3, 1],
    [3, 2, -3, 5, -2],
    [4, 1, -2, 3, 2],
    [1, 1, 1, 1, 1]
    ])
    b = np.array([15, 6, 11, 8, 7])
    solution = np.linalg.solve(A, b)
    return solution
print(f"Resultado del ejercicio_1_1: {ejercicio_1_1()}")

def ejercicio_1_2():
    precision=5
    eu_apro = 0
    n = 0
    while round(eu_apro, precision) != round(np.exp(1), precision):
        eu_apro += 1 / math.factorial(n)
        n += 1
    return  n
print(f"Resultado del ejercicio_1_2: {ejercicio_1_2()}")

def ejercicio_1_3():
    import numpy as np
    A = np.array([
    [2, 1, 1],
    [4, 3, 2],
    [1, 1, 2]
    ])
    autovalores = np.linalg.eigvals(A)
    return autovalores
print(f"Resultado del ejercicio_1_3: {ejercicio_1_3()}")

# EJERCICIO 2

def data_frame():
    exam_data = {
        'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, 10, 9, 20, 14.5, 12, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
    }
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    df = pd.DataFrame(exam_data, index=labels)
    return df

df = data_frame()

def ejercicio_2_1():
    return df[['name', 'score']]
print("Resultado del ejercicio_2_1:")
print(ejercicio_2_1())

def ejercicio_2_2():
    return df[df['attempts'] > 2]
print("Resultado del ejercicio_2_2:")
print(ejercicio_2_2())

def ejercicio_2_3():
    return df[(df['score']>=15) & (df['score']<=20)]
print("Resultado del ejercicio_2_3:")
print(ejercicio_2_3())

def ejercicio_2_4():
    df.loc['d', 'score'] = 11.5
    return df
print("Resultado del ejercicio_2_4:")
print(ejercicio_2_4())

def ejercicio_2_5():
    df.loc['k'] = ['Crimaes', 13.0, 2, 'no']
    return df
print("Resultado del ejercicio_2_5:")
print(ejercicio_2_5())

def ejercicio_2_6():
    df = ejercicio_2_5()
    df.drop('k',inplace=True, errors='ignore')
    return df
print("Resultado del ejercicio_2_6:")
print(ejercicio_2_6())

def ejercicio_2_7():
    return df.sort_values(by='name', ascending=True)
print("Resultado del ejercicio_2_7:")
print(ejercicio_2_7())

def ejercicio_2_8():
    df['qualify'] = df['qualify'].replace({'yes': True, 'no': False})
    return df
print("Resultado del ejercicio_2_8:")
print(ejercicio_2_8())

# EJERCICIO 3

def ejercicio_3_1():
    athletes = pd.read_csv('https://github.com/tomasate/Datos_Clases/blob/main/Datos_1/athletes.csv?raw=true')
    return athletes
print("Resultado del ejercicio_3_1:")
print(ejercicio_3_1())

def ejercicio_3_2():
    athletes = ejercicio_3_1()
    athletes_medallas = athletes[(athletes[['gold', 'silver', 'bronze']] > 0).any(axis=1)]
    return athletes_medallas
print("Resultado del ejercicio_3_2:")
print(ejercicio_3_2())

def ejercicio_3_3():
    athletes = ejercicio_3_2()
    athletes['muscle_mass'] = athletes['weight'] / (athletes['height'] ** 2)
    return athletes
print("Resultado del ejercicio_3_3:")
print(ejercicio_3_3())

def ejercicio_3_4():
    athletes = ejercicio_3_3()
    deportes_media = athletes.groupby('sport').agg({
    'weight': 'mean',
    'height': 'mean',
    'gold': 'mean',
    'silver': 'mean',
    'bronze': 'mean'
    })
    deportes_media = deportes_media.round(3)['weight'].idxmax()
    return f"{deportes_media}"
print(f"Resultado del ejercicio_3_4: {ejercicio_3_4()}")

def ejercicio_3_5():
    athletes = ejercicio_3_3()
    female_df = athletes[athletes['sex'] == 'female']
    part_female = female_df.groupby('nationality').size().idxmax()
    return f"{part_female}"
print(f"Resultado del ejercicio_3_5: {ejercicio_3_5()}")