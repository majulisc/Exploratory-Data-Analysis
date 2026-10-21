from sklearn.datasets import load_iris
import pandas as np
import pandas as pd
import scipy.stats as stats
import matplotlib as plt
import seaborn as sns 
import numpy as np

iris = load_iris()
x = iris.data 
y = iris.target

feature_names = iris.feature_names 
target_names = iris.target_names

print("nomes das features ( ou variaveis de entrada do iris)", feature_names)
print("nome das classes ( variaveis alvo)", target_names)
print("formato dos dados (x) : ", x.shape)

#DataFrame via pandas as pn
df_iris= pd.DataFrame (x, columns = feature_names)
df_iris['target'] = y

df_iris ['species'] = df_iris['target'].map({0: target_names[0],
                                             1: target_names[1],
                                             2: target_names[2]})
print("\nPrimeiras linhas do DataFrame:")
print(df_iris.head())
