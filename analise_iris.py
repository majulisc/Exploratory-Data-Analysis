from sklearn.datasets import load_iris
import pandas as np
import pandas as pd
import scipy.stats as stats
import matplotlib as plt
import seaborn as sns 
import numpy as np
from sklearn.model_selection import train_test_split

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
#print("\nPrimeiras linhas do DataFrame:")
#print(df_iris.head())
#print(df_iris.tail())
#print(df_iris.info())
#print(df_iris.species.unique())
#print(df_iris.describe().transpose())

setosa = df_iris.loc[df_iris['species'] == 'setosa']
versicolor = df_iris.loc[df_iris['species'] == 'versicolor']
virginica = df_iris.loc[df_iris['species'] == 'virginica']
#print(setosa.head())
#print("--------------------------------------------------------------------------------------------------")
#print(versicolor.tail())

train_setosa,test_setosa=train_test_split(setosa,test_size=10,random_state=42)
print(train_setosa)
print("---------------------------------------------------------------------------------")
print(test_setosa)
print("//////////////////////////////////////////////////////////////////////////////////")

train_versicolor,test_versicolor=train_test_split(versicolor,test_size=10,random_state=42)
print(train_versicolor)
print("---------------------------------------------------------------------------------")
print(test_versicolor)
print("//////////////////////////////////////////////////////////////////////////////////")

train_virginica,test_virginica=train_test_split(virginica,test_size=10,random_state=42)
print(train_virginica)
print("---------------------------------------------------------------------------------")
print(test_virginica)

