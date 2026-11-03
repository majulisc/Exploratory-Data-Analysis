from sklearn.datasets import load_iris
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from tabulate import tabulate
# para instalar o backend interativo
import matplotlib
from pathlib import Path
pasta_graficos = Path("imagens")
pasta_graficos.mkdir(exist_ok=True)
matplotlib.use('TkAgg')
alfa = 0.05

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

#train_setosa,test_setosa=train_test_split(setosa,test_size=10,random_state=42)
''''print(train_setosa)
print("---------------------------------------------------------------------------------")
print(test_setosa)
print("//////////////////////////////////////////////////////////////////////////////////")

train_versicolor,test_versicolor=train_test_split(versicolor,test_size=10,random_state=42)
print(train_versicolor)
print("---------------------------------------------------------------------------------")
print(test_versicolor)
print("//////////////////////////////////////////////////////////////////////////////////")

#train_virginica,test_virginica=train_test_split(virginica,test_size=10,random_state=42)
#print(train_virginica)
#print("---------------------------------------------------------------------------------")
#print(test_virginica)'''

train_setosa,test_setosa=train_test_split(setosa,test_size=10,random_state=42)
train_versicolor,test_versicolor=train_test_split(versicolor,test_size=10,random_state=42)
train_virginica,test_virginica=train_test_split(virginica,test_size=10,random_state=42)

data = pd.concat([train_setosa,train_versicolor,train_virginica])
'''
data.info()
'''

print(data.describe().transpose())

#grafico
'''data.boxplot(grid = False, figsize= (10,8))
plt.title("Boxplot - Iris Flower data")
plt.show()


data.hist(grid=False,figsize=(10,10))
plt.show()'''



#print(data.columns) # estou usando para rememorar o nome das colunas
#cheking the data distribution
#analisando a distribuição normal esperada com histplot
"""sns.histplot(
    data=data,
    x='sepal length (cm)',
    stat='density',
    kde=False
)

x = np.linspace(
    data['sepal length (cm)'].min(),
    data['sepal length (cm)'].max(),
    100
)

media = data['sepal length (cm)'].mean()
desvio = data['sepal length (cm)'].std()

plt.plot(x, stats.norm.pdf(x, media, desvio))
plt.show()

"""

"""
#teste estatistico apenas para as sepalas
shapiro_stat, shapiro_p = stats.shapiro(data['sepal length (cm)']) #diz qual coluna sofre o teste no caso ->sepal length (cm)
#print("O teste estatistico shapiro é :" , shapiro_p)

# vou criar uma variavel para armazenar o alpha de modo que eu nao precise mudar o alfa de acordo com o valor que eu tenho interesse
#comparar  o valor de p com o nivel de significancia pre estabelecido
#o P representa o valor que é comparado ao alpha para entender se a amostra é normal ou nao

#basicamente estou marcando no grafico se é normal ou nao normal
if shapiro_p > alfa:
    auxiliar = "Dist. Normal (p=" + "{:.3}".format(shapiro_p) + ")"
    print(...)
else:
    auxiliar = "Não normal (p=" + "{:.3}".format(shapiro_p) + ")"
    print(...)

coluna = data['sepal length (cm)']
pontos_curva = np.linspace(coluna.min(), coluna.max(), 100)
curva = stats.norm.pdf(pontos_curva, coluna.mean(), coluna.std())

# Gráfico 1: histplot apenas para as sepalas
plt.figure(figsize=(10, 8))
sns.histplot(coluna, stat='density', kde=False, label=auxiliar)
plt.plot(pontos_curva, curva, label='Distribuição normal')
plt.legend()
plt.savefig('histograma_histplot.png', dpi=150, bbox_inches='tight')

# Gráfico 2: displot
grafico = sns.displot(data=data, x='sepal length (cm)', stat='density', kde=False,
                      height=8, aspect=1.25)
grafico.ax.plot(pontos_curva, curva, label='Distribuição normal')
grafico.ax.legend()
grafico.savefig('histograma_displot.png', dpi=150, bbox_inches='tight')

plt.show()   
"""

#Loop to iterative over all the characteristics of the iris flower
column_names = data.columns
for i in range(len(column_names) - 2):

    coluna = data[column_names[i]]

    # Teste estatístico de Shapiro-Wilk
    shapiro_stat, shapiro_p = stats.shapiro(coluna)

    if shapiro_p > alfa:
        auxiliar = "Dist. Normal (p=" + "{:.3g}".format(shapiro_p) + ")"
        print(f"{column_names[i]}: não rejeitamos a hipótese de normalidade.")
    else:
        auxiliar = "Não normal (p=" + "{:.3g}".format(shapiro_p) + ")"
        print(f"{column_names[i]}: rejeitamos a hipótese de normalidade.")

    print(f"Estatística W: {shapiro_stat:.4f}")
    print(f"P-valor: {shapiro_p:.4f}")

    # Preparação da curva normal
    pontos_curva = np.linspace(coluna.min(), coluna.max(), 100)

    curva = stats.norm.pdf(
        pontos_curva,
        coluna.mean(),
        coluna.std()
    )

    # Nome da característica para salvar o arquivo
    nome_coluna = (
        column_names[i]
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    # Gráfico 1: histplot
    plt.figure(figsize=(10, 8))

    sns.histplot(
        coluna,
        stat='density',
        kde=False,
        label=auxiliar
    )

    plt.plot(
        pontos_curva,
        curva,
        label='Distribuição normal'
    )

    plt.title(f"Histplot - {column_names[i]}")
    plt.legend()

    plt.savefig(
        pasta_graficos / f"histplot_{nome_coluna}.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    # Gráfico 2: displot
    grafico = sns.displot(
        data=data,
        x=column_names[i],
        stat='density',
        kde=False,
        height=8,
        aspect=1.25
    )

    grafico.ax.plot(
        pontos_curva,
        curva,
        label='Distribuição normal'
    )

    grafico.ax.set_title(f"Displot - {column_names[i]}")
    grafico.ax.legend()

    grafico.savefig(
        pasta_graficos / f"displot_{nome_coluna}.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(grafico.figure)
