# Machine-learning-Financial-market



[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/R-3.6.0-red.svg)](https://www.r-project.org/)
[![](https://img.shields.io/badge/ggplot2-white.svg)](https://ggplot2.tidyverse.org/)
[![](https://img.shields.io/badge/dplyr-blue.svg)](https://dplyr.tidyverse.org/)
[![](https://img.shields.io/badge/readr-green.svg)](https://readr.tidyverse.org/)
[![](https://img.shields.io/badge/ggvis-black.svg)](https://ggvis.tidyverse.org/)
[![](https://img.shields.io/badge/Shiny-red.svg)](https://shiny.tidyverse.org/)
[![](https://img.shields.io/badge/plotly-green.svg)](https://plotly.com/)
[![](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/en/stable/#)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Caret-orange.svg)](https://caret.tidyverse.org/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)
[![](https://img.shields.io/badge/PowerBI-red.svg)](https://powerbi.microsoft.com/pt-br/)

![Logo](https://as2.ftcdn.net/v2/jpg/04/22/23/87/1000_F_422238712_Z2cdkhDgaLBqBkrvp5l0fyNtIlqhbzX0.jpg)


## Autores

- [@RafaelGallo](https://www.github.com/rafaelgallo)


## Objetivo do projeto de machine learning

Aplicando machine learning na área de finanças. Realizei alguns projetos de finanças como provisão de ações, lucros, previsão bitcoin útilizando modelos ARIMA, SARIMA, AR.
Projeto foi feito em python ou R.
Os projetos estão na pasta Jupyter Notebook. 


## Stack utilizada

**Programação** Python

**Leitura CSV**: Pandas

**Análise de dados**: Seaborn, Matplotlib

**Machine learning**: Scikit-learn





## Variáveis de Ambiente

Para rodar esse projeto, você vai precisar adicionar as seguintes variáveis de ambiente no seu .env

Instalando a virtualenv

`pip install virtualenv`

Nova virtualenv

`virtualenv nome_virtualenv`

Ativando a virtualenv

`source nome_virtualenv/bin/activate` (Linux ou macOS)

`nome_virtualenv/Scripts/Activate` (Windows)

Retorno da env

`projeto_py source venv/bin/activate` 

Desativando a virtualenv

`(venv) deactivate` 

Instalando pacotes

`(venv) projeto_py pip install flask`

Instalando as bibliotecas

`pip freeze`


## Instalação

Instalação das bibliotecas para esse projeto no python.

```bash
  conda install -c conda-forge pandas 
  conda install -c conda-forge scikitlearn
  conda install -c conda-forge numpy
  conda install -c conda-forge scipy

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```
    
## Demo modelo machine learning - Regressão linear

```bash
  # Carregando as bibliotecas 
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  # Carregando o dataset
  data = pd.read_csv("data.csv")
  
  # Visualizando os 5 primeiros itens
  data.head()

  # visualizando linhas e colunas com shape
  data.shape

  # Informações das variaveis
  data.info()

  # Definindo variaveis para modelo
  train = dados.iloc[:, 0:8].values
  test = dados.iloc[:, 1].values

  # Escalonamento dados
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  scaler.fit(valores_exames_v1)
  saler_fit = StandardScaler.transform(valores_exames_v1)

  # Treinamento do modelo
  from sklearn.model_selection import train_test_split
  X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

  # Modelo machine learning
  from sklearn.linear_model import LinearRegression
  modelo_regreesao = LinearRegression()
  modelo_regreesao_fit = modelo_regreesao.fit(x_train, y_train)
  modelo_regreesao_pred = modelo_regreesao.predict(x_test)
  print(modelo_regreesao_pred)

  # Score do modelo
  modelo_regreesao_score = modelo_regreesao.score(x_test, y_test)
  modelo_regreesao_score 

  # Coeficiente
  modelo_regreesao.coef_

  # Intercep
  modelo_regreesao.intercept_

  # Previsão modelo
  pred = modelo_regreesao.predict(x_train)
  y_pred = modelo_regreesao.predict(x_test)

  # Resultado Previsão
  df_data = pd.DataFrame(y_test, modelo_regreesao_pred)
  df_data_p = pd.DataFrame({"Previsão_da_ação": y_test, "Previsão":modelo_regreesao_pred})

  # Gráfico da regressão 
  plt.figure(figsize=(18, 8))
  sns.lineplot(x="Previsão_da_ação", y="Previsão", data = df_data_p)
  plt.title("Grafico de regressão linear - Google", fontsize = 20)
  plt.xlabel("Preço")
  plt.ylabel("Previsão")
  plt.legend(["Previsão"])
  plt.show()

  # Gráfico - Distribuição de Frequências dos Resíduos
  ax = sns.distplot(pred)
  ax.figure.set_size_inches(20, 8)
  ax.set_title('Distribuição de Frequências dos Resíduos', fontsize=18)
  plt.ylabel("Total de resíduos", fontsize = 18)
  plt.xlabel("Resíduos a frequência", fontsize = 18)
  plt.legend(["Resíduos"])
  plt.show()

  # Grafico de regressão linear
  plt.figure(figsize=(18, 8))
  plt.scatter(df_data_p.Previsão_da_ação, df_data_p.Previsão)
  plt.title("Grafico de regressão linear - Google", fontsize = 20)
  plt.xlabel("Preço")
  plt.ylabel("Previsão")
  plt.legend(["Previsão", "Preço"])
  plt.show()


  ## Metricas RMSE, MAE, MSE, r2
  from math import sqrt
  from sklearn import metrics
  from sklearn.metrics import mean_squared_error
  from sklearn.metrics import mean_absolute_error
  from sklearn.metrics import r2_score
  
  MSE = mean_squared_error(y_test, modelo_regreesao_pred)
  MAE = mean_absolute_error(y_test, modelo_regreesao_pred)
  r2 = r2_score(y_test, modelo_regreesao_pred)
  RMSE = np.sqrt(mean_squared_error(y_test, modelo_regreesao_pred))

  ## Salvando o modelo de Regressão linear
  
  import pickle
  
  with open('modelo_regreesao_pred.pkl', 'wb') as file:
    pickle.dump(modelo_regreesao_pred, file)


```


## Melhorias

Que melhorias você fez no seu código? 
- Ex: refatorações, melhorias de performance, acessibilidade, etc


## Suporte

Para suporte, mande um email para rafaelhenriquegallo@gmail.com

