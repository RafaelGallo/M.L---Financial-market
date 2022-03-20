#!/usr/bin/env python
# coding: utf-8

# # Modelo - Série temporal StarLink 

# - Nesse modelo série temporal com ações da starlink são de 6 mês para previsão.

# In[1]:


# Versão do python

from platform import python_version

print('Versão python neste Jupyter Notebook:', python_version())


# In[2]:


# Importação das bibliotecas 

# Pandas carregamento csv
import pandas as pd 

# Numpy para carregamento cálculos em arrays multidimensionais
import numpy as np 

# Visualização de dados
import seaborn as sns
import matplotlib as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Carregar as versões das bibliotecas
import watermark

# Warnings retirar alertas 
import warnings
warnings.filterwarnings("ignore")


# In[4]:


# Configuração para os gráficos largura e layout dos graficos

plt.rcParams["figure.figsize"] = (25, 20)

plt.style.use('fivethirtyeight')
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

m.rcParams['axes.labelsize'] = 25
m.rcParams['xtick.labelsize'] = 25
m.rcParams['ytick.labelsize'] = 25
m.rcParams['text.color'] = 'k'


# In[3]:


# Versões das bibliotecas

get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Versões das bibliotecas" --iversions')


# # Base dados

# In[5]:


df = pd.read_csv("SLNV2-USD.csv")
df


# In[6]:


# Exibido 5 primeiros dados
df.head()


# In[7]:


# Exibido 5 últimos dados 
df.tail()


# In[8]:


# Número de linhas e colunas
df.shape


# In[9]:


# Verificando informações das variaveis
df.info()


# In[10]:


# Exibido tipos de dados
df.dtypes


# In[11]:


# Total de colunas e linhas 

print("Números de linhas: {}" .format(df.shape[0]))
print("Números de colunas: {}" .format(df.shape[1]))


# In[12]:


# Exibindo valores ausentes e valores únicos

print("\nMissing values :  ", df.isnull().sum().values.sum())
print("\nUnique values :  \n",df.nunique())


# In[13]:


# Sum() Retorna a soma dos valores sobre o eixo solicitado
# Isna() Detecta valores ausentes

df.isna().sum()


# In[14]:


# Retorna a soma dos valores sobre o eixo solicitado
# Detecta valores não ausentes para um objeto semelhante a uma matriz.

df.notnull().sum()


# In[16]:


# Total de número duplicados

df.duplicated()


# # Estatística descritiva

# In[17]:


# Exibindo estatísticas descritivas visualizar alguns detalhes estatísticos básicos como percentil, média, padrão, etc. 
# De um quadro de dados ou uma série de valores numéricos.

df.describe().T


# In[18]:


# Gráfico distribuição normal
plt.figure(figsize=(18.2, 8))

ax = sns.distplot(df['High']);
plt.title("Distribuição normal", fontsize=20)
plt.xlabel("Umidade")
plt.ylabel("Total")
plt.axvline(df['High'].mean(), color='b')
plt.axvline(df['High'].median(), color='r')
plt.axvline(df['High'].mode()[0], color='g');
plt.legend(["Media", "Mediana", "Moda"])
plt.show()


# In[19]:


# Matriz correlação de pares de colunas, excluindo NA / valores nulos.
df.corr()


# In[20]:


# Gráfico da matriz de correlação

plt.figure(figsize=(20,11))
ax = sns.heatmap(df.corr(), annot=True, cmap='YlGnBu');
plt.title("Matriz de correlação")


# In[22]:


# Matriz de correlação interativa 
fig = px.imshow(df.iloc[:, 1:].corr())
fig.show()


# # Análise de dados

# In[26]:


# Cálculo da média movel

media_alta = df[['Date', 'High']].groupby('Date').mean()
media_baixa = df[["Date", "Low"]].groupby('Date').mean()

print("Média de média alta", media_alta)
print()
print("Média de média media baixa", media_baixa)


# In[28]:


# Gráfico média movel - Humidity e Wind speed

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(50.5, 25));
plt.rcParams['font.size'] = '25'

ax1.plot(media_alta, marker='o', color = 'blue', markersize = 15);
ax1.set(title="Média móvel - ações alta", xlabel = "Date", ylabel = "Humidity")

ax2.plot(media_baixa, marker='o', color = 'blue', markersize = 15);
ax2.set(title="Média móvel - ações baixa", xlabel="Date", ylabel="Wind speed")


# In[31]:


# Gráfico ações em alta
fig = px.line(df, x="Date", y="High", title="Ações em alta")
fig.show()


# In[32]:


# Gráfico da ações em baixo
fig = px.line(df, x="Date", y="Low", title="Ações em baixo")
fig.show()


# In[33]:


# Gráfico da ações em fechado
fig = px.line(df, x="Date", y="Close", title="Ações em fechado")
fig.show()


# In[35]:


fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

fig.show()


# In[42]:


fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'])
                     ])

fig.update_layout(
    title='Açãoes da Starlink',
    yaxis_title='Total',
    xaxis_title='Starlink Stock',
    xaxis_rangeslider_visible=False)
fig.show()


# In[45]:


fig = go.Figure(data=[go.Candlestick(
    x=df['Date'],
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'],
    increasing_line_color= 'cyan', decreasing_line_color= 'gray'
)])

fig.update_layout(
    title='Açãoes da Starlink',
    yaxis_title='Total',
    xaxis_title='Starlink Stock',
    xaxis_rangeslider_visible=False)
fig.show()


# In[63]:


plt.figure(figsize=(15,6))
df['Open'].plot(color='r')
df['Close'].plot(color='g')
plt.xlabel('')
plt.title('Índice ação (Abertura e Fechamento)',size=15)
plt.legend()
plt.show()


# In[65]:


plt.figure(figsize=(15,6))
df['High'].plot()
df['Low'].plot()
plt.xlabel('')
plt.title('Índice ação em alta, baixa)',size=15)
plt.legend()
plt.show()


# # Análise de dados = Univariada

# In[53]:


# Fazendo um comparativo dos dados 

df.hist(bins = 40, figsize=(50.2, 20))
plt.title("Gráfico de histograma")
plt.show()


# In[54]:


# Plot total
df.plot(subplots=True, figsize=(20.5, 18))
plt.show()


# # Decomposição Sazonal

# In[60]:


dateparse= lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d')
df=pd.read_csv('SLNV2-USD.csv',parse_dates=['Date'],index_col='Date',date_parser=dateparse)
df


# In[72]:


base = df["Close"]
base


# In[75]:


# Importação da biblioteca decomposição sazonal
from statsmodels.tsa.seasonal import seasonal_decompose

# Decomposição aditiva
sd = seasonal_decompose(base, freq = 12)
sd.plot()
plt.show()


# In[76]:


# Padrão de tendência extraído
dt = sd.trend

dt.plot(figsize=(10.5, 8))


# In[78]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(figsize=(20,8))
plot_acf(base, lags=25, zero=False, ax=ax)
plt.show()


# In[80]:


# Média movel
media_movel = base - base.rolling(20).mean()
media_movel = media_movel.dropna()

# Gráfico - Autocorrelation
fig, ax1 = plt.subplots(figsize=(20.5, 8))
plot_acf(media_movel, lags = 20, zero = False, ax = ax1)
plt.show()


# In[81]:


# SARIMA

# Gráfico 
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(20.5, 18))
plot_acf(media_movel, lags=11, zero=False, ax=ax1)

# Gráfico PACF
plot_pacf(media_movel, lags=11, zero=False, ax=ax2)
plt.show()


# In[82]:


# ACF corta no lag 1. então, temos que usar o modelo MA.

media_movel = media_movel.diff(1).diff(12).dropna()
media_movel


# # Modelo ARIMA
# 
# **O ACF não sazonal não mostra nenhum dos padrões usuais dos modelos MA, AR ou ARMA, então não escolhemos nenhum deles. O Seaosnal ACF e PACF parecem um modelo MA(1). Selecionamos o modelo que combina ambos.**
# 
# - Modelo ARIMA 1

# In[83]:


# Modelo ARIMA
from pmdarima.arima import auto_arima

modelo_arima_auto = auto_arima(base,easonal = True, 
                               m = 25, d = 0, D = 1, max_p = 2, max_q = 2,
                               trace = True, error_action ='ignore',
                               suppress_warnings = True)


# - Modelo menor AIC e um pouco diferente anterior a componente sazonal Deltra e 1 ao invés 2

# In[84]:


# Modelo - Auto ARIMA
modelo_arima_auto


# In[86]:


# Modelo aic - Maior que anterior modelo
modelo_arima_auto.aic()


# In[88]:


# Súmario do modelo
print(modelo_arima_auto.summary())


# # Previsão do modelo

# In[90]:


modelo_arima_pred = modelo_arima_auto.predict(n_periods = 100)
modelo_arima_pred


# In[91]:


# Dataframe da previsão da ação

pred = pd.DataFrame(modelo_arima_pred, columns=["Previsão"])
pred


# In[92]:


pd.concat([pred.Previsão],axis=1).plot(linewidth=1, figsize=(20,5))

plt.legend(["Previsão"])
plt.xlabel('Previsão da temperatura')
plt.title('Previsão',size=15)
plt.show();


# # Model SARIMA
# 
# - SARIMA(2, 0, 2)x(2, 1, 0, 12) tem um desempenho melhor que outro modelo de ordens e tem baixo valor de AIC.
# - Divida o conjunto de trem e o conjunto de teste do conjunto de dados de trem e ajuste nosso modelo.

# In[170]:


# Modelo SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ajuste o modelo SARIMAX ao conjunto de treino
modelo_sarimax = SARIMAX(base, order = (2,0, 2), seasonal_order = (2, 1, 0, 12))

# Treinamento do modelo SARIMA
modelo_sarimax_fit = modelo_sarimax.fit()


# In[167]:


# Summary dos dados
print(modelo_sarimax_fit.summary())


# **Modelo SARIMAX**
# 
# - Prob(Q) é >0,05, então não rejeitamos a hipótese nula de que os resíduos não são correlacionados. Prob(JB) >0,05, então não rejeitamos a hipótese nula de que os resíduos não são normalmente distribuídos Assim, com base no resumo dado, os Resíduos não são correlacionados e normalmente distribuídos

# In[171]:


# 4 gráfico diagnóstico do modelo SARIMA

modelo_sarimax_fit.plot_diagnostics(figsize=(28.5, 25))
plt.show()


# **Standardized residul**
# 
# - O gráfico de resíduos padronizado informa que não há padrões óbvios nos resíduos A curva KDE é muito semelhante à distribuição normal. A maioria dos Datapoints está na linha reta. Além disso, correlações de 95% para atraso maior que um não são significativas Nosso modelo segue um comportamento padronizado. se não, temos que melhorar nosso modelo Prever os valores para o conjunto de teste

# In[172]:


# Prever os valores para o conjunto de teste

x_1 = len(x)
y_2 = len(x) + len(y) - 1

pred = modelo_sarimax_fit.predict(start = x_1, end = y_2)
pred


# In[179]:


# Previsão 

pred = modelo_sarimax_fit.predict(n_periods=150)
pred = pd.DataFrame(pred)
pred


# In[180]:


plt.plot(pred["predicted_mean"])
plt.title("Previsão modelo SARIMA - Ação")
plt.xlabel("Ação")
pred.plot(label='Previsão')


# # Métricas para o modelo
# 
# - RMSE: Raiz do erro quadrático médio 
# - MAE: Erro absoluto médio  
# - MSE: Erro médio quadrático
# - MAPE: Erro Percentual Absoluto Médio
# - R2: O R-Quadrado, ou Coeficiente de Determinação, é uma métrica que visa expressar a quantidade da variança dos dados.

# In[177]:


from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(base, pred))
mae = mean_absolute_error(base, pred)
mape = mean_absolute_percentage_error(base, pred)
mse = mean_squared_error(base, pred)
r2 = r2_score(base, pred)

pd.DataFrame([rmse, mae, mse, mape, r2], ['RMSE', 'MAE', 'MSE', "MAPE",'R²'], columns=['Resultado'])


# In[ ]:




