import pandas as pd
import numpy as np

url = 'https://raw.githubusercontent.com/higoramario/univesp-com360-mineracao-dados/main/dados-covid-limpeza.csv'

casos_covid = pd.read_csv(url)
casos_covid.head(20)

#print(casos_covid.info())
#print(casos_covid.describe())

#print(casos_covid["uf"].value_counts()) mostra os dados nates de tratar

casos_covid["uf"] = casos_covid["uf"].str.upper()

#print(casos_covid["uf"].value_counts()) #mostra os dados tratados

#print(casos_covid["renda"].value_counts())

#print(casos_covid["vacina"].value_counts())

casos_covid["idade"].fillna(round(casos_covid["idade"].mean()), inplace=True)

#print(casos_covid["idade"].value_counts())

casos_covid["uf"].fillna(method='ffill', inplace=True)

#print(casos_covid["uf"].value_counts())

casos_covid['renda'].fillna(method="bfill", inplace= True )

#print(casos_covid["renda"].value_counts())

casos_covid.info()

print(casos_covid.head(len(casos_covid)))

### Para treino em mais bases de dados consulte:

#Portal brasileiro de dados abertos
#    https://dados.gov.br/home

#Base dos dados
    #https://basedosdados.org/
    
#Kaggle
    #https://www.kaggle.com/
    
#UC Irvine Machine Learning Repository
#    https://archive.ics.uci.edu/dataset/53/iris