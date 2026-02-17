import yfinance as yf
import numpy as np
import streamlit as st


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import warnings


warnings.filterwarnings('ignore')



# Función para descargar los datos históricos de los tickers
def obtener_datos(tickers, start_date, end_date):
    datos = yf.download(tickers, start=start_date, end=end_date, auto_adjust= False, threads=False)['Adj Close']
    return datos

# Función para calcular los rendimientos diarios
def calcular_rendimientos(datos):
    rendimientos = datos.pct_change().dropna() 
    return rendimientos

# Función para realizar la simulación de portafolios
def simular_portafolios(rendimientos, num_simulaciones=5000):
    num_activos = len(rendimientos.columns)
    resultados = np.zeros((num_simulaciones, 3))  # columnas: [rendimiento, volatilidad, ratio de Sharpe]
    pesos_portafolios = np.zeros((num_simulaciones, num_activos))  # Para almacenar los pesos
    
    for i in range(num_simulaciones):
        # Generar pesos aleatorios que sumen 1
        pesos = np.random.random(num_activos)
        pesos /= np.sum(pesos)
        
        # Calcular el rendimiento y volatilidad del portafolio
        rendimiento_portafolio = np.sum(pesos * rendimientos.mean()) * 252  # anualizado
        volatilidad_portafolio = np.sqrt(np.dot(pesos.T, np.dot(rendimientos.cov() * 252, pesos)))  # anualizado
        
        # Calcular el ratio de Sharpe (suponiendo tasa libre de riesgo = 0)
        ratio_sharpe = rendimiento_portafolio / volatilidad_portafolio
        
        # Guardar resultados
        resultados[i] = [rendimiento_portafolio, volatilidad_portafolio, ratio_sharpe]
        pesos_portafolios[i] = pesos  # Guardar los pesos correspondientes a este portafolio
    
    return resultados, pesos_portafolios