import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from functions import *

# ------------------ DATOS BASE ------------------

df = pd.read_excel('Component_Stocks.xlsx')
tickers_disponibles = sorted(set(df['Ticker'].tolist()))

st.title("Simulación de Portafolios de Inversión")

# ------------------ FORMULARIO ------------------

with st.form("configuracion"):

    tickers_seleccionados = st.multiselect(
        "Selecciona los tickers",
        options=tickers_disponibles,
        default=['AAPL', 'GOOG', 'MSFT']
    )

    start_date = st.date_input("Fecha de inicio", value=pd.to_datetime('2001-01-01'))
    end_date = st.date_input("Fecha de fin", value=pd.to_datetime('2025-01-01'))

    submitted = st.form_submit_button("Simular portafolio")


# ------------------ CACHE (NO REDESCARGA) ------------------

@st.cache_data(show_spinner="Descargando datos...")

def obtener_datos_cacheados(tickers, start, end):
    return obtener_datos(list(tickers), start, end)


# ------------------ EJECUCIÓN ------------------

if submitted and len(tickers_seleccionados) > 0:

    st.write(f"Tickers seleccionados: {tickers_seleccionados}")
    st.write(f"Rango de fechas: {start_date} a {end_date}")

    # DESCARGA SOLO AQUÍ
    tickers = tickers_seleccionados + ['SPY']


    all_data = obtener_datos_cacheados(tuple(tickers), start_date, end_date)

    datos = all_data.drop(columns='SPY')

    rendimientos = calcular_rendimientos(datos)

    resultados, pesos_portafolios = simular_portafolios(rendimientos)

    sim_out_df = pd.DataFrame(resultados, columns=['Portfolio_Return', 'Volatility', 'Sharpe_Ratio'])

    # ---------- PORTAFOLIO ÓPTIMO ----------
    idx_max_sharpe = np.argmax(resultados[:, 2])
    optimal_portfolio_return = resultados[idx_max_sharpe, 0]
    optimal_volatility = resultados[idx_max_sharpe, 1]
    optimal_sharpe_ratio = resultados[idx_max_sharpe, 2]
    optimal_weights = pesos_portafolios[idx_max_sharpe].round(2)

    tabla_pesos = pd.DataFrame({
        'Ticker': tickers_seleccionados,
        'Peso': optimal_weights
    })

    st.subheader("Pesos del Portafolio Óptimo")
    st.dataframe(tabla_pesos)

    st.subheader("Información del Portafolio Óptimo")
    st.write(f"**Rendimiento anualizado**: {optimal_portfolio_return*100:.2f}%")
    st.write(f"**Volatilidad anualizada**: {optimal_volatility*100:.2f}%")
    st.write(f"**Ratio de Sharpe**: {optimal_sharpe_ratio:.2f}")

    # ---------- RETORNO ACUMULADO ----------
    portafolio_acumulado = rendimientos.dot(optimal_weights).cumsum()*100

    start_common_date = portafolio_acumulado.index.min() - pd.Timedelta(days=1)
    datos_spy = obtener_datos_cacheados(('SPY',), start_common_date, end_date)
    datos_spy = calcular_rendimientos(datos_spy).cumsum()*100

    fig_ret_acumulado = go.Figure()

    fig_ret_acumulado.add_trace(go.Scatter(
        x=portafolio_acumulado.index,
        y=portafolio_acumulado,
        mode='lines',
        name='Portafolio Óptimo'
    ))

    fig_ret_acumulado.add_trace(go.Scatter(
        x=all_data.index,
        y=all_data['SPY'],
        mode='lines',
        name='SPY'
    ))

    fig_ret_acumulado.update_layout(
        title='Retorno Acumulado: Portafolio Óptimo vs S&P 500',
        xaxis_title='Fecha',
        yaxis_title='Retorno Acumulado (%)'
    )

    st.plotly_chart(fig_ret_acumulado)

    # ---------- FRONTERA EFICIENTE ----------
    slope = optimal_portfolio_return / optimal_volatility
    x_vals = [-optimal_volatility, 2 * optimal_volatility]
    y_vals = [slope * x for x in x_vals]

    fig = px.scatter(sim_out_df,
                     x='Volatility',
                     y='Portfolio_Return',
                     color='Sharpe_Ratio',
                     color_continuous_scale='Viridis')

    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', showlegend=False))
    fig.add_trace(go.Scatter(x=[optimal_volatility], y=[optimal_portfolio_return],
                             mode='markers', marker=dict(size=15)))

    fig.update_layout(
        title='Teoría Moderna del Portafolio y Frontera Eficiente',
        xaxis_title='Volatilidad',
        yaxis_title='Rendimiento Esperado'
    )

    st.plotly_chart(fig)

elif submitted:
    st.warning("Seleccioná al menos un ticker.")
