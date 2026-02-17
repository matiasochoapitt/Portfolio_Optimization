from functions import *

df = pd.read_excel('/Users/matiasochoa/Documents/GitHub/Portfolio_Optimization/Component_Stocks.xlsx')

tickers = df['Ticker'].tolist()

tickers = set(tickers)
tickers = list(set(tickers))

tickers_disponibles = tickers



# Título de la aplicación
st.title("Simulación de Portafolios de Inversión")

# Selección de tickers
tickers_seleccionados = st.multiselect(
    "Selecciona los tickers",
    options=tickers_disponibles,
    default=['AAPL', 'GOOG', 'MSFT']  # Valor por defecto
)

# Selector de rango de fechas
start_date = st.date_input("Fecha de inicio", value=pd.to_datetime('2001-01-01'))
end_date = st.date_input("Fecha de fin", value=pd.to_datetime('2025-01-01'))

# Mostrar los tickers seleccionados y las fechas
st.write(f"Tickers seleccionados: {tickers_seleccionados}")
st.write(f"Rango de fechas: {start_date} a {end_date}")

# Descargar y procesar los datos
datos = obtener_datos(tickers_seleccionados, start_date, end_date)
rendimientos = calcular_rendimientos(datos)

# Realizar la simulación de portafolios
resultados, pesos_portafolios = simular_portafolios(rendimientos)

# Convertir resultados a un DataFrame
sim_out_df = pd.DataFrame(resultados, columns=['Portfolio_Return', 'Volatility', 'Sharpe_Ratio'])

# Extraer el portafolio con el mayor ratio de Sharpe
idx_max_sharpe = np.argmax(resultados[:, 2])
optimal_portfolio_return = resultados[idx_max_sharpe, 0]
optimal_volatility = resultados[idx_max_sharpe, 1]
optimal_sharpe_ratio = resultados[idx_max_sharpe, 2]
optimal_weights = pesos_portafolios[idx_max_sharpe].round(2)  # Pesos del portafolio óptimo

# Crear la tabla con los pesos de los tickers en el portafolio óptimo
tabla_pesos = pd.DataFrame({
    'Ticker': tickers_seleccionados,
    'Peso': optimal_weights
})

# Mostrar la tabla con los pesos
st.subheader("Pesos del Portafolio Óptimo")
st.dataframe(tabla_pesos)

# Información del portafolio óptimo
st.subheader("Información del Portafolio Óptimo")
st.write(f"**Rendimiento anualizado**: {optimal_portfolio_return*100:.2f}%")
st.write(f"**Volatilidad anualizada**: {optimal_volatility*100:.2f}%")
st.write(f"**Ratio de Sharpe**: {optimal_sharpe_ratio:.2f}")

# Calcular el retorno acumulado del portafolio óptimo
portafolio_acumulado = rendimientos.dot(optimal_weights).cumsum()*100

# Descargar el dato de SPY y calcular su retorno acumulado
start_common_date = portafolio_acumulado.index.min() - pd.Timedelta(days=1)
datos_spy = obtener_datos(['SPY'], start_common_date, end_date)
datos_spy = calcular_rendimientos(datos_spy)
datos_spy = datos_spy.cumsum()
datos_spy['SPY'] = datos_spy['SPY']*100

# Graficar los retornos acumulados
fig_ret_acumulado = go.Figure()

# Agregar el gráfico del retorno acumulado del portafolio
fig_ret_acumulado.add_trace(go.Scatter(
    x=portafolio_acumulado.index,
    y=portafolio_acumulado,
    mode='lines',
    name='Portafolio Óptimo',
    line=dict(color='blue')
))

# Agregar el gráfico del retorno acumulado de SPY
fig_ret_acumulado.add_trace(go.Scatter(
    x=datos_spy.index,
    y=datos_spy['SPY'],
    mode='lines',
    name='SPY',
    line=dict(color='green')
))

# Configuración del gráfico
fig_ret_acumulado.update_layout(
    title='Retorno Acumulado: Portafolio Óptimo vs S&P 500',
    xaxis_title='Fecha',
    yaxis_title='Retorno Acumulado (%)',
    plot_bgcolor="white"
)

# Mostrar el gráfico
st.plotly_chart(fig_ret_acumulado)

# Crear el gráfico interactivo con Plotly para la simulación de portafolios
slope = optimal_portfolio_return / optimal_volatility
x_vals = [-optimal_volatility, 2 * optimal_volatility]
y_vals = [slope * x for x in x_vals]


import plotly.express as px
import plotly.graph_objects as go

fig = px.scatter(sim_out_df, 
                 x='Volatility', 
                 y='Portfolio_Return', 
                 color='Sharpe_Ratio', 
                 hover_data=['Sharpe_Ratio'], 
                 color_continuous_scale='Viridis')


fig.add_trace(go.Scatter(
    x=x_vals, 
    y=y_vals, 
    mode='lines', 
    name='Línea Extendida', 
    line=dict(color='grey'),
    showlegend=False,
     ))


fig.add_trace(go.Scatter(
    x=[optimal_volatility], 
    y=[optimal_portfolio_return], 
    mode='markers', 
    name='Optimal Point', 
    marker=dict(size=15, color='grey'),
))

fig.update_layout(
    title='Teoria Moderna del Portafolio y Frontera Eficiente',
    xaxis_title='Volatilidad',
    yaxis_title='Rendimiento Esperado',
    plot_bgcolor="white",
    coloraxis_colorbar=dict(y=0.7, dtick=5),
    xaxis=dict(range=[sim_out_df['Volatility'].min()*0.9, sim_out_df['Volatility'].max()]),
    yaxis=dict(range=[sim_out_df['Portfolio_Return'].min()*0.9, sim_out_df['Portfolio_Return'].max()])
)


st.plotly_chart(fig)
