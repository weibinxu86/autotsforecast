import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autotsforecast import AutoForecaster
from autotsforecast.models.base import MovingAverageForecaster, VARForecaster
from autotsforecast.models.external import ARIMAForecaster, ETSForecaster

st.set_page_config(page_title='Quick Forecast', layout='wide')
st.title('📈 Quick Time Series Forecast')

with st.sidebar:
    st.header('Settings')
    horizon = st.slider('Forecast horizon (days)', 7, 60, 14)
    model_choice = st.selectbox('Model', ['Auto (best)', 'Moving Average', 'ARIMA', 'ETS'])

uploaded = st.file_uploader('Upload a CSV (date index + numeric columns)', type='csv')

if uploaded:
    df = pd.read_csv(uploaded, index_col=0, parse_dates=True)
else:
    st.info('No file uploaded — using built-in demo data (2 products, 180 days).')
    np.random.seed(0)
    idx = pd.date_range('2024-01-01', periods=180, freq='D')
    df = pd.DataFrame({
        'product_a': 1000 + np.cumsum(np.random.randn(180)) * 5 + np.sin(np.linspace(0, 8*np.pi, 180)) * 40,
        'product_b':  500 + np.cumsum(np.random.randn(180)) * 3 + np.cos(np.linspace(0, 6*np.pi, 180)) * 25,
    }, index=idx)

st.subheader('Data preview')
st.dataframe(df.tail(10), use_container_width=True)

if st.button('🚀 Run forecast', type='primary'):
    with st.spinner('Training models…'):
        model_map = {
            'Moving Average': [MovingAverageForecaster(horizon=horizon)],
            'ARIMA':          [ARIMAForecaster(horizon=horizon)],
            'ETS':            [ETSForecaster(horizon=horizon)],
        }
        candidates = model_map.get(model_choice, [
            MovingAverageForecaster(horizon=horizon),
            VARForecaster(horizon=horizon),
            ARIMAForecaster(horizon=horizon),
            ETSForecaster(horizon=horizon),
        ])
        auto = AutoForecaster(candidate_models=candidates, metric='rmse', verbose=False)
        auto.fit(df)
        forecast = auto.forecast()
    st.success('Done!')

    fig, axes = plt.subplots(len(df.columns), 1, figsize=(12, 4 * len(df.columns)))
    if len(df.columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, df.columns):
        ax.plot(df.index, df[col], color='#60A5FA', linewidth=2, label='Historical')
        if col in forecast.columns:
            ax.plot(forecast.index, forecast[col], color='#F87171', linewidth=2.5, linestyle='--', label='Forecast')
        ax.set_title(col, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader('Forecast values')
    st.dataframe(forecast, use_container_width=True)

    st.download_button(
        label='⬇️ Download forecast CSV',
        data=forecast.to_csv(),
        file_name='forecast.csv',
        mime='text/csv',
    )
