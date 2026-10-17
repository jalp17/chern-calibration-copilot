"""Demo Streamlit — Copiloto de Calibracion Cuantica.

Ejecutar: streamlit run demo/app.py
"""

import streamlit as st

st.set_page_config(page_title="Chern Calibration Copilot", page_icon="⚛")

st.title("Chern Calibration Copilot")
st.markdown("Diagnostico de calibracion cuantica — AMD Developer Hackathon ACT III")

st.info("Esqueleto pendiente de implementar.")

with st.sidebar:
    st.header("Parametros")
    p_noise = st.slider("Ruido de puerta (p)", 0.0, 0.1, 0.01, 0.001)
    device = st.selectbox("Dispositivo", ["cuda (GPU AMD)", "cpu"])
    st.write(f"Ruido: {p_noise:.3f}")
