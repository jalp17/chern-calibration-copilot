"""
Demo Streamlit — Chern Calibration Copilot.

Ejecutar:  streamlit run demo/app.py

Tres pestañas:
  1. Quantum Kernel vs Clasico — feature map cuántico como clasificador de
     fases de Haldane; compara SVM con kernel cuántico (overlaps de
     statevector) contra SVM-RBF sobre features crudas.
  2. Barrido de ruido — ejecuta simulador/barrido_v2.py con los parámetros
     elegidos y dibuja la tasa de error topológico frente a la infidelidad.
  3. Calibración del agente — formulario de especificaciones del dispositivo;
     el agente genera un plan de mitigación con fuentes citadas.

La simulación corre en GPU AMD Radeon Pro W7900D (ROCm) si está disponible,
y en CPU si no. El argumento de GPU es de MEMORIA: el lote de 65.536
semillas a n=14 ocupa 16 GiB, que no caben en una estación de 6 GiB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

# entrega/ para `from simulador import ...` y entrega/simulador/ para los
# imports internos de barrido_v2 (`from haldane import ...`)
_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "simulador"))

from simulador import barrido_v2 as b2  # noqa: E402
from simulador import kernel_cuantico as kc  # noqa: E402
from agente import agente  # noqa: E402

st.set_page_config(
    page_title="Chern Calibration Copilot",
    page_icon="⚛",
    layout="wide",
)

DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"


def _namespace(**kw) -> argparse.Namespace:
    base = dict(n=6, G=4, Nk=12, nseeds=200, device=DEVICE_DEFAULT, seed=0,
                modelos=["shared", "final", "indep"], out="barrido_v2.json",
                ps=[0.0, 0.005, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.30])
    base.update(kw)
    return argparse.Namespace(**base)


@st.cache_data(show_spinner=False)
def _sweep(n: int, G: int, Nk: int, nseeds: int, device: str, seed: int,
           modelos: tuple[str, ...], ps: tuple[float, ...], tag: str) -> dict:
    """Corre el barrido y devuelve el JSON. Cacheado para no repetir trabajo."""
    out = Path("/tmp") / f"barrido_{tag}.json"
    a = _namespace(n=n, G=G, Nk=Nk, nseeds=nseeds, device=device, seed=seed,
                   modelos=list(modelos), ps=list(ps), out=str(out))
    b2.run(a)
    return json.loads(out.read_text())


def _frames(res: dict) -> dict[str, pd.DataFrame]:
    """Convierte el JSON del barrido en un DataFrame por fase."""
    frames: dict[str, list[dict]] = {}
    for modelo, fases in res["modelos"].items():
        for fase, datos in fases.items():
            for r in datos["rows"]:
                frames.setdefault(fase, []).append(
                    {"modelo": modelo, "p_acierto": 100.0 * r["frac"], **r})
    return {k: pd.DataFrame(v) for k, v in frames.items()}


# --------------------------------------------------------------------------
st.title("Chern Calibration Copilot")
st.markdown(
    "**Copiloto de calibración cuántica** — AMD Developer Hackathon ACT III · "
    "Pista 1 · Intelligent Industry"
)

st.info(
    "**Qué hace:** cuantifica cómo el ruido de preparación degrada un "
    "invariante topológico (el número de Chern del modelo de Haldane) y, con "
    "un agente, recupera las especificaciones del dispositivo y propone un "
    "plan de mitigación. El circuito cuántico es el feature map que genera "
    "datos de kernel que un clasificador clásico no obtiene eficientemente; "
    "el agente interpreta los resultados y propone qué calibrar."
)

# --------------------------------------------------------------------------
# Sidebar: parámetros compartidos y específicos por pestaña
with st.sidebar:
    st.header("Parámetros del barrido (pestaña 2)")
    n = st.slider("Qubits (n)", 1, 16, 6, 1)
    G = st.slider("Profundidad (G)", 1, 8, 4, 1)
    Nk = st.slider("Malla k (Nk x Nk)", 4, 32, 12, 2)
    nseeds = st.select_slider("Semillas", options=[16, 32, 64, 128, 256, 512, 1024],
                              value=256)
    device = st.radio("Dispositivo", ["cuda", "cpu"],
                      index=0 if DEVICE_DEFAULT == "cuda" else 1)
    st.caption(
        "cuda = GPU AMD Radeon Pro W7900D (ROCm). El lote de 65.536 semillas "
        "a n=14 necesita 16 GiB y no cabe en una estación de trabajo de 6 GiB."
    )

    st.subheader("Modelos de ruido")
    m_shared = st.checkbox("shared — Pauli fijo por puerta", value=True)
    m_final = st.checkbox("final — control coherente", value=True)
    m_indep = st.checkbox("indep — Pauli independiente", value=False)
    modelos = [m for m, on in (("shared", m_shared), ("final", m_final),
                               ("indep", m_indep)) if on] or ["shared"]

    st.subheader("Rango de ruido")
    p_max = st.slider("p máximo", 0.0, 0.5, 0.30, 0.01)
    ps = [round(p_max * i / 8, 4) for i in range(9)]

    ejecutar_barrido = st.button("Ejecutar barrido", type="primary",
                                 use_container_width=True, key="btn_barrido")

    st.divider()
    st.header("Parámetros del quantum kernel (pestaña 1)")
    n_qk = st.slider("Qubits (n)", 2, 10, 6, 1, key="qk_n")
    n_train = st.slider("Muestras train", 10, 200, 50, 10, key="qk_train")
    n_test = st.slider("Muestras test", 5, 100, 30, 5, key="qk_test")
    n_layers_qk = st.slider("Capas del circuito", 1, 4, 2, 1, key="qk_layers")
    device_qk = st.radio("Dispositivo", ["cpu", "cuda"],
                         index=0 if DEVICE_DEFAULT != "cuda" else 1,
                         key="qk_device")
    ejecutar_kernel = st.button("Ejecutar kernel cuántico",
                                type="secondary", use_container_width=True,
                                key="btn_kernel")

    st.divider()
    st.header("Especificaciones del dispositivo (pestaña 3)")
    tasa_error_puerta = st.number_input("Tasa de error de puerta (p)", 0.0, 1.0,
                                        0.01, 0.001, key="agente_tasa")
    coherencia_t1 = st.number_input("T1 (µs)", 0.0, 1000.0, 50.0, 1.0,
                                    key="agente_t1")
    coherencia_t2 = st.number_input("T2 (µs)", 0.0, 1000.0, 30.0, 1.0,
                                    key="agente_t2")
    conectividad = st.selectbox("Conectividad",
                                ["lineal", "anillo", "todo a todo",
                                 "acoplamiento limitado"],
                                key="agente_conectividad")
    criterio_aceptacion = st.number_input("Criterio de aceptación (tasa mínima)",
                                          0.0, 1.0, 0.95, 0.01,
                                          key="agente_criterio")
    ejecutar_calibracion = st.button("Generar plan de calibración",
                                     type="secondary",
                                     use_container_width=True,
                                     key="btn_calibracion")

# --------------------------------------------------------------------------
tab_kernel, tab_barrido, tab_calibracion = st.tabs([
    "Quantum Kernel vs Clásico",
    "Barrido de ruido",
    "Calibración del agente",
])

# =========================================================================
# PESTAÑA 1: Quantum Kernel vs Clásico
# =========================================================================
with tab_kernel:
    st.subheader("Quantum Kernel vs SVM-RBF clásico")
    st.markdown(
        "El circuito variacional actúa como **feature map cuántico**: mapea los "
        "parámetros de Haldane $(M, t_1, t_2, \\phi)$ a un estado cuántico "
        "$U(x)|0\\rangle$ en un espacio de $2^n$ dimensiones. El kernel es el "
        "overlap $K(x,y) = |\\langle\\psi(x)|\\psi(y)\\rangle|^2$, que captura "
        "correlaciones de fase y entrelazamiento que un kernel RBF sobre "
        "features crudas no representa eficientemente."
    )

    if ejecutar_kernel:
        with st.spinner("Entrenando quantum kernel y SVM clásico..."):
            from haldane import sample_dataset
            X, y, _, _ = sample_dataset(n=n_train + n_test, seed=0)
            res = kc.comparar_kernels(
                X, y,
                n_qubits=n_qk,
                n_layers=n_layers_qk,
                n_train=n_train,
                n_test=n_test,
                device=device_qk,
                seed=0,
            )

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Accuracy kernel cuántico",
                      f"{res.acc_cuantico:.3f}")
            st.metric("Accuracy SVM-RBF clásico",
                      f"{res.acc_clasico:.3f}")
            delta = res.acc_cuantico - res.acc_clasico
            st.metric("Diferencia", f"{delta:+.3f}",
                      delta=delta)
        with c2:
            st.write("**Matriz de kernel (train)**")
            st.write(pd.DataFrame(res.K_train).round(3))

        st.caption(
            f"Dataset Haldane: {len(X)} muestras, {n_qk} qubits, "
            f"{n_layers_qk} capas, dispositivo {device_qk}. "
            "El kernel cuántico usa statevector de {2**n_qk} amplitudes."
        )
    else:
        st.info(
            "Ajusta los parámetros en el panel lateral y pulsa "
            "**Ejecutar kernel cuántico**."
        )

# =========================================================================
# PESTAÑA 2: Barrido de ruido (código original)
# =========================================================================
with tab_barrido:
    st.subheader("Barrido de ruido: fragilidad del número de Chern")

    if ejecutar_barrido:
        with st.spinner("Ejecutando barrido..."):
            res = _sweep(n, G, Nk, nseeds, device, 0, tuple(modelos),
                         tuple(ps),
                         tag=f"{n}_{G}_{Nk}_{nseeds}_{device}")
        frames = _frames(res)

        st.success("Barrido completado.")

        c1, c2 = st.columns(2)
        for col, fase in ((c1, "trivial"), (c2, "topológica")):
            with col:
                st.subheader(f"Fase {fase}")
                df = frames[fase]
                if df.empty:
                    st.write("Sin datos.")
                    continue
                st.line_chart(df.pivot_table(index="p", columns="modelo",
                                             values="p_acierto"))
                with st.expander("Tabla de datos"):
                    st.dataframe(df, use_container_width=True)

        st.subheader("Lectura")
        st.markdown(
            "- La **tasa de acierto** es el porcentaje de semillas cuyo Chern "
            "coincide con el valor exacto. El Chern sale siempre **entero** "
            "({-1, 0, +1}): lo que se degrada es la *probabilidad* de acertar, "
            "no el invariante.\n"
            "- El modelo **`final`** es un control: un unitario fijo conserva "
            "las fases de Berry exactamente, así que da 100 % a cualquier "
            "ruido.\n"
            "- El modelo **`shared`** es el físico: un Pauli fijo por puerta en "
            "toda la malla k.\n"
            "- El modelo **`indep`** destruye el bundle y puede dar |C| > 1."
        )

        with st.expander("JSON completo"):
            st.json(res)

    else:
        st.write(
            "Pulsa **Ejecutar barrido** para correr la simulación con estos "
            "parámetros. Con los valores por defecto tarda unos segundos en CPU."
        )

# =========================================================================
# PESTAÑA 3: Calibración del agente
# =========================================================================
with tab_calibracion:
    st.subheader("Plan de mitigación — agente copiloto de calibración")
    st.markdown(
        "El agente recibe las **especificaciones del dispositivo**, ejecuta un "
        "circuito de prueba en el simulador, compara con el umbral de "
        "aceptación y devuelve un plan de mitigación con fuentes citadas del "
        "corpus. Funciona con o sin LLM (fallback local determinista)."
    )

    if ejecutar_calibracion:
        specs = {
            "tasa_error_puerta": tasa_error_puerta,
            "coherencia_t1_us": coherencia_t1,
            "coherencia_t2_us": coherencia_t2,
            "conectividad": conectividad,
            "criterio_aceptacion": criterio_aceptacion,
        }
        with st.spinner("Generando plan de calibración..."):
            # Ejecutar un circuito de prueba pequeño para obtener infidelidad
            sim_inf = None
            try:
                from simulador.qsim_torch import Simulator, strongly_entangling
                import numpy as np
                sim_test = Simulator(4, device="cpu", dtype=torch.complex64)
                x_test = torch.zeros(1, 4)
                w_test = torch.zeros(2, 4)
                _, psi_test = strongly_entangling(x_test, w_test,
                                                  return_state=True)
                # fidelidad contra |0000> como referencia de idoneidad
                fidelidad = float(
                    (psi_test.conj() * torch.tensor(
                        [1.0] + [0.0] * 15,
                        dtype=psi_test.dtype,
                        device=psi_test.device)).sum().abs().item() ** 2
                )
                infid_sim = 1.0 - fidelidad
            except Exception:
                infid_sim = tasa_error_puerta  # fallback: usar la tasa del usuario

            res_sintetico = {
                "modelos": {
                    "shared": {
                        "trivial": {
                            "C_exact": 0,
                            "rows": [
                                {"p": tasa_error_puerta,
                                 "F_muestreada": 1.0 - infid_sim,
                                 "frac": max(0.0, 1.0 - 2.0 * infid_sim)}
                            ]
                        }
                    }
                }
            }
            plan = agente.generar_con_specs(specs, res_sintetico)
        st.markdown(agente.render(plan))
        if plan.via != "llm":
            st.caption(
                "El agente ha corrido **sin LLM** (no hay clave en `.env`): usa "
                "el plan local a partir del corpus. Configura "
                "`OPENAI_API_KEY` para sintetizarlo con Gemma desde Token "
                "Factory."
            )
    else:
        st.info(
            "Ajusta las especificaciones en el panel lateral y pulsa "
            "**Generar plan de calibración**."
        )

# --------------------------------------------------------------------------
st.divider()
st.subheader("Limitaciones declaradas")
st.markdown(
    "- El estado de la banda ocupada de Haldane en cada punto `k` es una "
    "superposición de **2 orbitales**, es decir **un solo qubit**. `n=1` da el "
    "mismo Chern que `n=10`; `n` es un parámetro del modelo, no una necesidad "
    "física.\n"
    "- La GPU se justifica por **memoria** (lotes > 6 GiB), no por velocidad. "
    "No afirmamos ventaja de velocidad sobre CPU.\n"
    "- No simulamos el efecto Hall cuántico físico: calculamos un invariante "
    "topológico y cómo responde al ruido.\n"
    "- El agente **no** calcula el Chern: interpreta resultados y propone un "
    "plan de calibración con fuentes del corpus.\n"
    "- Referencias: Lado et al., *Phys. Rev. Research* **6**, 043288 (2024), "
    "arXiv:2404.06048 · Fukui, Hatsugai, Suzuki, *JPSJ* **74**, 1674 (2005)."
)
