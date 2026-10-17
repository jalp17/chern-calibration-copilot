"""Demo Streamlit — Chern Calibration Copilot.

Ejecutar:  streamlit run demo/app.py

La demo corre el barrido real (simulador/barrido_v2.py) con los parametros
elegidos en el panel lateral y dibuja la tasa de error topologico frente a
la infidelidad de preparacion.

Ejecuta en GPU AMD Radeon Pro W7900D (ROCm) si esta disponible, y en CPU si
no. El argumento de GPU es de MEMORIA: el lote de 65.536 semillas a n=14
ocupa 16 GiB, que no caben en una estacion de trabajo de 6 GiB.
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
    """Convierte el JSON del barrido en un DataFrame por fase.

    El JSON es `modelos -> fases -> rows`; aqui se transpone a
    `fases -> DataFrame` con una columna por modelo.
    """
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
    "**Diagnostico de calibracion cuantica** — AMD Developer Hackathon ACT III · "
    "Pista 1 · Intelligent Industry"
)

st.info(
    "**Que hace:** cuantifica como el ruido de preparacion degrada un "
    "invariante topologico (el numero de Chern del modelo de Haldane) y, con un "
    "agente, recupera las especificaciones del dispositivo y propone un plan de "
    "mitigacion. El agente **no** calcula el Chern: interpreta resultados."
)

with st.sidebar:
    st.header("Parametros del barrido")
    n = st.slider("Qubits (n)", 1, 16, 6, 1)
    G = st.slider("Profundidad (G)", 1, 8, 4, 1)
    Nk = st.slider("Malla k (Nk x Nk)", 4, 32, 12, 2)
    nseeds = st.select_slider("Semillas", options=[16, 32, 64, 128, 256, 512, 1024],
                              value=256)
    device = st.radio("Dispositivo", ["cuda", "cpu"],
                      index=0 if DEVICE_DEFAULT == "cuda" else 1)
    st.caption(
        "cuda = GPU AMD Radeon Pro W7900D (ROCm). El lote de 65.536 semillas "
        "a n=14 necesita 16 GiB y no cabe en una estacion de trabajo de 6 GiB."
    )

    st.subheader("Modelos de ruido")
    m_shared = st.checkbox("shared — Pauli fijo por puerta", value=True)
    m_final = st.checkbox("final — control coherente", value=True)
    m_indep = st.checkbox("indep — Pauli independiente", value=False)
    modelos = [m for m, on in (("shared", m_shared), ("final", m_final),
                               ("indep", m_indep)) if on] or ["shared"]

    st.subheader("Rango de ruido")
    p_max = st.slider("p maximo", 0.0, 0.5, 0.30, 0.01)
    ps = [round(p_max * i / 8, 4) for i in range(9)]

    ejecutar = st.button("Ejecutar barrido", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
if ejecutar:
    with st.spinner("Ejecutando barrido..."):
        res = _sweep(n, G, Nk, nseeds, device, 0, tuple(modelos), tuple(ps),
                     tag=f"{n}_{G}_{Nk}_{nseeds}_{device}")
    frames = _frames(res)

    st.success("Barrido completado.")

    c1, c2 = st.columns(2)
    for col, fase in ((c1, "trivial"), (c2, "topologica")):
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
        "({-1, 0, +1}): lo que se degrada es la *probabilidad* de acertar, no "
        "el invariante.\n"
        "- El modelo **`final`** es un control: un unitario fijo conserva las "
        "fases de Berry exactamente, asi que da 100 % a cualquier ruido.\n"
        "- El modelo **`shared`** es el fisico: un Pauli fijo por puerta en "
        "toda la malla k.\n"
        "- El modelo **`indep`** destruye el bundle y puede dar |C| > 1."
    )

    with st.expander("JSON completo"):
        st.json(res)

    st.divider()
    st.subheader("Plan de mitigacion (agente)")
    with st.spinner("Generando plan..."):
        plan = agente.generar(res)
    st.markdown(agente.render(plan))
    if plan.via != "llm":
        st.caption(
            "El agente ha corrido **sin LLM** (no hay clave en `.env`): usa el "
            "plan local a partir del corpus. Configura `OPENAI_API_KEY` para "
            "sintetizarlo con Gemma desde Token Factory."
        )
else:
    st.write(
        "Pulsa **Ejecutar barrido** para correr la simulacion con estos "
        "parametros. Con los valores por defecto tarda unos segundos en CPU."
    )

# --------------------------------------------------------------------------
st.divider()
st.subheader("Limitaciones declaradas")
st.markdown(
    "- El estado de la banda ocupada de Haldane en cada punto `k` es una "
    "superposicion de **2 orbitales**, es decir **un solo qubit**. `n=1` da el "
    "mismo Chern que `n=10`; `n` es un parametro del modelo, no una necesidad "
    "fisica.\n"
    "- La GPU se justifica por **memoria** (lotes > 6 GiB), no por velocidad.\n"
    "- No simulamos el efecto Hall cuantico fisico: calculamos un invariante "
    "topologico y como responde al ruido.\n"
    "- Referencias: Lado et al., *Phys. Rev. Research* **6**, 043288 (2024), "
    "arXiv:2404.06048 · Fukui, Hatsugai, Suzuki, *JPSJ* **74**, 1674 (2005)."
)
