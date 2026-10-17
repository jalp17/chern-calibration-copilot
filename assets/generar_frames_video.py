import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

DARK = '#0B0F19'
ACCENT = '#7C3AED'
os.makedirs('assets', exist_ok=True)

def texto_frame(nombre, titular, lineas, sub=None):
    """lineas: lista de (texto, color_hex, fontsize, fontweight)"""
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    ax.set_xlim(0, 12.8); ax.set_ylim(0, 7.2); ax.axis('off')
    ax.add_patch(mpatches.Rectangle((0, 6.6), 12.8, 0.6, facecolor='#ED1C24', edgecolor='none'))
    ax.text(0.4, 6.9, 'AMD Developer Hackathon ACT III · Pista 1 · Intelligent Industry',
            color='white', fontsize=12, fontweight='bold', va='center')
    ax.text(0.4, 5.6, titular, color='white', fontsize=26, fontweight='bold', va='center')
    ax.add_patch(mpatches.FancyArrowPatch((0.4, 5.0), (12.4, 5.0), arrowstyle='-',
                     color=ACCENT, lw=2, mutation_scale=0))
    y = 4.4
    for texto, color, fontsize, weight in lineas:
        ax.text(0.4, y, texto, color=color, fontsize=fontsize,
                fontweight=weight, va='center')
        y -= 0.62
    if sub:
        ax.text(0.4, 0.5, sub, color='#94A3B8', fontsize=10, va='center')
    plt.tight_layout()
    plt.savefig(f'assets/{nombre}.png', dpi=150, facecolor=DARK)
    plt.close()

texto_frame('f_problema', 'El problema',
    [('¿Cuánta infidelidad tolera el número de Chern?', '#FFFFFF', 15, 'bold'),
     ('El número de Chern clasifica fases de materia cuántica. Cuando se prepara el estado', '#CBD5E1', 13, 'normal'),
     ('con ruido, no se sabe cuándo el invariante deja de ser el valor exacto.', '#CBD5E1', 13, 'normal'),
     ('No hay respuesta publicada; los ordenadores cuánticos operan al 95–99 % de fidelidad.', '#CBD5E1', 13, 'normal')])

texto_frame('f_metodo', 'Metodología',
    [('Simulador statevector en PyTorch/ROCm, validado contra PennyLane a 10⁻¹⁵.', '#CBD5E1', 13, 'normal'),
     ('Preparamos la banda ocupada del modelo de Haldane con ruido depolarizante por puerta.', '#CBD5E1', 13, 'normal'),
     ('Medimos el Chern con Fukui–Hatsugai–Suzuki y probamos 3 modelos de ruido.', '#CBD5E1', 13, 'normal'),
     ('256 semillas, malla 12×12, barras de error.', '#CBD5E1', 13, 'normal')])

texto_frame('f_hallazgo', 'Hallazgo clave',
    [('El invariante es robusto ante error coherente:', '#FFFFFF', 14, 'bold'),
     ('• Error coherente (control): 100 % de acierto incluso a 1−F = 0,42', '#22C55E', 13, 'normal'),
     ('• Error en posiciones fijas: degradación suave (93 % → 28 %)', '#F59E0B', 13, 'normal'),
     ('• Ruido incoherente: destruido de inmediato (0 %)', '#F43F5E', 13, 'normal'),
     ('No hay umbral único: depende de la estructura del error.', '#CBD5E1', 13, 'normal')])

texto_frame('f_amd', 'AMD en el producto',
    [('GPU AMD Radeon Pro W7900D (ROCm 7.15, 48 GiB).', '#FFFFFF', 14, 'bold'),
     ('Justificada por memoria: 256 semillas × n=14 × malla 12×12', '#CBD5E1', 13, 'normal'),
     ('requieren ~45 GiB, que superan los 6 GiB de CPU.', '#CBD5E1', 13, 'normal'),
     ('Speedup vs CPU no medido; se declara ~1×.', '#F59E0B', 13, 'normal')])

texto_frame('f_demo', 'Demo y repositorio',
    [('streamlit run demo/app.py', '#FFFFFF', 14, 'bold'),
     ('Barra de ruido, modelo de ruido, semillas, curva de acierto', '#CBD5E1', 13, 'normal'),
     ('y plan del agente de mitigación.', '#CBD5E1', 13, 'normal'),
     ('Repo público: github.com/jalp17/chern-calibration-copilot', '#CBD5E1', 13, 'normal'),
     ('Licencia MIT.', '#CBD5E1', 13, 'normal')])

texto_frame('f_refs', 'Referencias',
    [('Lado et al., Phys. Rev. Research 6, 043288 (2024), arXiv:2404.06048', '#CBD5E1', 12, 'normal'),
     ('Fukui, Hatsugai, Suzuki, JPSJ 74, 1674 (2005)', '#CBD5E1', 12, 'normal'),
     ('Haldane, PRL 61, 2015 (1988)', '#CBD5E1', 12, 'normal'),
     ('Modelo de lenguaje: Gemma vía AMD Token Factory', '#CBD5E1', 12, 'normal')])

print('frames de texto generados')
