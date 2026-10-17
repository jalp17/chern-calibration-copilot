import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs('entrega/assets', exist_ok=True)

# ---------- Colores ----------
AMD_RED = '#ED1C24'
DARK = '#0B0F19'
ACCENT = '#7C3AED'

fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)
ax.set_xlim(0, 12.8); ax.set_ylim(0, 7.2)
ax.axis('off')

# Banda roja superior
ax.add_patch(mpatches.Rectangle((0, 6.6), 12.8, 0.6, facecolor=AMD_RED, edgecolor='none'))
ax.text(0.4, 6.9, 'AMD Developer Hackathon ACT III', color='white', fontsize=13, fontweight='bold', va='center')
ax.text(12.4, 6.9, 'Pista 1 · Intelligent Industry', color='white', fontsize=11, ha='right', va='center')

# Título
ax.text(0.4, 5.1, 'Chern Calibration Copilot', color='white', fontsize=40, fontweight='bold', va='center')
ax.text(0.4, 4.4, 'Diagnóstico de calibración cuántica: cómo el ruido degrada el número de Chern',
        color='#CBD5E1', fontsize=16, va='center')

# Línea divisoria
ax.add_patch(mpatches.FancyArrowPatch((0.4, 3.9), (12.4, 3.9), arrowstyle='-', color=ACCENT, lw=2, mutation_scale=0))

# Subtítulo / qué es
ax.text(0.4, 3.4, 'Cuantifica cómo el ruido de preparación degrada un invariante topológico',
        color='white', fontsize=15, fontweight='bold', va='center')
ax.text(0.4, 2.9, '(número de Chern del modelo de Haldane) y propone mitigación con fuentes.',
        color='#CBD5E1', fontsize=13, va='center')

# Resultado clave (cuadro simplificado)
rows = [
    ('final — control coherente', '0.42', '100 %'),
    ('shared — Pauli fijo por puerta', '0.42', '28 %'),
    ('indep — ruido por (puerta, k)', '0.42', '0 %'),
]
y0 = 2.1
for i, (modelo, inf, acierto) in enumerate(rows):
    y = y0 - i*0.42
    ax.add_patch(mpatches.Rectangle((0.4, y-0.16), 4.6, 0.34, facecolor='#1E293B', edgecolor=ACCENT, lw=1))
    ax.text(0.55, y, modelo, color='white', fontsize=11, va='center')
    ax.text(5.1, y, f'1−F = {inf}', color='#CBD5E1', fontsize=11, va='center', ha='right')
    ax.text(6.1, y, f'acierto: {acierto}', color='#22C55E' if acierto=='100 %' else '#F43F5E',
            fontsize=11, fontweight='bold', va='center', ha='right')

ax.text(0.4, 0.75, 'El invariante es robusto ante error coherente, se degrada suavemente ante error',
        color='#CBD5E1', fontsize=12, va='center')
ax.text(0.4, 0.45, 'en posiciones fijas de puerta y se destruye ante ruido incoherente.',
        color='#CBD5E1', fontsize=12, va='center')

ax.text(0.4, 0.12, 'Repo: github.com/jalp17/chern-calibration-copilot  ·  Licencia MIT',
        color='#94A3B8', fontsize=10, va='center')

plt.tight_layout()
plt.savefig('assets/portada.png', dpi=150, facecolor=DARK)
plt.close()
print('portada guardada')

# ---------- Frame de resultados (para el vídeo) ----------
fig2, ax2 = plt.subplots(figsize=(12.8, 7.2), dpi=150)
fig2.patch.set_facecolor(DARK); ax2.set_facecolor(DARK)
ax2.set_xlim(0, 12.8); ax2.set_ylim(0, 7.2); ax2.axis('off')
ax2.text(0.4, 6.8, 'Resultado principal: tasa de error topológico vs infidelidad',
         color='white', fontsize=17, fontweight='bold', va='center')
import numpy as np
ps = np.array([0.0, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.30])
shared_acc = np.array([100, 93.4, 85.5, 78.1, 67.2, 60.2, 41.0, 28.1, 15.0])
final_acc = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100])
indep_acc = np.array([100, 19.9, 8.6, 2.3, 1.2, 0.8, 1.2, 0.8, 0.6])
ax2.plot(ps, shared_acc, 'o-', color='#38BDF8', lw=2.5, markersize=7, label='shared (físico)')
ax2.plot(ps, final_acc, 's-', color='#22C55E', lw=2.5, markersize=7, label='final (control coherente)')
ax2.plot(ps, indep_acc, '^-', color='#F43F5E', lw=2.5, markersize=7, label='indep (incoherente)')
ax2.set_xlabel('Infidelidad de preparación (1−F)', color='white', fontsize=13)
ax2.set_ylabel('% acierto del Chern', color='white', fontsize=13)
ax2.tick_params(colors='white')
ax2.spines['bottom'].set_color('white'); ax2.spines['left'].set_color('white')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.legend(facecolor='#1E293B', edgecolor='none', labelcolor='white', fontsize=11)
ax2.grid(color='#334155', alpha=0.4)
ax2.set_facecolor('#0F172A')
plt.tight_layout()
plt.savefig('assets/frame_resultados.png', dpi=150, facecolor=DARK)
plt.close()
print('frame resultados guardado')
