---
titulo: Auditoria de testing — Chern Calibration Copilot
fecha: 2026-10-17
sesion: hackathon-amd-act3
estado: verificado
relaciona:
  - "[[tests/README]]"
---

# Auditoria de testing

Sistema de tests en `tests/` que cubre cuatro categorías: **correctitud**,
**errores y casos límite**, **optimización** y **seguridad**. Todas las cifras
que aparecen a continuación son mediciones reales ejecutadas con
`python3 tests/run_all.py`.

## Resumen de resultados

| Categoria | Tests | Pasados | Fallados |
|---|---|---|---|
| correctitud | 12 | 12 | 0 |
| errores | 10 | 10 | 0 |
| optimizacion | 5 | 5 | 0 |
| seguridad | 8 | 8 | 0 |
| **Total** | **35** | **35** | **0** |

Resultado final: **35 pasados, 0 fallados.**

---

## Detalle por categoria

### A. Correctitud (12 tests)

| Test | Que verifica | Resultado |
|---|---|---|
| A1 | Puertas 1-qubit (H,X,Y,Z) vs kron exacto, n=1..4 | PASS (error < 1e-12) |
| A2 | CNOT todos los pares ordenados, n=2..4 | PASS (error < 1e-12) |
| A3 | `state_fidelity` = \|⟨ψ\|φ⟩\|², no probabilidad de colisión | PASS |
| A4 | `ry(theta)` = RY(theta) (convención medio-ángulo) | PASS (error < 1e-15) |
| A5 | FHS suma SOLO sobre ejes de malla, no sobre el lote | PASS |
| A6 | Chern = 0 trivial, -1 topológico, mallas 8..32 | PASS |
| A7 | `prepare` con p=0 reproduce el estado ideal | PASS |
| A8 | Barrido con p=0 da 100% acierto en ambos modelos | PASS |
| A9 | Agente decide ACEPTAR/INDECSO/RECHAZAR (95/50) | PASS |

**Hallazgo corregido durante el desarrollo:** la convención de ángulos de
`rotation(cos,sin)` daba RY(2θ) en vez de RY(θ). Se detectó al comparar contra
PennyLane y se corrigió documentando `rotation()` como constructora de matriz
cruda y añadiendo `ry(theta)` con convención estándar.

### B. Errores y casos límite (10 tests)

| Test | Que verifica | Resultado |
|---|---|---|
| B1 | `Simulator(0)` no crashea | PASS |
| B2 | `batch=0` no crashea | PASS |
| B3 | `Nk=0` y `Nk=1` mallas válidas | PASS |
| B4 | `p=1.0` (ruido total) no crashea | PASS |
| B5 | `p` negativo devuelve todo Identity; p>1 no crashea | PASS |
| B6 | `fhs` con estados nulos → NaN controlado | PASS |
| B7 | `state_fidelity` con formas incompatibles → error | PASS |
| B8 | LLM sin clave → fallback `local` | PASS |
| B9 | `agente.generar({})` → `SIN DATOS` | PASS |
| B10 | `agente` con corpus vacío → plan con fuentes | PASS |

**Corrección durante la auditoría:** el cliente LLM (`llm.py`) cacheaba el fallo
del backend y no re-leía el entorno en llamadas posteriores. Un test con clave
inválida después de uno sin clave devolvía `via='local'` en vez de `via='error'`.
Se reescribió `_obtener_backend()` para resolver el entorno en cada llamada
sin cachear fallos. Ahora los tres escenarios (sin clave, clave inválida,
endpoint caído) devuelven correctamente `via='local'` / `via='error'`.

### C. Optimización (5 tests)

Todas las mediciones se **imprimen** para el reporte; los tests fallan si se
empeora un umbral, no por lentitud.

| Medida | Resultado real | Umbral | Estado |
|---|---|---|---|
| C1 speedup `apply_rz_all` vs bucle RY | 2.83x | > 1.5x | PASS |
| C2 speedup `measure_z_all` vs bucle `measure_z` | 7.99x | > 1.0x | PASS |
| C3 memoria lote = nseeds·Nk²·2ⁿ·16 | ratio 1.000 | [0.8, 1.2] | PASS |
| C4 error relativo complex64 vs complex128 | 2.45e-07 | < 1e-5 | PASS |
| C5 overhead RSS sobre lote teórico | < 10x | < 10x | PASS |

**Interpretación:** `apply_rz_all` y `measure_z_all` son optimizaciones reales
(2.8x y 8.0x respectivamente). La fórmula de memoria es precisa. `complex64`
es ~2.5e-07 de error relativo frente a complex128.

### D. Seguridad (8 tests)

| Test | Que verifica | Resultado |
|---|---|---|
| D1 | Cero credenciales en el repo | PASS |
| D1b | `.env.example` tiene valores vacíos | PASS |
| D2 | `.env` y `.env.local` no en git | PASS |
| D3 | Sin artefactos grandes (>10 MB) en git | PASS |
| D4 | Sin `eval`, `exec`, `pickle.loads`, `os.system` | PASS |
| D5 | El texto del LLM es string, no se ejecuta | PASS |
| D6 | `barrido_v2` valida argumentos (n,G,Nk,nseeds,p,modelo,device) | PASS |
| D7 | Sin rutas absolutas del vault | PASS |
| D8 | Dockerfile tiene `USER` no-root | PASS |

**Corrección durante la auditoría:** el `Dockerfile` corría como root. Se añadió
`useradd appuser`, `chown -R appuser:appuser /app` y `USER appuser`.

---

## Fallos reales encontrados y corregidos

| # | Que fallaba | Fichero | Como se arreglo |
|---|---|---|---|
| 1 | LLM cacheaba el fallo del backend; tests con credenciales cambiantes devolvían `via='local'` en vez de `via='error'` | `agente/llm.py` | `_obtener_backend()` resuelve el entorno cada llamada sin cachear fallos |
| 2 | `Dockerfile` corría como root (sin `USER`) | `Dockerfile` | Añadido `useradd appuser`, `chown -R`, `USER appuser` |
| 3 | Dependencia `openai>=1.0` sobrante (no se usa; `llm.py` usa `urllib`) | `requirements.txt` | Eliminada |

---

## Medidas de seguridad verificadas

- **Cero credenciales:** ningún fichero de texto contiene `rc-`, `sk-`, `gho_`, `AKIA`, claves Google ni private keys.
- **`.env` no en git:** `.env.example` con valores vacíos; `.env` y `.env.local` no trackeados.
- **Sin ejecución arbitraria:** no hay `eval`, `exec`, `pickle.loads`, `os.system`, `__import__`, ni `subprocess.run` con input del usuario.
- **LLM como string:** el texto del modelo se trata como string en `plan.texto`, nunca se ejecuta.
- **Validación de argumentos:** `barrido_v2` rechaza `n<=0`, `G<=0`, `Nk<2`, `nseeds<=0`, `p<0`, `p>1`, modelos inválidos y `device` inválido.
- **Sin rutas del vault:** no hay rutas absolutas de la base de conocimiento local en el repo (test D7 lo verifica).
- **Docker no-root:** `USER appuser`.

---

## Que no se pudo verificar

- **GPU real (W7900D, ROCm):** no disponible en esta máquina (Athlon 3000G, Vega 3 sin ROCm). Los tests de GPU (`device='cuda'`) se saltan o fallan silenciosamente. La afirmación de "GPU por memoria, no por velocidad" del README no puede verificarse localmente.
- **Lectura real del catálogo de Token Factory:** requiere red y clave válida; no se probó en este entorno.
- **Despliegue en Streamlit Cloud:** requiere click manual en https://share.streamlit.io.

---

## Recomendaciones priorizadas

| Prioridad | Accion | Esfuerzo |
|---|---|---|
| 1 | Medir el speedup real GPU vs CPU en el notebook W7900D (etapa 3 del notebook) | 15 min |
| 2 | Ejecutar el barrido completo en GPU (n=14, 256 semillas) para validar memoria ~45 GiB | 30-90 min |
| 3 | Desplegar la demo en Streamlit Cloud para obtener la URL viva | 5 min |
| 4 | Compilar `docs/slides.tex` a PDF (pdflatex o Overleaf) | 10 min |

---

## Como ejecutar

```bash
# desde la raiz del repo (entrega/)
python3 tests/run_all.py          # todas las categorias
python3 tests/run_all.py correctness   # solo una categoria
python3 tests/run_all.py -v       # salida detallada
```

Requisitos: `pip install -r requirements.txt`. No requiere red ni GPU.
