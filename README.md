# Chern Calibration Copilot

```yaml
titulo: "Chern Calibration Copilot"
titulo_caracteres: 41
descripcion_corta: "Diagnostico de calibracion cuantica: un agente que detecta como el ruido degrada el numero de Chern y recomienda correcciones."
licencia: MIT
hackathon: "AMD Developer Hackathon ACT III"
pista: "1 - Intelligent Industry"
estado: "esqueleto - no ejecutable aun"
```

## Que es

Un copiloto de calibracion para dispositivos cuanticos simulados. Diagnostica como el ruido de preparacion degrada un invariante topologico (el numero de Chern del modelo de Haldane) y emite un plan de mitigacion con fuentes, usando un agente que recupera especificaciones externas del dispositivo.

**Pista 1 - Intelligent Industry:** la anomalia es que el Chern deja de ser entero; el manual es la spec del dispositivo; la recomendacion es que calibrar.

## Como ejecutar

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Verificar el simulador (debe dar OK)
python simulador/test_qsim.py

# 3. Lanzar la demo
make demo
```

## Arquitectura (resumen)

- `simulador/` — motor statevector en PyTorch (qsim_torch, haldane, ruido, barrido_v2)
- `agente/` — agente LLM que lee specs y emite plan de mitigacion
- `demo/` — aplicacion Streamlit interactiva
- `docs/` — presentacion (slides PDF, video, notas)

Ejecuta en GPU AMD Radeon Pro W7900D (ROCm) o CPU.

## Licencia

MIT — ver `LICENSE`.

## Atribuciones

- AMD — hardware y plataforma (W7900D, ROCm)
- lablab.ai — organizador del hackathon
- Lado et al., arXiv:2404.06048 — formulacion del clasificador de Chern
- Fukui, Hatsugai & Suzuki, JPSJ 74, 1674 (2005) — formula FHS del Chern
- Haldane, PRL 61, 2015 (1988) — modelo de Haldane
- MIT — licencia de este repositorio

## Que NO afirmamos

- NO simulamos el efecto Hall cuantico fisico. Calculamos el invariante topologico.
- Con n=1 el Chern es el mismo; la GPU se justifica por memoria (9 GiB > 6 GiB RAM), no por velocidad fisica.
- El agente NO calcula el Chern; interpreta resultados y recomienda calibracion.
