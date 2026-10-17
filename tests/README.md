# Sistema de testing — Chern Calibration Copilot

Ejecuta todas las categorías de tests y reporta resultados reales (pasados /
fallados, mediciones de optimización, hallazgos de seguridad).

## Estructura

```
tests/
  conftest.py       # path + torch.set_num_threads(2)
  run_all.py        # runner propio: ejecuta todas o una categoria
  test_correctness.py   # A: fisica y codigo (12 tests)
  test_errores.py   # B: errores y casos limite (10 tests)
  test_optimizacion.py  # C: mediciones de rendimiento (5 tests)
  test_seguridad.py   # D: credenciales, ejecucion arbitraria, validacion (8 tests)
```

## Ejecutar

```bash
# Desde la raiz del repo (entrega/)
python3 tests/run_all.py              # todo
python3 tests/run_all.py correctness  # solo una categoria
python3 tests/run_all.py -v         # salida detallada
```

No requiere red ni GPU. Todas las mediciones de optimización se imprimen para
el reporte; los tests fallan si empeora un umbral, no por lentitud.

## Categorias

| Categoria | Que cubre | Tests |
|---|---|---|
| correctness | puertas unitarias vs kron, CNOT, fidelidad, ry medio-angulo, FHS sobre malla, Chern Haldane, prepare p=0, barrido p=0, agente umbrales | 12 |
| errores | Simulator(0), batch=0, Nk=0/1, p=1.0, p<0/p>1, fhs estados nulos, fidelidad formas distintas, LLM sin clave/invalida/caida, agente JSON vacio, corpus vacio | 10 |
| optimizacion | speedup apply_rz_all, speedup measure_z_all, memoria lote, complex64 vs complex128, overhead RSS | 5 |
| seguridad | cero credenciales, .env no en git, sin artefactos grandes, sin ejecucion arbitraria, LLM texto no ejecuta, barrido valida argumentos, sin rutas del vault, Docker no-root | 8 |

## Resultado esperado

```
RESULTADO: 35 pasados, 0 fallados
```

Si algún test falla, revisa `docs/AUDITORIA.md` para el contexto y las
recomendaciones.
