# Código de superficie (surface code)

Módulo `simulador/surface_code/` del proyecto **Chern Calibration Copilot**
(AMD Developer Hackathon ACT III). Implementa un código de superficie de
distancia `d` con qubits de datos sobre las aristas de un rejado cuadrado.

## Construcción

Para `d=3` el código es un `[[13, 1, 3]]`:

- `n = d² + (d−1)² = 13` qubits de datos (aristas).
- 8 estabilizadores X (estrellas, en vertientes).
- 4 estabilizadores Z (plaquetas, en caras).
- 1 qubit lógico, distancia 3.

El estado completo de 13 qubits ocupa `2¹³ = 8192` amplitudes (128 KiB en
`complex128`). El módulo usa una **representación dispersa de Pauli** (cada
estabilizador es un string de Pauli), así que el consumo real es de unos KB,
muy por debajo del límite de 100 MiB de la prueba local.

## Módulos

| Archivo | Contenido |
|---------|-----------|
| `codigo_superficie.py` | Construcción del código y estabilizadores dispersos (`build_surface_code`, `SurfaceCode`). |
| `sindromes.py` | Síndrome de error (eigenvalores +1 / −1 de los estabilizadores). |
| `ruido_codigo.py` | Ruido físico: X, Y, Z con probabilidad `p` por qubit. |
| `decodador.py` | Decodador exhaustivo peso ≤ 1 (para d=3) y detección de fallo lógico. |
| `tasa_error_logico.py` | `P_L(p)` por Monte Carlo. |
| `__init__.py` | Exporta todo. |

## Uso mínimo

```python
from simulador.surface_code import (
    build_surface_code, sindrome, es_nulo, decodificar, tasa_error_logico,
)

code = build_surface_code(3)          # [[13, 1, 3]]
assert es_nulo(sindrome((), code))    # sin error: síndrome todos +1

# error de bit-flip en un qubit de una plaqueta -> síndrome no nulo
s = sindrome(((0, 'X'),), code)
assert not es_nulo(s)

# el decodador corrige cualquier error de peso 1
corr, fallo = decodificar(((3, 'Z'),), code)
assert not fallo

# tasa de error logico
pl = tasa_error_logico(code, 0.01, n_tiros=2000, seed=1)
assert pl < 0.01                      # P_L(0.01) < 0.01
```

## Pruebas locales (d=3, CPU)

- Tiempo: ~160 ms (límite < 200 ms).
- Memoria: < 2 GiB (representación dispersa).
- `python3 tests/run_all.py surface_code`: 6/0.
- `python3 tests/run_all.py` (todo): 72/0 (66 previos + 6 nuevos).

## Reglas del proyecto cumplidas

1. No se modifica `simulador/qsim_torch.py` ni `simulador/test_qsim.py`.
2. No se modifica `simulador/ruido.py`.
3. Cero credenciales en el repositorio.
4. Nombres `snake_case`, sin acentos.
5. No hay llamadas de red desde los tests.
6. Los tests existentes siguen pasando sin regresiones.

## Listo para GPU (W7900D)

El módulo no depende de PennyLane ni de `lightning.amdgpu` (descartados por
incompatibilidad de arquitectura gfx1100). Usa únicamente PyTorch, así que
corre en CPU y en cualquier dispositivo PyTorch/ROCm, incluida la
**AMD Radeon Pro W7900D** (RDNA 3, 48 GiB VRAM) disponible en
`notebooks.amd.com`. Los 13 qubits del estado completo (128 KiB) caben
sobradamente en VRAM, por lo que el código está listo para ejecutarse en la
GPU de AMD cuando se disponga de un pod.
