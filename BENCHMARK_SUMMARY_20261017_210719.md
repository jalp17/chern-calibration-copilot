# Resumen de Benchmark — 20261017_210719

## 1. Detección de hardware
- GPU: CPU (sin GPU detectada)
- Arquitectura: n/a
- ROCm/HIP: desconocida
- Wheel precompilada disponible: NO

### Comando de instalación (bajo demanda)
```bash
pip install pennylane-lightning-gpu # Si falla: --no-binary pennylane-lightning-gpu --config-settings=Kokkos_ARCH_AMD_N/A=ON
```

## 2. Dispositivo de benchmark
- Nombre: CPU
- Tipo: cpu

## 3. Configuración
- Qubits (n): [10, 12, 14]
- Batches: [1, 8, 64]
- Dtypes: ['torch.complex128', 'torch.complex64']
- Capas por circuito: 4

## 4. Resultados globales
- Total de pruebas: 18
- Pasadas: 18
- Falladas/saltadas: 0

## 5. Mejor configuración
- n=10, batch=1, dtype=complex128
- 989.46 capas/s
- 25330.1033 amplitudes/s

## 6. ¿Es la GPU load-bearing?
- Ejecutando en CPU. No hay GPU para comparar.

## 7. Techo de memoria (complex128, 65% reservado)
- batch     1: n_max=30 (16.0 GiB)
- batch    64: n_max=24 (16.0 GiB)
- batch   256: n_max=22 (16.0 GiB)
- batch  1024: n_max=20 (16.0 GiB)

## 8. Archivo de datos brutos
- `benchmark_results_20261017_210719.json`