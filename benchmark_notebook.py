"""
benchmark_notebook.py - Modo de benchmark para notebook (AMD W7900D y cualquier GPU/CPU).

Uso en notebook:
    %run benchmark_notebook.py
    # o
    exec(open('benchmark_notebook.py').read())

Flujo:
  1. Detecta hardware (GPU, arquitectura GFX, versión ROCm).
  2. Imprime comando de instalación de pennylane-lightning-gpu bajo demanda
     (solo si el usuario quiere probarlo; el simulador PyTorch funciona sin él).
  3. Ejecuta barrido de benchmarks (n × batch × dtype).
  4. Genera:
     - benchmark_results_YYYYMMDD_HHMMSS.json  (datos brutos)
     - BENCHMARK_SUMMARY_YYYYMMDD_HHMMSS.md    (resumen legible)

No compila nada de antemano. Todo es bajo demanda.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Añadir directorio simulador al path para imports
_HERE = Path(__file__).resolve().parent
if str(_HERE / "simulador") not in sys.path:
    sys.path.insert(0, str(_HERE / "simulador"))

from qsim_torch import Simulator, LAYERS
from hardware_wheels import detect_gpu_architecture, install_command, quick_summary

BYTES_PER_AMP_C128 = 16  # complex128
BYTES_PER_AMP_C64 = 8    # complex64


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _bench(fn, device: torch.device, warmup: int = 1, runs: int = 3) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    best = math.inf
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        best = min(best, time.perf_counter() - t0)
    return best


def _make_workload(sim: Simulator, batch: int, dtype: torch.dtype):
    n = sim.n_qubits
    x = torch.randn(batch, n, dtype=torch.float64, device=sim.device)
    w = torch.randn(LAYERS, n, dtype=torch.float64, device=sim.device,
                    requires_grad=True)

    def step():
        out = sim.variational_circuit(x, w)
        out.pow(2).mean().backward()
        w.grad = None

    return step


def _mem_statevector(n: int, dtype: torch.dtype) -> float:
    bpc = BYTES_PER_AMP_C64 if dtype == torch.complex64 else BYTES_PER_AMP_C128
    return (1 << n) * bpc / 1024**3


def _report_device(device: torch.device) -> dict:
    info = {"device": str(device), "type": device.type}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": p.name,
            "vram_gib": round(p.total_memory / 1024**3, 2),
            "rocm_hip": torch.version.hip,
            "compute_capability": f"{p.major}.{p.minor}",
            "multiprocessors": p.multi_processor_count,
        })
    else:
        info.update({
            "torch_version": torch.__version__,
            "num_threads": torch.get_num_threads(),
        })
    return info


def run_benchmark(
    device_str: str = None,
    ns: list = None,
    batches: list = None,
    dtypes: list = None,
    output_dir: str = ".",
) -> dict:
    """
    Ejecuta el barrido completo de benchmark y guarda resultados.

    Flujo:
      1. Detecta hardware (GPU, arquitectura, ROCm).
      2. Muestra comando de instalación bajo demanda para lightning.amdgpu.
      3. Ejecuta benchmarks (n × batch × dtype).
      4. Guarda JSON con datos brutos y Markdown con resumen.

    Parámetros:
      device_str: 'cuda' o 'cpu'. Default: auto-detectar.
      ns: lista de números de qubits. Default según dispositivo.
      batches: lista de tamaños de batch. Default según dispositivo.
      dtypes: lista de torch.dtype. Default [complex128, complex64].
      output_dir: directorio donde guardar los archivos de salida.

    Retorna:
      dict con todos los resultados y rutas de archivos generados.
    """
    # =====================================================================
    # PASO 1: Detección de hardware (runtime, no compila nada)
    # =====================================================================
    gpu = detect_gpu_architecture()
    hw_summary = quick_summary()

    print("=" * 70)
    print("DETECCIÓN DE HARDWARE (runtime)")
    print("=" * 70)
    print(f"GPU             : {gpu.name}")
    print(f"Arquitectura    : {gpu.architecture}")
    print(f"Versión ROCm    : {gpu.rocm_version}")
    print(f"Wheel precompilada: {'SÍ' if gpu.supports_precompiled else 'NO'}")
    print()
    print("Comando de instalación de pennylane-lightning-gpu (bajo demanda):")
    print(f"  {install_command(gpu)}")
    print()
    print("Nota: el simulador PyTorch puro (qsim_torch.py) funciona sin "
          "pennylane-lightning-gpu. Solo instálalo si quieres comparar.")
    print("=" * 70)

    # =====================================================================
    # PASO 2: Configuración del dispositivo y parámetros
    # =====================================================================
    device = torch.device(
        device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(0)

    if ns is None:
        ns = [16, 20, 24] if device.type == "cuda" else [10, 12, 14]
    if batches is None:
        batches = [1, 256, 1024] if device.type == "cuda" else [1, 8, 64]
    if dtypes is None:
        dtypes = [torch.complex128, torch.complex64]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device_info = _report_device(device)
    device_info["hardware_detection"] = hw_summary

    results = {
        "timestamp": timestamp,
        "device": device_info,
        "config": {
            "ns": ns,
            "batches": batches,
            "dtypes": [str(dt) for dt in dtypes],
            "layers": LAYERS,
        },
        "benchmarks": [],
        "scaling": [],
        "memory": [],
        "summary": {},
    }

    print("=" * 70)
    print(f"BENCHMARK NOTEBOOK — {timestamp}")
    print(f"Dispositivo: {device_info.get('gpu_name', device_info.get('torch_version', 'cpu'))}")
    print("=" * 70)

    # =====================================================================
    # PASO 3: Barrido n × batch × dtype
    # =====================================================================
    for dtype in dtypes:
        dtype_name = "complex64" if dtype == torch.complex64 else "complex128"
        print(f"\n--- dtype: {dtype_name} ---")
        for n in ns:
            sim = Simulator(n, device=str(device), dtype=dtype)
            for b in batches:
                est_gb = _mem_statevector(n, dtype) * b
                if est_gb > 6.0:
                    results["benchmarks"].append({
                        "n": n, "batch": b, "dtype": dtype_name,
                        "status": "skipped_memory",
                        "estimated_gb": round(est_gb, 3),
                    })
                    print(f"  n={n:2d} batch={b:5d} -> SALTADO (est. {est_gb:.2f} GiB > 6 GiB)")
                    continue
                try:
                    step = _make_workload(sim, b, dtype)
                    t = _bench(step, device)
                    layers = LAYERS * n
                    amps = b * (1 << n)
                    entry = {
                        "n": n, "batch": b, "dtype": dtype_name,
                        "status": "ok",
                        "time_sec": round(t, 6),
                        "layers_per_sec": round(layers / t, 2),
                        "amplitudes_per_sec": round(amps / t, 4),
                        "est_memory_gib": round(est_gb, 3),
                        "sec_per_layer": round(t / layers, 8),
                    }
                    results["benchmarks"].append(entry)
                    print(f"  n={n:2d} batch={b:5d} "
                          f"{layers/t:8.1f} capas/s  "
                          f"{amps/t:10.2e} amp/s  "
                          f"{est_gb:6.3f} GiB")
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    results["benchmarks"].append({
                        "n": n, "batch": b, "dtype": dtype_name,
                        "status": "oom",
                        "error": str(exc)[:100],
                    })
                    print(f"  n={n:2d} batch={b:5d} -> OOM/ERROR: {str(exc)[:60]}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    # =====================================================================
    # PASO 4: Escalado de batch (ganancia real de GPU)
    # =====================================================================
    print("\n--- Escalado de batch ---")
    for dtype in dtypes:
        dtype_name = "complex64" if dtype == torch.complex64 else "complex128"
        n_test = ns[len(ns) // 2] if ns else 16
        sim = Simulator(n_test, device=str(device), dtype=dtype)
        base = None
        for b in batches:
            est_gb = _mem_statevector(n_test, dtype) * b
            if est_gb > 4.0:
                results["scaling"].append({
                    "n": n_test, "batch": b, "dtype": dtype_name,
                    "status": "skipped_memory",
                })
                continue
            try:
                step = _make_workload(sim, b, dtype)
                t = _bench(step, device, runs=2)
                per_sample = b / t
                entry = {
                    "n": n_test, "batch": b, "dtype": dtype_name,
                    "time_sec": round(t, 6),
                    "samples_per_sec": round(per_sample, 2),
                    "speedup_vs_batch1": None if base is None else round(per_sample / base, 2),
                }
                if base is None:
                    base = per_sample
                results["scaling"].append(entry)
                print(f"  {dtype_name:10s} batch={b:5d} "
                      f"{t:8.4f}s  {per_sample:10.1f} muestras/s  "
                      f"{'' if base is None else per_sample/base:7.2f}x")
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # =====================================================================
    # PASO 5: Reporte de memoria
    # =====================================================================
    print("\n--- Techo de memoria ---")
    total_vram = 48.0
    if device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    reserve = 0.65

    for dtype in dtypes:
        dtype_name = "complex64" if dtype == torch.complex64 else "complex128"
        bpc = BYTES_PER_AMP_C64 if dtype == torch.complex64 else BYTES_PER_AMP_C128
        for b in (1, 64, 256, 1024):
            n_max = 1
            while (1 << (n_max + 1)) * bpc * b / 1024**3 < total_vram * reserve:
                n_max += 1
            entry = {
                "dtype": dtype_name, "batch": b,
                "vram_gib": round(total_vram, 2),
                "reserve_fraction": reserve,
                "max_qubits": n_max,
                "memory_at_max_gib": round((1 << n_max) * bpc * b / 1024**3, 3),
            }
            results["memory"].append(entry)
            print(f"  {dtype_name:10s} batch={b:5d} -> n_max={n_max}, "
                  f"mem={(1 << n_max) * bpc * b / 1024**3:.2f} GiB")

    pico_vram = 0.0
    if device.type == "cuda":
        pico_vram = torch.cuda.max_memory_reserved() / 1024**3
        print(f"\nPico VRAM reservado: {pico_vram:.2f} GiB")

    # =====================================================================
    # PASO 6: Resumen ejecutivo
    # =====================================================================
    ok_results = [r for r in results["benchmarks"] if r["status"] == "ok"]
    if ok_results:
        best = max(ok_results, key=lambda r: r["layers_per_sec"])
        worst = min(ok_results, key=lambda r: r["layers_per_sec"])
    else:
        best = worst = None

    summary = {
        "total_benchmarks": len(results["benchmarks"]),
        "passed": len(ok_results),
        "failed_or_skipped": len(results["benchmarks"]) - len(ok_results),
        "best_configuration": best,
        "worst_configuration": worst,
        "device": device_info,
        "timestamp": timestamp,
        "gpu_load_bearing": None,
    }

    # Determinar si la GPU es load-bearing comparando CPU vs GPU si hay ambos
    # (requiere ejecutar en ambos dispositivos; aquí solo se anota si hay GPU)
    if device.type == "cuda":
        summary["gpu_load_bearing"] = {
            "note": "Para determinar si la GPU es load-bearing, ejecute el "
                    "benchmark también en CPU y compare layers_per_sec.",
            "gpu_name": device_info.get("gpu_name"),
            "vram_gib": device_info.get("vram_gib"),
        }
    else:
        summary["gpu_load_bearing"] = {
            "note": "Ejecutando en CPU. No hay GPU para comparar.",
        }

    results["summary"] = summary

    # =====================================================================
    # PASO 7: Guardar archivo JSON
    # =====================================================================
    json_name = f"benchmark_results_{timestamp}.json"
    json_path = out_path / json_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDatos brutos guardados en: {json_path}")

    # =====================================================================
    # PASO 8: Generar resumen Markdown
    # =====================================================================
    md_name = f"BENCHMARK_SUMMARY_{timestamp}.md"
    md_path = out_path / md_name
    lines = []
    lines.append(f"# Resumen de Benchmark — {timestamp}")
    lines.append("")
    lines.append("## 1. Detección de hardware")
    lines.append(f"- GPU: {gpu.name}")
    lines.append(f"- Arquitectura: {gpu.architecture}")
    lines.append(f"- ROCm/HIP: {gpu.rocm_version}")
    lines.append(f"- Wheel precompilada disponible: {'SÍ' if gpu.supports_precompiled else 'NO'}")
    lines.append("")
    lines.append("### Comando de instalación (bajo demanda)")
    lines.append(f"```bash")
    lines.append(f"{install_command(gpu)}")
    lines.append(f"```")
    lines.append("")
    lines.append("## 2. Dispositivo de benchmark")
    lines.append(f"- Nombre: {device_info.get('gpu_name', 'CPU')}")
    lines.append(f"- Tipo: {device_info['type']}")
    if "vram_gib" in device_info:
        lines.append(f"- VRAM: {device_info['vram_gib']} GiB")
    if "rocm_hip" in device_info:
        lines.append(f"- ROCm/HIP: {device_info['rocm_hip']}")
    lines.append("")
    lines.append("## 3. Configuración")
    lines.append(f"- Qubits (n): {results['config']['ns']}")
    lines.append(f"- Batches: {results['config']['batches']}")
    lines.append(f"- Dtypes: {results['config']['dtypes']}")
    lines.append(f"- Capas por circuito: {LAYERS}")
    lines.append("")
    lines.append("## 4. Resultados globales")
    lines.append(f"- Total de pruebas: {summary['total_benchmarks']}")
    lines.append(f"- Pasadas: {summary['passed']}")
    lines.append(f"- Falladas/saltadas: {summary['failed_or_skipped']}")
    lines.append("")
    if summary["best_configuration"]:
        bc = summary["best_configuration"]
        lines.append("## 5. Mejor configuración")
        lines.append(f"- n={bc['n']}, batch={bc['batch']}, dtype={bc['dtype']}")
        lines.append(f"- {bc['layers_per_sec']} capas/s")
        lines.append(f"- {bc['amplitudes_per_sec']} amplitudes/s")
    lines.append("")
    lines.append("## 6. ¿Es la GPU load-bearing?")
    glb = summary["gpu_load_bearing"]
    lines.append(f"- {glb['note']}")
    if "gpu_name" in glb:
        lines.append(f"- GPU: {glb['gpu_name']}")
    if "vram_gib" in glb:
        lines.append(f"- VRAM: {glb['vram_gib']} GiB")
    lines.append("")
    lines.append("## 7. Techo de memoria (complex128, 65% reservado)")
    for m in results["memory"]:
        if m["dtype"] == "complex128":
            lines.append(f"- batch {m['batch']:5d}: n_max={m['max_qubits']} ({m['memory_at_max_gib']} GiB)")
    lines.append("")
    lines.append("## 8. Archivo de datos brutos")
    lines.append(f"- `{json_name}`")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Resumen guardado en: {md_path}")

    results["output_files"] = {
        "json": str(json_path),
        "markdown": str(md_path),
    }

    print("=" * 70)
    print("BENCHMARK COMPLETADO")
    print("=" * 70)

    return results


# Ejecutar automáticamente si se lanza como script o en notebook
if __name__ == "__main__":
    run_benchmark()
