"""
Detección de hardware AMD/ROCm y gestión de wheels precompiladas
para pennylane-lightning-amdgpu.

Propósito: portabilidad. Este módulo NO compila nada de antemano. Solo detecta
el hardware en runtime y devuelve el comando de instalación adecuado para que
el usuario (o el notebook) ejecute la compilación/instalación bajo demanda en
su propia máquina.

Las wheels precompiladas de lightning.amdgpu son SOLO para MI300 (gfx942).
En gfx1100 (W7900D, Radeon RX 7000 series) no hay wheel precompilada;
hay que compilar desde fuente con -DKokkos_ARCH_AMD_GFX1100=ON.

Uso en notebook:
    from simulador.hardware_wheels import detect_gpu, install_command, print_report
    gpu = detect_gpu()
    print(install_command(gpu))   # comando para copiar y pegar en terminal
    print_report()                # informe legible
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class GpuInfo:
    """Información de la GPU detectada."""
    name: str
    architecture: str       # ej. "gfx1100", "gfx942", "gfx900"
    rocm_version: str
    is_mi300: bool          # gfx942 -> wheel precompilada disponible
    is_gfx1100: bool        # gfx1100 -> requiere compilación desde fuente
    supports_precompiled: bool


def _run(cmd: list[str], timeout: int = 30) -> Optional[str]:
    """Ejecuta un comando y devuelve stdout, o None si falla."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def detect_rocm_version() -> str:
    """Intenta detectar la versión de ROCm instalada."""
    out = _run(["rocminfo", "--version"])
    if out:
        for line in out.splitlines():
            if "ROCm" in line or "version" in line.lower():
                return line.strip()
    try:
        import torch
        if torch.version.hip:
            return torch.version.hip
    except Exception:
        pass
    return "desconocida"


def detect_gpu_architecture() -> GpuInfo:
    """
    Detecta la GPU AMD presente y su arquitectura GFX.

    Prioridad de detección:
      1. torch.cuda.get_device_name(0) + torch.version.hip (si torch está instalado)
      2. rocm-smi
      3. rocminfo
    """
    import torch

    if not torch.cuda.is_available():
        return GpuInfo(
            name="CPU (sin GPU detectada)",
            architecture="n/a",
            rocm_version=detect_rocm_version(),
            is_mi300=False,
            is_gfx1100=False,
            supports_precompiled=False,
        )

    name = torch.cuda.get_device_name(0)
    hip_version = torch.version.hip or "desconocida"

    arch_map = {
        "MI300": "gfx942",
        "MI200": "gfx908",
        "MI100": "gfx906",
        "W7900": "gfx1100",
        "W7800": "gfx1100",
        "RX 7900": "gfx1100",
        "RX 7800": "gfx1100",
        "RX 7700": "gfx1100",
        "RX 7600": "gfx1100",
        "RX 6900": "gfx1030",
        "RX 6800": "gfx1030",
        "RX 6700": "gfx1030",
        "RX 6600": "gfx1036",
    }

    architecture = "desconocida"
    for key, arch in arch_map.items():
        if key in name:
            architecture = arch
            break

    is_mi300 = architecture == "gfx942"
    is_gfx1100 = architecture == "gfx1100"

    return GpuInfo(
        name=name,
        architecture=architecture,
        rocm_version=hip_version,
        is_mi300=is_mi300,
        is_gfx1100=is_gfx1100,
        supports_precompiled=is_mi300,
    )


def install_command(gpu: GpuInfo, version: str = "latest") -> str:
    """
    Devuelve el comando de pip install adecuado para la GPU detectada.

    - gfx942 (MI300): wheel precompilada desde PyPI (instalación rápida).
    - gfx1100 (W7900D y otras RDNA3): compilación desde fuente con
      Kokkos_ARCH_AMD_GFX1100=ON (lento, 30-60 min).
    - Otra arquitectura: intentar wheel, si falla, compilar desde fuente.

    Este comando está pensado para ejecutarse bajo demanda en la máquina
    del usuario, no de antemano.
    """
    if gpu.supports_precompiled:
        if version == "latest":
            return "pip install pennylane-lightning-gpu"
        return f"pip install pennylane-lightning-gpu=={version}"

    if gpu.is_gfx1100:
        return (
            "pip install pennylane-lightning-gpu --no-binary pennylane-lightning-gpu "
            "--config-settings=Kokkos_ARCH_AMD_GFX1100=ON"
        )

    return (
        f"pip install pennylane-lightning-gpu "
        f"# Si falla: --no-binary pennylane-lightning-gpu "
        f"--config-settings=Kokkos_ARCH_AMD_{gpu.architecture.upper()}=ON"
    )


def compile_from_source_cmd(gpu: GpuInfo, version: str = "latest") -> str:
    """
    Devuelve el comando completo para compilar lightning.amdgpu desde fuente,
    incluyendo las dependencias de Kokkos con el flag de arquitectura correcto.
    """
    arch_flag = gpu.architecture.upper()
    return (
        f"pip install pennylane-lightning-gpu --no-binary pennylane-lightning-gpu "
        f"--config-settings=Kokkos_ARCH_AMD_{arch_flag}=ON "
        f"--config-settings=build_ext=--parallel=$(nproc)"
    )


def print_report() -> None:
    """Imprime un informe legible del hardware detectado y cómo instalar."""
    gpu = detect_gpu_architecture()
    print("=" * 68)
    print("DETECCIÓN DE HARDWARE AMD / ROCm")
    print("=" * 68)
    print(f"GPU             : {gpu.name}")
    print(f"Arquitectura    : {gpu.architecture}")
    print(f"Versión ROCm    : {gpu.rocm_version}")
    print(f"Es MI300 (gfx942): {gpu.is_mi300}")
    print(f"Es gfx1100      : {gpu.is_gfx1100}")
    print(f"Wheel precompilada: {'SÍ' if gpu.supports_precompiled else 'NO'}")
    print()
    print("Comando de instalación (ejecutar bajo demanda):")
    print(f"  {install_command(gpu)}")
    print()
    if not gpu.supports_precompiled:
        print("Nota: compilar desde fuente con Kokkos puede tardar 30-60 min.")
        print("      No es viable en 4 días de hackathon. Usar simulador PyTorch puro.")
    print("=" * 68)


def quick_summary() -> dict:
    """Devuelve un dict resumido del hardware para incluir en reportes de benchmark."""
    gpu = detect_gpu_architecture()
    return {
        "gpu_name": gpu.name,
        "architecture": gpu.architecture,
        "rocm_version": gpu.rocm_version,
        "is_mi300": gpu.is_mi300,
        "is_gfx1100": gpu.is_gfx1100,
        "supports_precompiled_wheel": gpu.supports_precompiled,
        "install_command": install_command(gpu),
    }


if __name__ == "__main__":
    print_report()
