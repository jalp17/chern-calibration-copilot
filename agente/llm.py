"""Cliente del modelo de lenguaje.

Usa un endpoint compatible con OpenAI (Token Factory de AMD). Si no hay clave
o el endpoint no responde, el agente degrada a un plan **local** basado en el
corpus, para que la demo siempre funcione.

Variables de entorno (ver `.env.example`):
    OPENAI_API_KEY   clave con formato `rc-` + 48 hex
    OPENAI_BASE_URL  p. ej. https://developer.amd.com.cn/radeon/api/v1
    LLM_MODEL        nombre del modelo (NO hardcodear: el catálogo cambia)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Respuesta:
    texto: str
    modelo: str
    via: str          # 'llm' | 'local' | 'error'
    error: str | None = None


def _post(base_url: str, api_key: str, payload: dict, timeout: int = 90) -> dict:
    """Llamada HTTP minima, sin dependencias mas alla de la stdlib."""
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _catalogo(base_url: str, api_key: str, timeout: int = 20) -> list[str]:
    """Devuelve los modelos disponibles. El catalogo cambia sin aviso."""
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return [m.get("id", "") for m in data.get("data", [])]


def preguntar(system: str, user: str, timeout: int = 90) -> Respuesta:
    """Pregunta al LLM. Si falla, devuelve via='error' con el motivo."""
    base = os.environ.get("OPENAI_BASE_URL", "")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not base or not key:
        return Respuesta("", "", "local",
                         "OPENAI_API_KEY / OPENAI_BASE_URL sin configurar")

    modelo = os.environ.get("LLM_MODEL", "")
    if not modelo:
        try:
            disponibles = _catalogo(base, key)
            modelo = disponibles[0] if disponibles else ""
        except Exception as exc:                       # noqa: BLE001
            return Respuesta("", "", "error", f"catalogo: {exc}")
    if not modelo:
        return Respuesta("", "", "error", "catalogo vacio")

    try:
        data = _post(base, key, {
            "model": modelo,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.2,
        }, timeout=timeout)
        texto = data["choices"][0]["message"]["content"]
        return Respuesta(texto, modelo, "llm")
    except (urllib.error.URLError, KeyError, TimeoutError, OSError) as exc:
        return Respuesta("", modelo, "error", str(exc))
