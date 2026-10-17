"""Cliente del modelo de lenguaje.

Carga .env si existe y soporta varios proveedores de LLM sin dependencias
externas (solo stdlib). Si no hay proveedor configurado o falla, el agente
degrada a un plan **local** basado en el corpus, para que la demo siempre
funcione.

Variables de entorno (ver `.env.example`):
    LLM_PROVIDER   openai | anthropic | ollama | auto  (por defecto: auto)
    LLM_MODEL      nombre del modelo (si se deja vacio, se consulta el catalogo)
    OPENAI_API_KEY / OPENAI_BASE_URL   proveedor OpenAI-compatible (Token Factory)
    ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL   proveedor Anthropic
    OLLAMA_BASE_URL / OLLAMA_MODEL   proveedor Ollama local
    LLM_TIMEOUT    segundos de timeout por peticion (por defecto: 60)
    LLM_RETRIES    numero de reintentos (por defecto: 2)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Respuesta:
    texto: str
    modelo: str
    via: str          # 'llm' | 'local' | 'error'
    error: str | None = None


# --------------------------------------------------------------------------- #
# Carga de .env (stdlib, sin dependencias nuevas)
# --------------------------------------------------------------------------- #
def _cargar_env(ruta: str | None = None) -> None:
    """Lee .env y asigna las variables que no esten ya definidas en el entorno."""
    if ruta is None:
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", ".env")
    ruta = os.path.abspath(ruta)
    if not os.path.isfile(ruta):
        return
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            if "=" not in linea:
                continue
            clave, valor = linea.split("=", 1)
            clave = clave.strip()
            valor = valor.strip().strip('"').strip("'")
            if clave and clave not in os.environ:
                os.environ[clave] = valor


_cargar_env()


@dataclass
class LLMConfig:
    proveedor: str
    base_url: str
    api_key: str
    modelo: str
    timeout: int = 60
    retries: int = 2


def _config_por_proveedor(proveedor: str) -> LLMConfig | None:
    """Construye la configuracion para un proveedor si tiene credenciales."""
    timeout = int(os.environ.get("LLM_TIMEOUT", "60"))
    retries = int(os.environ.get("LLM_RETRIES", "2"))
    if proveedor == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        base = os.environ.get("OPENAI_BASE_URL", "")
        modelo = os.environ.get("LLM_MODEL", "")
        if key and base:
            return LLMConfig("openai", base, key, modelo, timeout, retries)
    elif proveedor == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        base = os.environ.get("ANTHROPIC_BASE_URL", "")
        modelo = os.environ.get("LLM_MODEL", "")
        if key and base:
            return LLMConfig("anthropic", base, key, modelo, timeout, retries)
    elif proveedor == "ollama":
        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        modelo = os.environ.get("OLLAMA_MODEL", "")
        if modelo:
            return LLMConfig("ollama", base, "", modelo, timeout, retries)
    return None


def _config_auto() -> LLMConfig | None:
    for prov in ("openai", "anthropic", "ollama"):
        cfg = _config_por_proveedor(prov)
        if cfg is not None:
            return cfg
    return None


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class LLMBackend:
    """Interfaz: cualquier proveedor expone listar_modelos() y completar()."""

    def listar_modelos(self) -> list[str]:
        raise NotImplementedError

    def completar(self, system: str, user: str) -> tuple[str, str]:
        """Devuelve (texto, nombre_modelo)."""
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg

    def listar_modelos(self) -> list[str]:
        url = self.cfg.base_url.rstrip("/") + "/models"
        req = urllib.request.Request(url,
                                     headers={"Authorization": f"Bearer {self.cfg.api_key}"})
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
            data = json.loads(r.read().decode())
        return [m.get("id", "") for m in data.get("data", [])]

    def completar(self, system: str, user: str) -> tuple[str, str]:
        modelo = self.cfg.modelo or self._primero_de_catalogo()
        if not modelo:
            raise RuntimeError("modelo vacio")
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": modelo,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.cfg.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
            data = json.loads(r.read().decode())
        return data["choices"][0]["message"]["content"], modelo

    def _primero_de_catalogo(self) -> str:
        try:
            modelos = self.listar_modelos()
        except Exception:
            return ""
        return modelos[0] if modelos else ""


class AnthropicBackend(LLMBackend):
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg

    def listar_modelos(self) -> list[str]:
        # Anthropic no expone catalogo publico; usamos el modelo configurado.
        return [self.cfg.modelo] if self.cfg.modelo else []

    def completar(self, system: str, user: str) -> tuple[str, str]:
        modelo = self.cfg.modelo or "claude-3-haiku-20240307"
        url = self.cfg.base_url.rstrip("/") + "/v1/messages"
        payload = {
            "model": modelo,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "x-api-key": self.cfg.api_key,
                     "anthropic-version": "2023-06-01"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
            data = json.loads(r.read().decode())
        # Anthropic devuelve lista de bloques de contenido.
        bloques = data.get("content", [])
        texto = "".join(b.get("text", "") for b in bloques)
        return texto, modelo


class OllamaBackend(LLMBackend):
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg

    def listar_modelos(self) -> list[str]:
        try:
            url = self.cfg.base_url.rstrip("/") + "/api/tags"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
                data = json.loads(r.read().decode())
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []

    def completar(self, system: str, user: str) -> tuple[str, str]:
        modelo = self.cfg.modelo
        url = self.cfg.base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": modelo,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
            data = json.loads(r.read().decode())
        mensaje = data.get("message", {})
        return mensaje.get("content", ""), modelo


def _cargar_backend() -> LLMBackend | None:
    proveedor = os.environ.get("LLM_PROVIDER", "auto").lower()
    cfg: LLMConfig | None
    if proveedor == "auto":
        cfg = _config_auto()
    else:
        cfg = _config_por_proveedor(proveedor)
    if cfg is None:
        return None
    if cfg.proveedor == "openai":
        return OpenAIBackend(cfg)
    if cfg.proveedor == "anthropic":
        return AnthropicBackend(cfg)
    if cfg.proveedor == "ollama":
        return OllamaBackend(cfg)
    return None


def _obtener_backend() -> tuple[LLMBackend | None, str]:
    """Resuelve el backend leyendo el entorno en cada llamada.

    No cachea el fallo: si cambian las variables de entorno (p.ej. tests que
    ponen credenciales tras una llamada previa sin clave), se vuelve a intentar.
    """
    proveedor = os.environ.get("LLM_PROVIDER", "auto").lower()
    cfg: LLMConfig | None
    if proveedor == "auto":
        cfg = _config_auto()
    else:
        cfg = _config_por_proveedor(proveedor)
    if cfg is None:
        return None, "no hay proveedor configurado"
    try:
        if cfg.proveedor == "openai":
            backend: LLMBackend = OpenAIBackend(cfg)
        elif cfg.proveedor == "anthropic":
            backend = AnthropicBackend(cfg)
        elif cfg.proveedor == "ollama":
            backend = OllamaBackend(cfg)
        else:
            return None, f"proveedor desconocido: {cfg.proveedor}"
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    return backend, ""


# --------------------------------------------------------------------------- #
# Punto de entrada
# --------------------------------------------------------------------------- #
def preguntar(system: str, user: str, timeout: int = 90) -> Respuesta:
    """Pregunta al LLM. Si falla, devuelve via='local' con motivo.

    La firma se mantiene para que `agente.py` no necesite cambios.
    """
    backend, error = _obtener_backend()
    if backend is None:
        return Respuesta("", "", "local",
                         f"sin LLM ({error or 'no hay proveedor configurado'}). "
                         "Plan local determinista a partir del corpus.")

    ultimo_error: str | None = error
    for intento in range(1, int(os.environ.get("LLM_RETRIES", "2")) + 1):
        try:
            texto, modelo = backend.completar(system, user)
            return Respuesta(texto, modelo, "llm")
        except (urllib.error.URLError, KeyError, TimeoutError,
                OSError, json.JSONDecodeError) as exc:
            ultimo_error = str(exc)
            if intento < int(os.environ.get("LLM_RETRIES", "2")):
                time.sleep(0.5 * intento)
    return Respuesta("", "", "error",
                     f"LLM fallo tras reintentos ({ultimo_error})")
