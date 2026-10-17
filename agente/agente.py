"""Agente de calibración.

NO calcula el Chern ni interpreta la curva: recupera el hecho externo que no
esta en el JSON del barrido (presupuesto de error, coherencia del equipo,
criterio de aceptacion) y emite un plan de mitigacion **con fuentes**.

Flujo:
    1. lee el JSON del barrido  -> infidelidad y tasa de acierto
    2. lee el corpus local      -> presupuesto y opciones de mitigacion
    3. decide segun el criterio de aceptacion (>=95 % aceptar, 50-95 % indeciso)
    4. si hay LLM, sintetiza el plan; si no, usa la plantilla local

El agente **siempre** devuelve algo util, con o sin red.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agente import llm

CORPUS = Path(__file__).resolve().parent / "corpus"

CRITERIO_ACEPTAR = 0.95
CRITERIO_RECHAZAR = 0.50

SYSTEM_PROMPT = """Eres un copiloto de calibracion cuantica.

Tu unica tarea: a partir de (a) el resultado de un barrido que mide como el
ruido de preparacion degrada un invariante topologico (numero de Chern) y
(b) un corpus de especificaciones del dispositivo, emitir un plan de
mitigacion de calibracion.

REGLAS ESTRICTAS:
- NO calculas el Chern. No reinterpretas la curva. No propones circuitos.
- Solo recuperas hechos externos del corpus: presupuesto de error de puerta,
  coherencia del equipo, criterio de aceptacion.
- Cada accion recomendada DEBE llevar la fuente del corpus entre corchetes,
  p. ej. [presupuesto-y-criterios.md > Presupuesto de error].
- Si el corpus no dice algo, escribe "no consta en el corpus". No lo inventes.
- Devuelve markdown con: Decision, Infidelidad medida, Acciones (ordenadas por
  coste), Fuentes.
"""


@dataclass
class Plan:
    decision: str
    infidelidad: float
    tasa_acierto: float
    acciones: list[str] = field(default_factory=list)
    fuentes: list[str] = field(default_factory=list)
    texto: str = ""
    via: str = "local"
    modelo: str = ""
    aviso: str = ""


def leer_corpus() -> dict[str, str]:
    """Lee todos los .md del corpus. Siempre disponible."""
    return {p.name: p.read_text(encoding="utf-8")
            for p in sorted(CORPUS.glob("*.md"))}


def _resumen_barrido(res: dict) -> dict:
    """Extrae de cada fase/modelo el punto donde la tasa cae por debajo de 95 %."""
    out = {}
    for modelo, fases in res.get("modelos", {}).items():
        for fase, datos in fases.items():
            filas = datos.get("rows", [])
            if not filas:
                continue
            # ultimo p con tasa >= 95 %
            ok = [r for r in filas if r["frac"] >= CRITERIO_ACEPTAR]
            peor = filas[-1]
            out[f"{fase}/{modelo}"] = {
                "C_exact": datos.get("C_exact"),
                "p_max_ok": ok[-1]["p"] if ok else None,
                "infid_max_ok": 1.0 - ok[-1]["F_muestreada"] if ok else None,
                "infid_peor": 1.0 - peor["F_muestreada"],
                "tasa_peor": peor["frac"],
                "p_peor": peor["p"],
            }
    return out


def _decidir(tasa: float) -> tuple[str, str]:
    if tasa >= CRITERIO_ACEPTAR:
        return "ACEPTAR", "el invariante es fiable"
    if tasa >= CRITERIO_RECHAZAR:
        return "INDECSO", "no se puede afirmar ni negar la fase con una medicion"
    return "RECHAZAR", "el ruido destruye el bundle; no hay senal"


def _plan_local(resumen: dict, corpus: dict[str, str]) -> Plan:
    """Plan sin LLM: determinista, a partir del corpus."""
    if not resumen:
        return Plan("SIN DATOS", 0.0, 0.0, aviso="el barrido no devolvio filas")

    # peor caso entre todas las fases y modelos: la decision se toma con el
    # punto mas degradado, que es el que acota el experimento
    clave = min(resumen, key=lambda k: resumen[k]["tasa_peor"])
    r = resumen[clave]
    decision, motivo = _decidir(r["tasa_peor"])
    infid = r["infid_peor"]

    acciones: list[str] = []
    fuentes: list[str] = []

    if infid > 0.10:
        acciones.append(
            f"Reducir la infidelidad de preparacion: medida {infid:.3f}, "
            "presupuesto 0.10.")
        fuentes.append("presupuesto-y-criterios.md > Presupuesto de error")

    if decision == "INDECSO":
        acciones.append(
            "Reducir la profundidad G: menos puertas ruidosas acumulan menos "
            "infidelidad.")
        acciones.append(
            "Reordenar las puertas: el modelo `shared` muestra que un Pauli al "
            "final preserva el invariante exactamente, y intercalado lo degrada.")
        fuentes.append("mitigacion.md > 1. Reducir la profundidad")
        fuentes.append("mitigacion.md > 2. Reordenar las puertas")

    if decision == "RECHAZAR":
        acciones.append(
            "Recalibracion diaria: la deriva intradia degrada p.")
        acciones.append(
            "Dynamical decoupling si el ruido dominante es de baja frecuencia.")
        fuentes.append("spec-dispositivo-ejemplo.md > Limites operativos")
        fuentes.append("mitigacion.md > 3. Recalibracion diaria")

    acciones.append(
        "Repetir el barrido con mas semillas para acotar la tasa de acierto.")
    fuentes.append("presupuesto-y-criterios.md > Criterio de aceptacion")

    return Plan(decision, infid, r["tasa_peor"], acciones,
                list(dict.fromkeys(fuentes)))


def generar(res: dict) -> Plan:
    """Punto de entrada: barrido (dict) -> Plan."""
    corpus = leer_corpus()
    resumen = _resumen_barrido(res)
    plan = _plan_local(resumen, corpus)

    user = (
        f"Resultado del barrido:\n{json.dumps(resumen, indent=1, ensure_ascii=False)}\n\n"
        f"Corpus de especificaciones:\n" +
        "\n\n".join(f"### {n}\n{c}" for n, c in corpus.items())
    )
    r = llm.preguntar(SYSTEM_PROMPT, user)

    if r.via == "llm":
        plan.texto = r.texto
        plan.via = "llm"
        plan.modelo = r.modelo
    else:
        plan.via = "local"
        plan.aviso = (f"sin LLM ({r.error}). Plan local determinista "
                      "a partir del corpus.")
    return plan


def render(plan: Plan) -> str:
    """Markdown listo para mostrar o para el PDF."""
    lineas = [f"## Plan de mitigacion", "",
              f"**Decision:** {plan.decision}", "",
              f"- Infidelidad de preparacion medida: **{plan.infidelidad:.3f}**",
              f"- Tasa de acierto en el peor punto: **{100*plan.tasa_acierto:.1f} %**",
              ""]
    if plan.acciones:
        lineas.append("### Acciones (ordenadas por coste)")
        for i, a in enumerate(plan.acciones, 1):
            lineas.append(f"{i}. {a}")
        lineas.append("")
    if plan.fuentes:
        lineas.append("### Fuentes")
        for f in plan.fuentes:
            lineas.append(f"- `{f}`")
        lineas.append("")
    if plan.via == "llm":
        lineas.append(f"_Sintetizado por `{plan.modelo}`._")
    elif plan.aviso:
        lineas.append(f"_{plan.aviso}_")
    if plan.texto:
        lineas += ["", "---", "", plan.texto]
    return "\n".join(lineas)
