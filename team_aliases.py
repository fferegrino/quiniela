"""Canonical Liga MX club names and common aliases for tool calls."""

from __future__ import annotations

# Canonical display names used in queries and tool arguments.
CANONICAL_TEAMS: tuple[str, ...] = (
    "América",
    "Atlas",
    "Atlético San Luis",
    "Cruz Azul",
    "Guadalajara",
    "FC Juárez",
    "León",
    "Mazatlán",
    "Monterrey",
    "Necaxa",
    "Pachuca",
    "Puebla",
    "Pumas UNAM",
    "Querétaro",
    "Santos Laguna",
    "Tijuana",
    "Toluca",
    "Tigres UANL",
)

# Lowercased alias → canonical name. Includes the canonical forms themselves.
_ALIAS_TO_CANONICAL: dict[str, str] = {}


def _register(canonical: str, *aliases: str) -> None:
    _ALIAS_TO_CANONICAL[canonical.casefold()] = canonical
    for alias in aliases:
        key = alias.casefold().strip()
        if key:
            _ALIAS_TO_CANONICAL[key] = canonical


_register("América", "america", "club america", "club américa", "águilas", "aguilas")
_register("Atlas", "atlas fc", "zapatos")
_register(
    "Atlético San Luis",
    "atletico san luis",
    "atlético de san luis",
    "atletico de san luis",
    "san luis",
    "atleti san luis",
)
_register("Cruz Azul", "la máquina", "la maquina", "cementeros")
_register(
    "Guadalajara",
    "chivas",
    "chivas rayadas",
    "chivas del guadalajara",
    "cd guadalajara",
    "club deportivo guadalajara",
    "rebaño",
    "rebano",
)
_register("FC Juárez", "fc juarez", "juárez", "juarez", "bravos", "bravos de juárez", "bravos de juarez")
_register("León", "leon", "club león", "club leon", "esmeraldas", "panzas verdes")
_register("Mazatlán", "mazatlan", "mazatlán fc", "mazatlan fc", "cañoneros", "canoneros")
_register("Monterrey", "rayados", "cf monterrey", "club de fútbol monterrey", "club de futbol monterrey")
_register("Necaxa", "rayos", "club necaxa")
_register("Pachuca", "tuzos", "cf pachuca", "club de fútbol pachuca", "club de futbol pachuca")
_register("Puebla", "la franja", "club puebla", "puebla fc")
_register(
    "Pumas UNAM",
    "pumas",
    "unam",
    "universidad",
    "universidad nacional",
    "club universidad nacional",
)
_register("Querétaro", "queretaro", "gallos", "gallos blancos", "querétaro fc", "queretaro fc")
_register("Santos Laguna", "santos", "laguneros", "club santos laguna")
_register("Tijuana", "xolos", "club tijuana", "xoloitzcuintles")
_register("Toluca", "diablos", "diablos rojos", "deportivo toluca", "toluca fc")
_register(
    "Tigres UANL",
    "tigres",
    "uanl",
    "tigres de la uanl",
    "club de fútbol tigres de la uanl",
    "club de futbol tigres de la uanl",
)


def resolve_team_name(name: str) -> str:
    """Return the canonical Liga MX name when known; otherwise the stripped input."""
    cleaned = " ".join(name.strip().split())
    if not cleaned:
        return cleaned
    return _ALIAS_TO_CANONICAL.get(cleaned.casefold(), cleaned)


def resolve_team_names(names: list[str]) -> list[str]:
    """Resolve a list of team names, dropping empties and preserving order (unique)."""
    resolved: list[str] = []
    seen: set[str] = set()
    for name in names:
        canonical = resolve_team_name(name)
        if not canonical:
            continue
        key = canonical.casefold()
        if key in seen:
            continue
        seen.add(key)
        resolved.append(canonical)
    return resolved


def canonical_teams() -> list[str]:
    """Return the list of canonical Liga MX club names."""
    return list(CANONICAL_TEAMS)
