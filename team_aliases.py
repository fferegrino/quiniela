"""Canonical Liga MX club names, API slugs, and common aliases for tool calls."""

from __future__ import annotations

import re
import unicodedata

# Canonical display names used in news queries.
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

# Liga MX MCP / API slug for each canonical club (from list_teams).
CANONICAL_TO_SLUG: dict[str, str] = {
    "América": "america",
    "Atlas": "atlas",
    "Atlético San Luis": "atletico-san-luis",
    "Cruz Azul": "cruz-azul",
    "Guadalajara": "guadalajara",
    "FC Juárez": "juarez",
    "León": "leon",
    "Mazatlán": "mazatlan",
    "Monterrey": "monterrey",
    "Necaxa": "necaxa",
    "Pachuca": "pachuca",
    "Puebla": "puebla",
    "Pumas UNAM": "pumas",
    "Querétaro": "queretaro",
    "Santos Laguna": "santos-laguna",
    "Tijuana": "tijuana",
    "Toluca": "toluca",
    "Tigres UANL": "tigres",
}

# Lowercased alias → canonical name. Includes the canonical forms themselves.
_ALIAS_TO_CANONICAL: dict[str, str] = {}
_KNOWN_SLUGS: set[str] = set(CANONICAL_TO_SLUG.values())


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
    "atletico-san-luis",
    "san luis",
    "atleti san luis",
)
_register("Cruz Azul", "cruz-azul", "la máquina", "la maquina", "cementeros")
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
_register(
    "FC Juárez",
    "fc juarez",
    "juárez",
    "juarez",
    "bravos",
    "bravos de juárez",
    "bravos de juarez",
)
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
_register("Santos Laguna", "santos", "santos-laguna", "laguneros", "club santos laguna")
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

# Also accept hyphenated slug spellings as aliases where they differ from casefold(canonical).
for _canonical, _slug in CANONICAL_TO_SLUG.items():
    _ALIAS_TO_CANONICAL.setdefault(_slug, _canonical)


def _clean_team_token(name: str) -> str:
    return " ".join(name.strip().split())


def resolve_team_name(name: str) -> str:
    """Return the canonical Liga MX display name when known; otherwise the stripped input."""
    cleaned = _clean_team_token(name)
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


def _ascii_slug(value: str) -> str:
    """Fallback slugify for unknown names (ASCII, hyphenated)."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only.casefold()).strip("-")
    return slug


def resolve_team_slug(name: str) -> str:
    """Return the Liga MX MCP slug when known; otherwise a best-effort slugified token."""
    cleaned = _clean_team_token(name)
    if not cleaned:
        return cleaned

    folded = cleaned.casefold()
    if folded in _KNOWN_SLUGS:
        return folded

    # Hyphen forms already look like slugs; keep if known after light normalization.
    hyphenated = folded.replace(" ", "-")
    if hyphenated in _KNOWN_SLUGS:
        return hyphenated

    canonical = resolve_team_name(cleaned)
    if canonical in CANONICAL_TO_SLUG:
        return CANONICAL_TO_SLUG[canonical]

    return _ascii_slug(cleaned)


def resolve_team_slugs(names: list[str]) -> list[str]:
    """Resolve a list of names to MCP slugs (unique, order preserved)."""
    resolved: list[str] = []
    seen: set[str] = set()
    for name in names:
        slug = resolve_team_slug(name)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        resolved.append(slug)
    return resolved


def canonical_teams() -> list[str]:
    """Return the list of canonical Liga MX club names."""
    return list(CANONICAL_TEAMS)
