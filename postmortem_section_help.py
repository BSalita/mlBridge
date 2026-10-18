"""Section-title wording shared by the postmortem report.

A personal name is used only when one player is the declarer. Other
"you" and "your" stay partnership words. Given names are shown unless
the two players share a first name, in which case the full names are
used. "we declared" and "we played" are written "our partnership
declared" and "our partnership played".
"""

from __future__ import annotations

import re


def section_waits_on_row_click(favorite_id: str) -> bool:
    """Ranking sections rebuild the report grids when a row is clicked."""
    return "rank" in str(favorite_id or "").casefold()

_HELP_PLAYER1_RE = re.compile(r"\{player1_name\}")
_HELP_PLAYER2_RE = re.compile(r"\{player2_name\}")
_HELP_YOU_AND_PARTNER_RE = re.compile(r"\byou and your partner\b", re.IGNORECASE)
_HELP_YOUR_PARTNER_RE = re.compile(r"\byour partner\b", re.IGNORECASE)
_HELP_PARTNER_DECLARED_RE = re.compile(
    r"\byour partner(?=\s+declared\b)",
    re.IGNORECASE,
)
_HELP_YOU_DECLARED_RE = re.compile(r"\byou(?=\s+declared\b)", re.IGNORECASE)
_HELP_PARTNERSHIP_RE = re.compile(r"\b(?:your|our) partnership\b", re.IGNORECASE)
_HELP_PARTNERSHIP_DECLARED_RE = re.compile(
    r"\b(?:your|our) partnership(?=\s+declared\b)",
    re.IGNORECASE,
)
_HELP_YOUR_RE = re.compile(r"\byour\b(?!\s+partner(?:ship)?\b)", re.IGNORECASE)
_HELP_YOU_RE = re.compile(r"\byou\b(?!\s+and\s+your\s+partner\b)", re.IGNORECASE)
_HELP_YOUR_OPPONENT_RE = re.compile(r"\byour opponent\b", re.IGNORECASE)
_HELP_WE_ACTION_RE = re.compile(r"\bwe (declared|played)\b", re.IGNORECASE)
# One person declares a board. Naming both players as declarers is wrong
# when the opponents gifted the contract.
_HELP_GIFTED_RE = re.compile(
    r"\b(?:your|our) partnership declared which your opponents gifted "
    r"(?:you|your partnership) the contract or tricks\b",
    re.IGNORECASE,
)


def _replace_help_token(pattern: re.Pattern[str], replacement: str, text: str) -> str:
    if not replacement:
        return text
    return pattern.sub(lambda _match: replacement, text)


def _is_all_caps_name(token: str) -> bool:
    letters = [char for char in token if char.isalpha()]
    return bool(letters) and all(char.isupper() for char in letters)


def given_name(full_name: str) -> str | None:
    """First given name, when the surname can be told apart.

    A comma means the text after it is the surname. Otherwise an all-caps
    word is the surname (French convention). Returns None when that split
    is not possible.
    """
    text = " ".join(str(full_name or "").split())
    if not text:
        return None
    if "," in text:
        given, surname = text.split(",", 1)
        given = given.strip()
        surname = surname.strip()
        if not given or not surname:
            return None
        return given.split()[0]
    tokens = text.split()
    given = [token for token in tokens if not _is_all_caps_name(token)]
    surnames = [token for token in tokens if _is_all_caps_name(token)]
    if not given or not surnames:
        return None
    return given[0]


def help_display_names(player1: str, player2: str = "") -> tuple[str, str]:
    """Given names for a report pair, or the full name when that would be ambiguous."""
    first_full = " ".join(str(player1 or "").split())
    second_full = " ".join(str(player2 or "").split())
    if not first_full:
        return "", ""
    given1 = given_name(first_full)
    given2 = given_name(second_full) if second_full else None
    if given1 and given2 and given1.casefold() == given2.casefold():
        return first_full, second_full
    show1 = given1 or first_full
    show2 = (given2 or second_full) if second_full else ""
    return show1, show2


def personalize_section_help(help_text: str, player1: str, player2: str = "") -> str:
    """Replace you/partner in a section title with the report pair's names.

    ``player1`` is the selected player. ``player2`` is the partner. Each is
    shown as a given name, or the full name when both given names match.
    A personal name is used only when one player is the declarer. Other
    "you" and "your" stay partnership words ("our partnership", "we", "our").
    Without a partner name, partner wording stays as written.
    """
    text = str(help_text or "")
    first, second = help_display_names(player1, player2)
    if not text or not first:
        return text
    text = _HELP_GIFTED_RE.sub(
        "we declared and the opponents gifted the contract or tricks",
        text,
    )
    text = _replace_help_token(_HELP_PLAYER1_RE, first, text)
    text = _replace_help_token(_HELP_PLAYER2_RE, second, text)
    text = _HELP_YOUR_OPPONENT_RE.sub("the opponent", text)
    text = _HELP_YOU_DECLARED_RE.sub(lambda _match: first, text)
    text = _HELP_PARTNERSHIP_DECLARED_RE.sub("we", text)
    if second:
        pair = f"{first} and {second}"
        text = _replace_help_token(_HELP_YOU_AND_PARTNER_RE, pair, text)
        text = _HELP_PARTNER_DECLARED_RE.sub(lambda _match: second, text)
        text = _replace_help_token(_HELP_YOUR_PARTNER_RE, "our partner", text)
    text = _replace_help_token(_HELP_PARTNERSHIP_RE, "our partnership", text)
    text = _replace_help_token(_HELP_YOUR_RE, "our", text)
    text = _replace_help_token(_HELP_YOU_RE, "we", text)
    text = _HELP_WE_ACTION_RE.sub(r"our partnership \1", text)
    return text


def section_help_for_players(
    help_text: str,
    player1: str,
    player2: str = "",
    *,
    language: str = "en",
) -> str:
    """Fill ``{player1_name}`` and ``{player2_name}`` from the report pair.

    English titles also rewrite you/your/we. Other languages keep their
    wording and only receive the name tokens.
    """
    text = str(help_text or "")
    if str(language or "en").strip().lower() == "en":
        return personalize_section_help(text, player1, player2)
    first, second = help_display_names(player1, player2)
    if not text or not first:
        return text
    text = _replace_help_token(_HELP_PLAYER1_RE, first, text)
    if second:
        text = _replace_help_token(_HELP_PLAYER2_RE, second, text)
    return text
