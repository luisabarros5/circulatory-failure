"""Feature selection and fail-closed leakage checks for temporal dataset v2."""

from __future__ import annotations

import re
from collections.abc import Iterable


KEY_COLUMNS = ("stay_id", "charttime")

EXACT_FORBIDDEN = {
    "falencia",
    "vasopressor_ativo",
    "target_onset_0_8h",
    "target_first_onset_0_8h",
    "eligible_for_training",
    "at_onset",
    "post_onset",
    "episode_id",
    "first_onset",
    "failure_state_new_at_t",
}

# Vasoativo/inotrópico usado na definição do endpoint.
VASOACTIVE_ITEM_PATTERNS = (
    r"(^|_)lab_50990($|_)",  # Theophylline e todas as derivações.
)

# Exames excluídos da matriz X após auditoria global:
# zero observações reais, ou somente 3–4 observações sem variância útil.
RARE_IRRELEVANT_ITEM_PATTERNS = (
    r"(^|_)lab_50928($|_)",  # Gastrin
    r"(^|_)lab_50975($|_)",  # Protein electrophoresis
    r"(^|_)lab_51290($|_)",  # Sickle cell preparation
    r"(^|_)lab_51291($|_)",  # Sickle cells
    r"(^|_)lab_51292($|_)",  # Spherocytes
    r"(^|_)lab_51643($|_)",  # Hemoglobin A2
    r"(^|_)lab_51647($|_)",  # Hemoglobin S
    r"(^|_)lab_51568($|_)",  # Neonatal bilirubin
    r"(^|_)lab_51569($|_)",  # Neonatal direct bilirubin
    r"(^|_)lab_51570($|_)",  # Neonatal indirect bilirubin
)

FORBIDDEN_PATTERNS = (
    r"vasopress",
    r"vasoactive",
    r"vasoativo",
    r"pressor",
    r"inotrop",
    r"norepine",
    r"noradren",
    r"epineph",
    r"adrenalin",
    r"dopamin",
    r"dobut",
    r"milrin",
    r"phenyleph",
    r"fenilefr",
    r"levosim",
    r"theophyll",
    r"teofil",
)

ENDPOINT_PATTERNS = (
    r"(^|_)target($|_)",
    r"(^|_)label($|_)",
    r"(^|_)onset($|_)",
    r"post_onset",
    r"eligible_for_training",
    r"censor",
    r"failure_state",
    r"circulatory_failure",
    r"falencia",
)


def forbidden_reason(name: str) -> str | None:
    low = name.strip().casefold()

    if low in EXACT_FORBIDDEN:
        return "exact_forbidden"

    if any(
        re.search(pattern, low)
        for pattern in VASOACTIVE_ITEM_PATTERNS
    ):
        return "vasoactive_or_inotrope_itemid"

    if any(
        re.search(pattern, low)
        for pattern in RARE_IRRELEVANT_ITEM_PATTERNS
    ):
        return "rare_irrelevant_itemid"

    if any(
        re.search(pattern, low)
        for pattern in FORBIDDEN_PATTERNS
    ):
        return "vasoactive_or_inotrope"

    if any(
        re.search(pattern, low)
        for pattern in ENDPOINT_PATTERNS
    ):
        return "endpoint_or_future_label"

    return None


def assert_no_forbidden_features(columns: Iterable[str]) -> None:
    violations = {
        name: forbidden_reason(name)
        for name in columns
    }
    violations = {
        name: reason
        for name, reason in violations.items()
        if reason
    }

    if violations:
        rendered = ", ".join(
            f"{name} ({reason})"
            for name, reason in violations.items()
        )
        raise ValueError(
            f"Forbidden features reached model matrix: {rendered}"
        )


def select_allowed_features(columns: Iterable[str]) -> list[str]:
    selected = [
        name
        for name in columns
        if name not in KEY_COLUMNS
        and forbidden_reason(name) is None
    ]

    assert_no_forbidden_features(selected)
    return selected
