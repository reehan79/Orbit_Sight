"""Official public-timeline output contract (serialization only).

Frozen detector rows remain (ws, we, cx, cy, w, h, confidence).
This module only adapts those rows for ChallengeON packaging.
"""

from __future__ import annotations

import os
from pathlib import Path

# Dataset docs: event column `label` — 0 = background, 1 = RSO Object.
# Organizer class_id numeric confirmation still pending; default follows RSO=1.
DEFAULT_CLASS_ID = 1
CLASS_ID_DOC_SOURCE = "OrbitSight_DataLoader/README.md + dataset description: label 1 = RSO"

FILENAME_MODE_ADMIN_PRED = "admin_pred"  # <sequence>_pred.txt (admin answer #1)
FILENAME_MODE_PUBLIC_PLAIN = "public_plain"  # <sequence>.txt (public timeline wording)
DEFAULT_FILENAME_MODE = FILENAME_MODE_ADMIN_PRED

# Public-timeline nine fields (organizer written clarification 2026-09-09).
PUBLIC_COLUMNS = (
    "sequence_id",
    "window_start_timestamp_us",
    "window_end_timestamp_us",
    "centre_x",
    "centre_y",
    "w",
    "h",
    "class_id",
    "confidence",
)

# Organizer materials specify TSV + header for submission tables historically;
# public timeline did not restate delimiter. We use TSV + exact header of nine fields.
PUBLIC_DELIMITER = "\t"
PUBLIC_HEADER_REQUIRED = True
PUBLIC_FORMAT_DECISION = (
    "TSV with exact nine-field header row. Delimiter/header not restated in the "
    "latest public-timeline reply; chosen for consistency with OrbitSight_DataLoader "
    "README submission tables (tab-separated + header)."
)


def resolved_class_id(override: int | None = None) -> int:
    if override is not None:
        return int(override)
    env = os.environ.get("ORBITSIGHT_CLASS_ID")
    if env is not None and env.strip() != "":
        return int(env)
    return int(DEFAULT_CLASS_ID)


def resolved_filename_mode(override: str | None = None) -> str:
    mode = (override or os.environ.get("ORBITSIGHT_PRED_FILENAME_MODE") or DEFAULT_FILENAME_MODE).strip()
    if mode not in (FILENAME_MODE_ADMIN_PRED, FILENAME_MODE_PUBLIC_PLAIN):
        raise ValueError(
            f"unknown ORBITSIGHT_PRED_FILENAME_MODE={mode!r}; "
            f"expected {FILENAME_MODE_ADMIN_PRED!r} or {FILENAME_MODE_PUBLIC_PLAIN!r}"
        )
    return mode


def prediction_filename(sequence: str, mode: str | None = None) -> str:
    """Single configurable prediction filename (never emits both variants)."""
    m = resolved_filename_mode(mode)
    if m == FILENAME_MODE_PUBLIC_PLAIN:
        return f"{sequence}.txt"
    return f"{sequence}_pred.txt"


def write_public_prediction_file(
    path: Path,
    sequence_id: str,
    rows: list[tuple],
    *,
    class_id: int | None = None,
) -> Path:
    """Serialize frozen internal rows to the official public nine-column TSV.

    Internal row: (ws, we, cx, cy, w, h, confidence) — values unchanged except
    integer rounding of geometry identical to prior TII writer (round then int).
    """
    cid = resolved_class_id(class_id)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(PUBLIC_DELIMITER.join(PUBLIC_COLUMNS) + "\n")
        for ws, we, cx, cy, w, h, conf in rows:
            handle.write(
                PUBLIC_DELIMITER.join(
                    [
                        str(sequence_id),
                        str(int(ws)),
                        str(int(we)),
                        str(int(round(cx))),
                        str(int(round(cy))),
                        str(int(round(w))),
                        str(int(round(h))),
                        str(int(cid)),
                        str(float(conf)),
                    ]
                )
                + "\n"
            )
    return path
