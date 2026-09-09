from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv


@dataclass(frozen=True)
class Detection:
    start_us: int
    end_us: int
    cx: float
    cy: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


def _coerce_row(row: list[str], path: Path) -> Detection:
    if len(row) < 6:
        raise ValueError(f"{path}: expected at least 6 columns, got {len(row)}")
    return Detection(
        start_us=int(float(row[0])),
        end_us=int(float(row[1])),
        cx=float(row[2]),
        cy=float(row[3]),
        width=float(row[4]),
        height=float(row[5]),
        confidence=float(row[6]) if len(row) >= 7 and row[6] != "" else 1.0,
    )


def read_detection_file(path: str | Path) -> list[Detection]:
    """Read OrbitSight-style GT/prediction files.

    Supported rows are whitespace/tab separated:
      start_us end_us cx cy width height [confidence]

    A header is allowed. Blank/comment lines are ignored.
    """
    path = Path(path)
    detections: list[Detection] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            row = line.replace(",", " ").split()
            try:
                detections.append(_coerce_row(row, path))
            except ValueError:
                lowered = " ".join(row).lower()
                if any(token in lowered for token in ("window_start", "center_x", "confidence")):
                    continue
                raise
    return detections


def write_prediction_file(path: str | Path, detections: Iterable[Detection]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for d in detections:
            writer.writerow([d.start_us, d.end_us, f"{d.cx:.6f}", f"{d.cy:.6f}", f"{d.width:.6f}", f"{d.height:.6f}", f"{d.confidence:.8f}"])
