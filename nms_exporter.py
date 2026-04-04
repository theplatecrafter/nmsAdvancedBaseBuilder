"""
core/nms_exporter.py
====================
Converts a list of placed parts (from surface_fitter) into the NMS
base building JSON format and writes it to disk.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable


def _make_object(
    object_id: str,
    position: list[float],
    up: list[float],
    at: list[float],
    user_data: int = 0,
    timestamp: int = 0,
) -> dict:
    return {
        "Timestamp": timestamp,
        "ObjectID": object_id,
        "UserData": user_data,
        "Position": [float(x) for x in position],
        "Up":       [float(x) for x in up],
        "At":       [float(x) for x in at],
    }


def build_nms_json(
    placed_parts: list[dict],
    timestamp: int | None = None,
    log: Callable = print,
) -> list[dict]:
    """
    Build the full NMS JSON array from a list of placed-part dicts.

    Each placed-part dict must contain:
        part_info  : PartInfo  — for the ObjectID
        position   : [x, y, z]
        up         : [x, y, z]
        at         : [x, y, z]
    """
    ts = timestamp or int(time.time())
    objects: list[dict] = []

    # Every NMS base requires a BASE_FLAG anchor at the origin
    objects.append(_make_object(
        object_id="^BASE_FLAG",
        position=[0.0, 0.0, 0.0],
        up=[5.9604651880817983e-08, 1.0000002384185791, 0.0],
        at=[-4.2840834879598333e-08, 0.0, 1.0],
        user_data=0,
        timestamp=ts,
    ))

    for p in placed_parts:
        objects.append(_make_object(
            object_id=p["part_info"].object_id,
            position=p["position"],
            up=p["up"],
            at=p["at"],
            user_data=0,
            timestamp=ts,
        ))

    log(f"[exporter] {len(objects):,} total objects (incl. BASE_FLAG)")
    return objects


def write_nms_json(
    placed_parts: list[dict],
    output_path: str | Path,
    timestamp: int | None = None,
    log: Callable = print,
) -> Path:
    """Build and write the NMS JSON file. Returns the output Path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    objects = build_nms_json(placed_parts, timestamp=timestamp, log=log)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(objects, f, indent=2)

    log(f"[exporter] Written → {output_path}")
    return output_path