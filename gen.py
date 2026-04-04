#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║    No Man's Sky Mesh → Base Parts Converter  (generalized)      ║
╚══════════════════════════════════════════════════════════════════╝

Dependencies:
    pip install trimesh numpy

Usage (CLI):
    python nms_converter.py mymodel.obj out.json \\
        --parts T_FLOOR.fbx T_WALL.fbx T_ROOF6.fbx T_DOOR1.fbx ...

Usage (Python API):
    import trimesh
    from nms_converter import load_nms_parts, convert

    mesh  = trimesh.load("mymodel.obj", force="mesh")
    parts = load_nms_parts(["T_FLOOR.fbx", "T_WALL.fbx", "T_ROOF6.fbx"])
    convert(mesh, "out.json", parts, scale=0.05)
"""

from __future__ import annotations

import json
import math
import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from functions import fbx_to_trimesh

import numpy as np

try:
    import trimesh
except ImportError:
    sys.exit("trimesh is required:  pip install trimesh")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
NMS_FLOOR_SIZE          = 5.4   # one floor tile = 5.4 NMS world-units edge-to-edge
NMS_USERDATA_STRUCTURAL = 74    # structural/building parts
NMS_USERDATA_FUNCTIONAL = 0     # machines, lights, decorations …


# ─────────────────────────────────────────────────────────────────────────────
# NMS Part descriptor
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class NMSPart:
    """
    Everything the converter needs to know about one NMS base part.

    object_id     – The NMS internal ID, e.g. "^T_FLOOR".
                    Inferred automatically as  '^' + FBX-filename-stem.

    user_data     – Save-file field; 74 for structural parts, 0 for functional.

    facing_normal – Unit vector (axis-aligned) pointing along the part's
                    "open face".  Derived from the thinnest bounding-box axis
                    of the FBX model:
                      thin Y → (0, 1, 0)  ← floor / ceiling panels
                      thin X → (1, 0, 0)  ← wall running along Z
                      thin Z → (0, 0, 1)  ← wall running along X

    extents       – Raw bounding-box dimensions (dx, dy, dz) in FBX units.

    fbx_path      – Source file (informational).
    """
    object_id:     str
    user_data:     int
    facing_normal: np.ndarray   # unit vector, axis-aligned
    extents:       np.ndarray   # (dx, dy, dz)
    fbx_path:      str = ""

    def __repr__(self) -> str:
        return (f"NMSPart({self.object_id!r}, "
                f"facing={self.facing_normal}, "
                f"extents={self.extents.round(2)})")


# ─────────────────────────────────────────────────────────────────────────────
# FBX part loading
# ─────────────────────────────────────────────────────────────────────────────
def load_nms_part(fbx_path: str, user_data: int = NMS_USERDATA_STRUCTURAL) -> NMSPart:
    """
    Load a single FBX file and produce an NMSPart descriptor.

    The ObjectID is inferred as  '^' + uppercased filename stem.
    e.g.  "models/T_FLOOR.fbx"  →  "^T_FLOOR"

    The facing direction is determined by finding the thinnest bounding-box
    axis; that axis is the part's face-normal direction.
    """
    path = Path(fbx_path)
    if not path.exists():
        raise FileNotFoundError(f"FBX not found: {path}")

    object_id = "^" + path.stem.upper()
    mesh: trimesh.Trimesh = fbx_to_trimesh(fbx_path)  # convert from whatever FBX structure into a single mesh

    if mesh.is_empty:
        raise ValueError(f"FBX loaded but contains no geometry: {path}")

    extents = np.array(mesh.bounding_box.extents, dtype=float)   # (dx, dy, dz)

    # The axis with the smallest extent is the "flat" face direction
    thin_axis             = int(np.argmin(extents))
    facing                = np.zeros(3, dtype=float)
    facing[thin_axis]     = 1.0

    return NMSPart(
        object_id=object_id,
        user_data=user_data,
        facing_normal=facing,
        extents=extents,
        fbx_path=str(path),
    )


def load_nms_parts(
    fbx_paths: list[str],
    user_data: int = NMS_USERDATA_STRUCTURAL,
) -> list[NMSPart]:
    """
    Load a list of FBX files into NMSPart descriptors.
    Prints a summary line for each part.
    Files that fail to load emit a warning and are skipped.
    Raises RuntimeError if *no* parts load successfully.
    """
    parts: list[NMSPart] = []
    for p in fbx_paths:
        try:
            part = load_nms_part(p, user_data)
            parts.append(part)
            print(f"  Loaded  {part.object_id:35s}"
                  f"  extents={part.extents.round(2)}"
                  f"  facing={part.facing_normal}")
        except Exception as exc:
            print(f"  WARNING: skipping {p} — {exc}", file=sys.stderr)

    if not parts:
        raise RuntimeError(
            "No NMS parts could be loaded from the provided FBX files. "
            "Make sure trimesh can open them (pip install trimesh[easy])."
        )
    return parts


# ─────────────────────────────────────────────────────────────────────────────
# Part selection
# ─────────────────────────────────────────────────────────────────────────────
def best_part_for_normal(
    surface_normal: np.ndarray,
    parts: list[NMSPart],
) -> NMSPart:
    """
    Return the NMSPart whose facing_normal best aligns with the given surface
    normal.  We use |dot| so both +Y and −Y normals can match a horizontal
    panel (a floor tile lying flat has the same shape whether the face points
    up or down).
    """
    best      = parts[0]
    best_score = -1.0
    for part in parts:
        score = abs(float(np.dot(surface_normal, part.facing_normal)))
        if score > best_score:
            best_score = score
            best       = part
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Orientation vectors
# ─────────────────────────────────────────────────────────────────────────────
def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def normal_to_up_at(normal: np.ndarray) -> tuple[list[float], list[float]]:
    """
    Produce the NMS (Up, At) orientation vectors for a surface normal.

    All NMS structural parts share the same convention observed in real
    save data:
      Up  ≈ (0, 1, 0)  — always world-up; NMS never tilts parts via this field.
      At  = surface normal projected onto the XZ plane, normalised.
            This encodes yaw-only rotation (which way the part faces).

    For a purely vertical normal (straight up / down) the At direction is
    arbitrary; we default to +Z.
    """
    n     = _norm(np.asarray(normal, dtype=float))
    up    = [0.0, 1.0, 0.0]

    at_xz = np.array([n[0], 0.0, n[2]])
    mag   = np.linalg.norm(at_xz)
    at    = (at_xz / mag).tolist() if mag > 1e-6 else [0.0, 0.0, 1.0]

    return up, at


# ─────────────────────────────────────────────────────────────────────────────
# Surface sampling
# ─────────────────────────────────────────────────────────────────────────────
def _face_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    n  = np.cross(e1, e2)
    lengths = np.linalg.norm(n, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    return n / lengths


def _face_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    return np.linalg.norm(np.cross(e1, e2), axis=1) * 0.5


def sample_surface(
    mesh: trimesh.Trimesh,
    samples_per_sq_unit: float = 6.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample points on every triangle of *mesh*, proportional to
    face area.  Returns (positions [N,3], normals [N,3]).
    """
    v = np.array(mesh.vertices, dtype=float)
    f = np.array(mesh.faces,    dtype=int)

    normals = _face_normals(v, f)
    areas   = _face_areas(v, f)
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]

    all_pos: list[np.ndarray] = []
    all_norm: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for i in range(len(f)):
        n_pts = max(1, int(math.ceil(areas[i] * samples_per_sq_unit)))

        # Uniform barycentric sampling (Osada et al. 2002)
        r     = rng.random((n_pts, 2))
        mask  = r[:, 0] + r[:, 1] > 1.0
        r[mask] = 1.0 - r[mask]

        pts = (r[:, 0:1] * (v1[i] - v0[i]) +
               r[:, 1:2] * (v2[i] - v0[i]) +
               v0[i])
        all_pos.append(pts)
        all_norm.append(np.tile(normals[i], (n_pts, 1)))

    return np.vstack(all_pos), np.vstack(all_norm)


# ─────────────────────────────────────────────────────────────────────────────
# Voxelisation
# ─────────────────────────────────────────────────────────────────────────────
def voxelise(
    positions: np.ndarray,
    normals:   np.ndarray,
    voxel_size: float,
) -> dict[tuple[int, int, int], np.ndarray]:
    """
    Snap sample positions to a 3-D grid and average normals per cell.
    Returns  {(ix, iy, iz): normalised_average_normal}.
    """
    idx   = np.round(positions / voxel_size).astype(int)
    cells: dict[tuple, list] = defaultdict(list)
    for key, n in zip(map(tuple, idx), normals):
        cells[key].append(n)
    return {k: _norm(np.mean(v, axis=0)) for k, v in cells.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────────────────────
def convert(
    target_mesh:            trimesh.Trimesh,
    output_path:     str,
    part_fbx_paths:       list[NMSPart],
    scale:           float = 1.0,
    voxel_size:      float = NMS_FLOOR_SIZE,
    samples:         float = 6.0,
    center:          bool  = True,
    timestamp_start: int   = 1766251924,
) -> list[dict]:
    """
    Convert a trimesh.Trimesh into an NMS base-parts JSON file.

    Parameters
    ----------
    mesh            : The source mesh to voxelise and reconstruct.
    output_path     : Path to write the output .json file.
    part_fbx_paths  : List of paths to FBX files containing part models.
    scale           : Uniform scale applied to the mesh before processing.
    voxel_size      : Grid step in NMS world-units.  One floor tile = 5.4.
    samples         : Surface sample density (points per square unit).
                      Higher = fewer missed thin features, slower.
    center          : Auto-centre the mesh at the world origin.
    timestamp_start : Starting Timestamp value for emitted records.

    Returns
    -------
    list[dict]  The list of part records (also written to output_path).
    """
    
    if not part_fbx_paths:
        raise ValueError("part_fbx_paths is empty — provide at least one FBX path.")
    nms_parts = load_nms_parts(part_fbx_paths)

    # ── Pre-process mesh ────────────────────────────────────────────
    m = target_mesh.copy()
    m.apply_scale(scale)
    if center:
        m.apply_translation(-m.bounding_box.centroid)

    bb = np.array(m.bounding_box.extents)

    print("─" * 64)
    print(f"  Vertices     : {len(m.vertices):,}")
    print(f"  Faces        : {len(m.faces):,}")
    print(f"  Extent       : {bb[0]:.2f} × {bb[1]:.2f} × {bb[2]:.2f}  NMS units")
    print(f"  Scale        : {scale}")
    print(f"  Voxel size   : {voxel_size}  ({voxel_size / NMS_FLOOR_SIZE:.2f} floor-tiles)")
    print(f"  Parts pool   : {len(nms_parts)} parts")
    print("─" * 64)

    # ── Sample + voxelise ───────────────────────────────────────────
    print("  Sampling surface …")
    positions, normals_s = sample_surface(m, samples)
    print(f"  Surface samples : {len(positions):,}")

    voxels = voxelise(positions, normals_s, voxel_size)
    print(f"  Voxel cells     : {len(voxels):,}")

    # ── Assign parts + build output ─────────────────────────────────
    out_records: list[dict] = []
    usage: dict[str, int]   = defaultdict(int)

    for t, ((ix, iy, iz), normal) in enumerate(sorted(voxels.items())):
        chosen = best_part_for_normal(normal, nms_parts)
        usage[chosen.object_id] += 1

        pos    = [float(ix * voxel_size),
                  float(iy * voxel_size),
                  float(iz * voxel_size)]
        up, at = normal_to_up_at(normal)

        out_records.append({
            "Timestamp": timestamp_start + t,
            "ObjectID":  chosen.object_id,
            "UserData":  chosen.user_data,
            "Position":  pos,
            "Up":        up,
            "At":        at,
        })

    # ── Summary ─────────────────────────────────────────────────────
    print("  Part usage:")
    for pid, cnt in sorted(usage.items(), key=lambda x: -x[1]):
        print(f"    {pid:40s} × {cnt:,}")

    # ── Write JSON ──────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(out_records, fh, indent=2)

    print("─" * 64)
    print(f"  Written {len(out_records):,} parts → {out}")
    print("─" * 64)
    return out_records

