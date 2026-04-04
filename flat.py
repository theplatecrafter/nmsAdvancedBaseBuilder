#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║    No Man's Sky Mesh → Base Parts Converter                      ║
║    Converts .obj / .stl files into NMS base-building JSON        ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python nms_converter.py <input.obj|stl> <output.json> [options]

Options:
    --scale FLOAT         Scale factor applied to mesh vertices (default: 1.0)
    --voxel-size FLOAT    NMS grid step in units (default: 5.4 = 1 floor tile)
    --samples FLOAT       Surface samples per square unit (default: 6)
    --no-center           Don't auto-center the mesh at origin
    --shell               Only output surface shell (default: surface only)
    --timestamp INT       Starting timestamp value (default: 1766251924)

Example:
    python nms_converter.py mybase.obj mybase.json --scale 0.1 --voxel-size 5.4

The output JSON can be imported with tools like:
  • NMS Save Editor  (https://nomansskysaveeditor.com/)
  • GoatFungus NMS Save Editor
  • nms-base-builder Blender addon

How the algorithm works:
  1. Parse the OBJ/STL mesh into vertices + triangulated faces.
  2. Densely sample points on every triangle surface (proportional to area).
  3. Snap each sample to the NMS voxel grid (5.4 unit floor tiles by default).
  4. Average the surface normal for every voxel cell.
  5. Classify each cell as floor / wall / roof by how much its normal
     points in the Y direction.
  6. Compute the NMS orientation vectors (Up, At) from the averaged normal.
  7. Emit JSON records matching the NMS base-object save format.

Part IDs used:
  ^T_FLOOR   – horizontal surface (normal mostly +Y)
  ^T_WALL    – vertical surface   (normal mostly horizontal)
  ^T_ROOF6   – ceiling / overhang (normal mostly -Y)
"""

import json
import math
import struct
import argparse
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import trimesh

# ─────────────────────────────────────────────────────────────────────────────
# NMS constants
# ─────────────────────────────────────────────────────────────────────────────
NMS_FLOOR_SIZE = 5.4        # Floor-tile edge length in NMS world units
NMS_USERDATA_STRUCTURAL = 74
NMS_USERDATA_FUNCTIONAL = 0

# ObjectID → UserData mapping (structural parts use 74)
PART_CATALOGUE = {
    "floor":   ("^T_FLOOR",  NMS_USERDATA_STRUCTURAL),
    "wall":    ("^T_WALL",   NMS_USERDATA_STRUCTURAL),
    "ceiling": ("^T_ROOF6",  NMS_USERDATA_STRUCTURAL),
}


# ─────────────────────────────────────────────────────────────────────────────
# Vector helpers
# ─────────────────────────────────────────────────────────────────────────────
def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def normal_to_up_at(normal: np.ndarray):
    """
    Convert a surface normal into NMS (Up, At) orientation vectors.

    Key insight from real NMS save data: ALL structural parts (floors, walls,
    roofs) store Up ≈ (0, 1, 0) — always world-up. The part shape (flat tile,
    standing wall panel, etc.) is entirely determined by the ObjectID model.
    The At vector is simply a *horizontal* unit vector in the XZ plane that
    says which way the part "faces" — it carries the yaw rotation only.

    For walls:   At = surface normal projected onto XZ plane (facing direction)
    For floors / ceilings: At = any horizontal direction; we use the XZ
                               projection of the normal if non-zero, else +Z.
    """
    n      = _norm(np.asarray(normal, dtype=float))
    up     = np.array([0.0, 1.0, 0.0])   # always world-up for all NMS parts

    # Project normal onto the XZ plane to get a horizontal facing direction
    at_xz  = np.array([n[0], 0.0, n[2]])
    length = np.linalg.norm(at_xz)

    if length < 1e-6:
        # Normal is straight up or down — facing direction is arbitrary
        at = np.array([0.0, 0.0, 1.0])
    else:
        at = at_xz / length

    return up.tolist(), at.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Mesh geometry helpers
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


# ─────────────────────────────────────────────────────────────────────────────
# Surface sampling + voxelisation
# ─────────────────────────────────────────────────────────────────────────────
def sample_surface(v: np.ndarray, f: np.ndarray, samples_per_sq_unit: float = 6.0):
    """
    Randomly sample points on triangle surfaces proportional to their area.
    Returns (positions [N,3], normals [N,3]).
    """
    normals = _face_normals(v, f)
    areas   = _face_areas(v, f)
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]

    all_pos, all_norm = [], []
    rng = np.random.default_rng(42)

    for i in range(len(f)):
        n_pts = max(1, int(math.ceil(areas[i] * samples_per_sq_unit)))
        # Barycentric random samples (Osada et al. 2002)
        r = rng.random((n_pts, 2))
        mask = r[:, 0] + r[:, 1] > 1.0
        r[mask] = 1.0 - r[mask]
        pts = (r[:, 0:1] * (v1[i] - v0[i]) +
               r[:, 1:2] * (v2[i] - v0[i]) +
               v0[i])
        all_pos.append(pts)
        all_norm.append(np.tile(normals[i], (n_pts, 1)))

    return np.vstack(all_pos), np.vstack(all_norm)


def voxelise(positions: np.ndarray, normals: np.ndarray, voxel_size: float):
    """
    Snap positions to a 3-D grid and average normals per cell.
    Returns dict  (ix, iy, iz) → normalised average normal.
    """
    idx = np.round(positions / voxel_size).astype(int)
    cells: dict[tuple, list] = defaultdict(list)
    for key, n in zip(map(tuple, idx), normals):
        cells[key].append(n)

    return {k: _norm(np.mean(v, axis=0)) for k, v in cells.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────
def classify(normal: np.ndarray, floor_thresh: float = 0.70) -> str:
    """Return 'floor', 'wall', or 'ceiling' based on the surface normal."""
    ny = float(normal[1])
    if ny >  floor_thresh: return "floor"
    if ny < -floor_thresh: return "ceiling"
    return "wall"


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion entry point
# ─────────────────────────────────────────────────────────────────────────────
def convert(
    target_mesh:      trimesh.Trimesh,
    output_path:     str,
    voxel_size:      float = NMS_FLOOR_SIZE,
    samples:         float = 6.0,
    center:          bool  = True,
    timestamp_start: int   = 1766251924,
    floor_thresh:    float = 0.70,
):
    # ── Load ────────────────────────────────────────────────────────
    verts, faces = target_mesh.vertices, target_mesh.faces

    if verts.size == 0:
        raise RuntimeError("No vertices found – is the file empty or corrupt?")
    if faces.size == 0:
        raise RuntimeError("No faces found – the file has vertices but no polygons.")

    # ── Pre-process ─────────────────────────────────────────────────

    if center:
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        verts  = verts - (lo + hi) / 2.0

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    dims     = bbox_max - bbox_min

    print("─" * 60)
    print(f"  Vertices   : {len(verts):,}")
    print(f"  Faces      : {len(faces):,}")
    print(f"  Extent     : {dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f}  NMS units")
    print(f"  Voxel size : {voxel_size}  ({voxel_size / NMS_FLOOR_SIZE:.2f} floor tiles)")
    print("─" * 60)

    # ── Sample + voxelise ───────────────────────────────────────────
    print("  Sampling surface …")
    positions, normals_s = sample_surface(verts, faces, samples)
    print(f"  Surface samples : {len(positions):,}")

    voxels = voxelise(positions, normals_s, voxel_size)
    print(f"  Voxel cells     : {len(voxels):,}")

    # Count by type
    counts = defaultdict(int)
    for n in voxels.values():
        counts[classify(n, floor_thresh)] += 1
    print(f"  → floors   : {counts['floor']:,}")
    print(f"  → walls    : {counts['wall']:,}")
    print(f"  → ceilings : {counts['ceiling']:,}")

    # ── Build JSON ──────────────────────────────────────────────────
    parts = []
    for t, ((ix, iy, iz), normal) in enumerate(sorted(voxels.items())):
        kind           = classify(normal, floor_thresh)
        obj_id, udata = PART_CATALOGUE[kind]
        pos            = [float(ix * voxel_size),
                          float(iy * voxel_size),
                          float(iz * voxel_size)]
        up, at         = normal_to_up_at(normal)

        parts.append({
            "Timestamp": timestamp_start + t,
            "ObjectID":  obj_id,
            "UserData":  udata,
            "Position":  pos,
            "Up":        up,
            "At":        at,
        })

    # ── Write ───────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(parts, fh, indent=2)

    print("─" * 60)
    print(f"  Written {len(parts):,} parts → {out}")
    print("─" * 60)
    return parts
