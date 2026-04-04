from __future__ import annotations
from functions import get_mesh_dimensions, fbx_to_trimesh, _build_frame, convert_mesh
from nms_exporter import write_nms_json


import struct
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
import trimesh

T_FLOOR = "models/timber_structures/T_FLOOR.fbx"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
# Decimal places for vertex-key hashing.  1e-4 = 0.1 mm at 1 unit = 1 mm.
# Increase to 5 if your model uses very fine geometry (< 0.01 mm features).
_MERGE_PREC: int = 4
 
 
# ---------------------------------------------------------------------------
# 1. STL Parser
# ---------------------------------------------------------------------------
 
def _parse_stl(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """Parse a binary or ASCII STL file.
 
    Returns
    -------
    list of (3, 3) float64 arrays — one per triangle [v0, v1, v2].
    """
 
    triangles: list[np.ndarray] = []
    
    # Extract triangles directly from the trimesh object
    for face in mesh.faces:
        tri = mesh.vertices[face]
        triangles.append(np.array(tri, dtype=np.float64))
    
    if not triangles:
        raise ValueError("No triangles found in the mesh.")
    return triangles
 
 
# ---------------------------------------------------------------------------
# 2. Geometry helpers
# ---------------------------------------------------------------------------
 
def _vtkey(v: np.ndarray) -> tuple:
    return tuple(round(float(x), _MERGE_PREC) for x in v)
 
 
def _tri_normal(tri: np.ndarray) -> np.ndarray:
    """Unit normal of a triangle, or a zero vector for degenerate ones."""
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    length = np.linalg.norm(n)
    return n / length if length > 1e-10 else np.zeros(3)
 
 

# ---------------------------------------------------------------------------
# 3. Connected planar-patch detection (BFS)
# ---------------------------------------------------------------------------
 
def _group_planar_patches(
    triangles: list[np.ndarray],
    normal_dot_tol: float,
    dist_tol: float,
) -> list[dict]:
    """
    BFS on the triangle adjacency graph to cluster triangles into connected,
    coplanar patches.
 
    Two adjacent triangles are merged into the same patch when:
      |dot(n_i, n_j)| > normal_dot_tol   (nearly parallel normals)
      |d_i − d_j|     < dist_tol         (nearly the same plane offset)
 
    Parameters
    ----------
    normal_dot_tol : cosine threshold (default ≈ 0.998, i.e. < 3.6° apart)
    dist_tol       : maximum plane-offset difference in model units
 
    Returns
    -------
    list of {'normal': ndarray(3), 'indices': list[int]}
    """
    n_tris = len(triangles)
    normals   = np.array([_tri_normal(t) for t in triangles])   # (N, 3)
    centroids = np.array([t.mean(axis=0) for t in triangles])   # (N, 3)
    ds = (normals * centroids).sum(axis=1)                       # signed plane offsets
 
    # Build triangle adjacency via shared (directed → undirected) edges
    edge_map: defaultdict[tuple, list[int]] = defaultdict(list)
    for i, tri in enumerate(triangles):
        for j in range(3):
            a = _vtkey(tri[j])
            b = _vtkey(tri[(j + 1) % 3])
            edge_map[tuple(sorted([a, b]))].append(i)
 
    adj: list[list[int]] = [[] for _ in range(n_tris)]
    for nbrs in edge_map.values():
        if len(nbrs) == 2:
            a, b = nbrs
            adj[a].append(b)
            adj[b].append(a)
 
    visited = np.zeros(n_tris, dtype=bool)
    patches: list[dict] = []
 
    for start in range(n_tris):
        if visited[start]:
            continue
        n0 = normals[start]
        if np.linalg.norm(n0) < 1e-10:
            visited[start] = True
            continue
 
        d0 = ds[start]
        stack = [start]
        visited[start] = True
        component: list[int] = []
 
        while stack:
            cur = stack.pop()
            component.append(cur)
            for nb in adj[cur]:
                if visited[nb]:
                    continue
                n1 = normals[nb]
                if (
                    np.linalg.norm(n1) > 1e-10
                    and abs(np.dot(n0, n1)) > normal_dot_tol
                    and abs(ds[nb] - d0) < dist_tol
                ):
                    visited[nb] = True
                    stack.append(nb)
 
        # Average normal for the patch (weighted by valid members)
        comp_normals = normals[component]
        avg_n = comp_normals.mean(axis=0)
        nlen  = np.linalg.norm(avg_n)
        avg_n = avg_n / nlen if nlen > 1e-10 else n0
 
        patches.append({"normal": avg_n, "indices": component})
 
    return patches
 
 
# ---------------------------------------------------------------------------
# 4. Vectorised point-in-triangle test (2-D)
# ---------------------------------------------------------------------------
 
def _inside_any_tri(
    centers: np.ndarray,    # (N, 2)
    tris_2d: np.ndarray,    # (M, 3, 2)
    chunk: int = 512,
) -> np.ndarray:            # (N,) bool
    """
    Returns True for each center that lies inside at least one triangle.
    Processed in triangle-batches to limit peak memory use.
    """
    N      = len(centers)
    result = np.zeros(N, dtype=bool)
    P      = centers[np.newaxis, :, :]   # (1, N, 2)
 
    for start in range(0, len(tris_2d), chunk):
        batch = tris_2d[start : start + chunk]   # (B, 3, 2)
        A  = batch[:, 0:1, :]
        B_ = batch[:, 1:2, :]
        C  = batch[:, 2:3, :]
 
        def _sign(p, a, b):
            # Cross-product sign — works for both CW and CCW winding.
            return (
                (p[..., 0] - b[..., 0]) * (a[..., 1] - b[..., 1])
                - (a[..., 0] - b[..., 0]) * (p[..., 1] - b[..., 1])
            )
 
        d1 = _sign(P, A, B_)   # (B, N)
        d2 = _sign(P, B_, C)
        d3 = _sign(P, C,  A)
 
        # Point is inside if all signs are the same (all ≥ 0 or all ≤ 0)
        in_batch = (
            ~((d1 < 0) | (d2 < 0) | (d3 < 0))
            | ~((d1 > 0) | (d2 > 0) | (d3 > 0))
        )
        result |= in_batch.any(axis=0)
 
    return result
 
 
# ---------------------------------------------------------------------------
# 5. Per-patch square tessellation
# ---------------------------------------------------------------------------
 
def _patch_to_squares(
    patch_tris: list[np.ndarray],
    normal: np.ndarray,
    grid_size: float,
) -> list[list[np.ndarray]]:
    """
    Project a planar patch onto its plane and tile it with a world-aligned
    square grid.  Return the 3-D corners of every cell whose centre lands
    inside at least one projected triangle.
 
    World-alignment
    ---------------
    The grid lines are positioned so that the 3-D coordinate of each grid
    vertex is a multiple of `grid_size` along both in-plane world axes.
    Concretely:
        dot(grid_vertex, u) = k * grid_size    for some integer k
        dot(grid_vertex, v) = l * grid_size    for some integer l
 
    Because both u and v are determined purely by the patch's normal
    (via _build_frame), two patches that share an edge will produce the
    same u/v directions along that edge and therefore place grid lines at
    the same world coordinates — giving coincident boundary vertices that
    the OBJ writer merges.
    """
    u, v = _build_frame(normal)
 
    all_verts = np.vstack(patch_tris)   # (3*M, 3)
    P0        = all_verts.mean(axis=0)  # stable origin on the plane
 
    def to_2d(pts: np.ndarray) -> np.ndarray:
        d = pts - P0
        return np.column_stack([d @ u, d @ v])
 
    tris_2d = np.array([to_2d(t) for t in patch_tris])  # (M, 3, 2)
    flat    = tris_2d.reshape(-1, 2)
    lo, hi  = flat.min(axis=0), flat.max(axis=0)
 
    gs = grid_size
 
    # Align grid to world coordinates.
    # We want dot(P0 + s*u, u) = dot(P0,u) + s to be a multiple of gs.
    # So grid lines in local 2-D are at: s = k*gs - (dot(P0,u) % gs).
    off_u = np.dot(P0, u) % gs
    off_v = np.dot(P0, v) % gs
 
    x0 = np.floor((lo[0] + off_u) / gs) * gs - off_u
    y0 = np.floor((lo[1] + off_v) / gs) * gs - off_v
 
    # Grid edge positions (N+1 values); cell origins are xs[:-1], ys[:-1]
    xs = np.arange(x0, hi[0] + gs, gs)
    ys = np.arange(y0, hi[1] + gs, gs)
 
    cx = xs[:-1] + gs * 0.5
    cy = ys[:-1] + gs * 0.5
    ni, nj = len(cx), len(cy)
 
    if ni == 0 or nj == 0:
        return []
 
    # Build all cell centres and test which lie inside the patch
    gx, gy  = np.meshgrid(cx, cy, indexing="ij")          # (ni, nj)
    centers = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (ni*nj, 2)
    mask = _inside_any_tri(centers, tris_2d)  # (ni*nj,) bool
    mask2d = mask.reshape(ni, nj)

    squares: list[list[np.ndarray]] = []
    rects = _mask_to_rectangles(mask2d)

    for i0, i1, j0, j1 in rects:
        squares.extend(_rectangle_to_squares(i0, i1, j0, j1, xs, ys, gs, P0, u, v))

    return squares
 
def _row_spans(row: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open [start, end) spans of True cells in one row."""
    spans = []
    j = 0
    n = len(row)
    while j < n:
        if not row[j]:
            j += 1
            continue
        j0 = j
        while j < n and row[j]:
            j += 1
        spans.append((j0, j))
    return spans


def _mask_to_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Merge consecutive rows with identical True spans into rectangles.

    Returns rectangles as (i0, i1, j0, j1), using half-open index ranges.
    """
    rects = []
    active: dict[tuple[int, int], int] = {}  # span -> row_start

    for i, row in enumerate(mask):
        spans = _row_spans(row)
        span_set = set(spans)

        new_active: dict[tuple[int, int], int] = {}
        for span in spans:
            if span in active:
                new_active[span] = active[span]
            else:
                new_active[span] = i

        for span, i0 in active.items():
            if span not in span_set:
                j0, j1 = span
                rects.append((i0, i, j0, j1))

        active = new_active

    for span, i0 in active.items():
        j0, j1 = span
        rects.append((i0, len(mask), j0, j1))

    return rects


def _rectangle_to_squares(
    i0: int,
    i1: int,
    j0: int,
    j1: int,
    xs: np.ndarray,
    ys: np.ndarray,
    gs: float,
    P0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> list[list[np.ndarray]]:
    """
    Tile one axis-aligned rectangle of grid cells into the fewest large
    square faces using a greedy square-packing pass.
    """
    x = float(xs[i0])
    y = float(ys[j0])
    w = float((i1 - i0) * gs)
    h = float((j1 - j0) * gs)

    squares: list[list[np.ndarray]] = []

    while w > 1e-12 and h > 1e-12:
        s = min(w, h)
        squares.append([
            P0 + x        * u + y        * v,
            P0 + (x + s)  * u + y        * v,
            P0 + (x + s)  * u + (y + s)  * v,
            P0 + x        * u + (y + s)  * v,
        ])

        if w > h:
            x += s
            w -= s
        else:
            y += s
            h -= s

    return squares
 
# ---------------------------------------------------------------------------
# 6. Automatic grid-size estimation
# ---------------------------------------------------------------------------
 
def _auto_grid_size(triangles: list[np.ndarray]) -> float:
    """Median triangle edge length — matches the natural resolution of the mesh."""
    lengths = [
        np.linalg.norm(tri[(j + 1) % 3] - tri[j])
        for tri in triangles
        for j in range(3)
    ]
    return float(np.median(lengths)) if lengths else 1.0
 
 
# ---------------------------------------------------------------------------
# 7. Public API
# ---------------------------------------------------------------------------
 
def stl_to_squares(
    target_mesh: trimesh.Trimesh,
    grid_size: float | None = None,
    normal_dot_tol: float = 0.998,
    dist_tol: float | None = None,
) -> tuple[list[list[float]], list[list[int]]]:
    """
    Convert an STL file to a *connected* OBJ mesh of perfect square faces.
 
    Every output face is an exact square:
      - all four sides equal in length,
      - all four interior angles 90°,
      - area equal to ``grid_size ** 2``.
 
    Connectivity guarantee
    ----------------------
    Within each planar patch every pair of grid-adjacent squares shares
    exactly two OBJ vertex indices — no floating disconnected faces.
    Squares from *different* patches whose boundary vertices coincide in
    3-D space are also connected via global vertex merging in the OBJ writer.
 
    Parameters
    ----------
    stl_path : str
        Path to the input STL file (binary or ASCII).
    obj_path : str
        Path for the output OBJ file.
    grid_size : float or None
        Side length of each output square, in model units.
        ``None`` (default) uses the median edge length of the input mesh,
        giving roughly one square per original input triangle.
        Use a larger value for a coarser output, smaller for finer.
    normal_dot_tol : float
        Cosine similarity threshold for merging adjacent triangles into the
        same planar patch.  Default 0.998 ≈ 3.6°.  Lower to ~0.99 for
        meshes with slightly noisy or faceted normals.
    dist_tol : float or None
        Maximum plane-offset difference (in model units) for two triangles
        to be considered coplanar.  ``None`` auto-sets to 0.1 % of the
        bounding-box diagonal (a sensible default for most CAD models).
 
    Returns
    -------
    tuple of (vertices, faces)
        vertices: list of [x, y, z] coordinates
        faces: list of [i0, i1, i2, i3] quad indices (NOT triangulated)
    
    Notes
    -----
    Curved surfaces are decomposed into many small single-triangle patches.
    Each such patch still becomes one perfect square cell, so the density
    tracks the original mesh but the curvature is approximated as flat
    squares — identical behaviour to the original version, but now properly
    connected within each planar region.
    """
    triangles = _parse_stl(target_mesh)
    print(f"[stl_to_squares] input     : {len(triangles):,} triangles  ← {target_mesh}")
 
    gs = grid_size if grid_size is not None else _auto_grid_size(triangles)
    print(f"[stl_to_squares] grid size : {gs:.6g}")
 
    if dist_tol is None:
        all_verts = np.vstack(triangles)
        bbox_diag = float(np.linalg.norm(
            all_verts.max(axis=0) - all_verts.min(axis=0)
        ))
        dist_tol = max(bbox_diag * 0.001, gs * 0.1)
    print(f"[stl_to_squares] dist tol  : {dist_tol:.4g}")
 
    patches = _group_planar_patches(triangles, normal_dot_tol, dist_tol)
    print(f"[stl_to_squares] patches   : {len(patches):,}")
 
    all_squares: list[list[np.ndarray]] = []
    for patch in patches:
        tris = [triangles[i] for i in patch["indices"]]
        all_squares.extend(_patch_to_squares(tris, patch["normal"], gs))
 
    print(f"[stl_to_squares] squares   : {len(all_squares):,}")
    # Convert squares to vertex/face lists (keep quads, don't triangulate)
    vertices = []
    faces = []
    vertex_map = {}
    
    for square in all_squares:
        square_indices = []
        for vertex in square:
            key = _vtkey(vertex)
            if key not in vertex_map:
                vertex_map[key] = len(vertices)
                vertices.append(vertex.tolist() if isinstance(vertex, np.ndarray) else list(vertex))
            square_indices.append(vertex_map[key])
        faces.append(square_indices)
    
    print("[stl_to_squares] done.")
    return vertices, faces


def load_squares(verts_faces: tuple[list[list[float]], list[list[int]]]):
    """Return quad vertices and faces directly (already in the right format)."""
    verts, faces = verts_faces
    return verts, faces


def T_FLOOR_mesh_to_json(
    target_mesh: trimesh.Trimesh,
    out_path: str,
    grid_size: float | None = None,
    normal_dot_tol: float = 0.998,
    dist_tol: float | None = None
) -> None:
    floor_dim = get_mesh_dimensions(fbx_to_trimesh(T_FLOOR))["max_dim"]
    target_mesh = convert_mesh(target_mesh, "stl")

    if grid_size is None:
        triangles = _parse_stl(target_mesh)
        all_verts = np.vstack(triangles)
        bbox = all_verts.max(axis=0) - all_verts.min(axis=0)
        stl_face_size = float(np.max(bbox))          # largest dimension of the STL
        grid_size = stl_face_size / round(stl_face_size / (floor_dim / 100))

    verts, faces = load_squares(stl_to_squares(target_mesh, grid_size, normal_dot_tol, dist_tol))
    squares = [np.array([verts[i] for i in face]) for face in faces]
    
    out = []
    for sq in squares:
        v0, v1, v2, v3 = sq
        center = (v0 + v1 + v2 + v3) / 4
        edge1 = v1 - v0
        edge2 = v3 - v0
        normal = np.cross(edge1, edge2)
        normal /= np.linalg.norm(normal) + 1e-10  # avoid division by zero
        size = np.linalg.norm(edge1)
        
        scale = size/floor_dim*100
        out.append((center, normal, scale))
    
    # Replace the entire placed_parts building loop:
    placed_parts = []

    class PartInfo:
        def __init__(self, object_id):
            self.object_id = object_id

    t_floor_part_info = PartInfo("^T_FLOOR")

    for center, normal, scale_factor in out:
        normal_hat = np.array(normal) / (np.linalg.norm(normal) + 1e-10)

        # Compute an in-plane vector perpendicular to normal
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(normal_hat, arbitrary)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])

        in_plane_vec = np.cross(normal_hat, arbitrary)
        in_plane_vec /= np.linalg.norm(in_plane_vec) + 1e-10

        # At = surface normal pointing outward
        at_vec = normal_hat
        # Up = in-plane vector scaled by scale_factor (lies on face)
        # (json_to_stl extracts scale as np.linalg.norm(up))
        up_vec = in_plane_vec * scale_factor

        placed_parts.append({
            "part_info": t_floor_part_info,
            "position": center.tolist() if isinstance(center, np.ndarray) else list(center),
            "up": up_vec.tolist(),   # ← surface normal (scaled) = NMS "Up"
            "at": at_vec.tolist()    # ← in-plane forward vector  = NMS "At"
        })
        
    write_nms_json(placed_parts, out_path)
