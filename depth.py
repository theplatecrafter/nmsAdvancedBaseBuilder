from __future__ import annotations
from functions import get_mesh_dimensions, fbx_to_trimesh, _build_frame, convert_mesh
from exporter import write_nms_json


from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import scale, rotate, translate
from scipy.optimize import minimize
import trimesh




# ---------------------------------------------------------------------------
# depth_fit helpers
# ---------------------------------------------------------------------------

def _extract_largest_cross_section(fbx_path: str) -> tuple[Polygon, int]:
    """
    Load FBX and return its largest cross-sectional Shapely Polygon and the axis it's perpendicular to.
    Projects all faces onto each of the 3 axis-aligned planes, unions them,
    and returns whichever projection has the greatest area, along with the axis index.
    
    Returns
    -------
    tuple of (polygon, drop_axis)
        polygon: Shapely Polygon of the largest cross-section
        drop_axis: which axis was dropped (0=X, 1=Y, 2=Z), i.e., the normal direction
    """
    from shapely.ops import unary_union

    mesh = fbx_to_trimesh(fbx_path)

    best_poly = None
    best_area = 0.0
    best_drop_axis = 0

    for drop_axis in range(3):          # try dropping X, Y, Z
        keep = [i for i in range(3) if i != drop_axis]
        polys = []
        for face in mesh.faces:
            coords = mesh.vertices[face][:, keep]
            p = Polygon(coords)
            if p.is_valid and p.area > 1e-10:
                polys.append(p)
        if not polys:
            continue
        combined = unary_union(polys)
        if combined.area > best_area:
            best_area = combined.area
            best_poly = combined
            best_drop_axis = drop_axis

    if best_poly is None:
        raise ValueError(f"Could not extract any cross section from {fbx_path}")

    # Centre it at the origin so the optimizer starts clean
    c = best_poly.centroid
    centered_poly = translate(best_poly, xoff=-c.x, yoff=-c.y)
    return centered_poly, best_drop_axis


def _face_to_shapely(face_verts: np.ndarray,
                     u: np.ndarray,
                     v: np.ndarray,
                     P0: np.ndarray) -> Polygon:
    """Project 3-D face vertices onto the local (u, v) plane → Shapely Polygon."""
    d = face_verts - P0
    coords_2d = np.column_stack([d @ u, d @ v])
    poly = Polygon(coords_2d)
    return poly if poly.is_valid else poly.buffer(0)


def _rotation_matrix_from_axis_to_vec(from_axis: int, to_vec: np.ndarray) -> np.ndarray:
    """
    Compute a rotation matrix that rotates the given axis direction to align with to_vec.
    
    Parameters
    ----------
    from_axis : int
        Which axis to rotate (0=X, 1=Y, 2=Z)
    to_vec : np.ndarray
        Target direction (should be unit vector)
    
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    # Source direction (the axis we're starting from)
    from_vec = np.zeros(3)
    from_vec[from_axis] = 1.0
    
    # Normalize target
    to_vec = to_vec / (np.linalg.norm(to_vec) + 1e-10)
    
    # If already aligned, return identity
    if np.abs(np.dot(from_vec, to_vec)) > 0.9999:
        return np.eye(3)
    
    # Otherwise use Rodrigues' rotation formula
    axis = np.cross(from_vec, to_vec)
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    cos_angle = np.dot(from_vec, to_vec)
    sin_angle = np.sqrt(1 - cos_angle**2)
    
    # Skew-symmetric cross-product matrix of axis
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    # Rodrigues formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
    R = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
    return R


def _fit_poly_in_poly(bounding_poly: Polygon,
                      fitting_poly: Polygon):
    """
    Optimise scale + rotation + translation so that fitting_poly covers as much
    of bounding_poly as possible while staying inside it.

    Returns (score, s, theta_deg, tx, ty)
      score = area of intersection (higher = better)
    """
    bound_c = bounding_poly.centroid

    def objective(params):
        s, theta, tx, ty = params
        if s <= 0:
            return 1e9
        temp = scale(fitting_poly, xfact=s, yfact=s, origin='centroid')
        temp = rotate(temp, theta, origin='centroid', use_radians=False)
        temp = translate(temp, xoff=tx, yoff=ty)
        intersection = bounding_poly.intersection(temp)
        covered = intersection.area
        overflow = temp.area - covered          # area outside bounding_poly
        return -covered + overflow * 10.0       # penalise overflow heavily

    best_res = None
    best_val = float('inf')

    # Try a few starting scales so the optimizer doesn't get stuck
    for s0 in [0.8, 0.4, 0.15]:
        x0 = [s0, 0.0, bound_c.x, bound_c.y]
        res = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6},
        )
        if res.fun < best_val:
            best_val = res.fun
            best_res = res

    s, theta, tx, ty = best_res.x
    if s <= 0:
        return 0.0, s, theta, tx, ty

    temp = scale(fitting_poly, xfact=s, yfact=s, origin='centroid')
    temp = rotate(temp, theta, origin='centroid', use_radians=False)
    temp = translate(temp, xoff=tx, yoff=ty)
    score = bounding_poly.intersection(temp).area
    return score, s, theta, tx, ty


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert(
    target_mesh: trimesh.Trimesh,
    part_fbx_paths: list[str],
    output_path: str,
    depth: int = 1,
) -> None:
    """
    For every face in the target mesh, greedily pack NMS parts (up to `depth`
    times) by fitting each part's largest cross-section into the remaining
    empty area of the face, always picking the part that covers the most area.

    Parameters
    ----------
    mesh_path       : path to the target STL or OBJ file
    part_fbx_paths  : list of FBX file paths; filename stem → ObjectID
                      e.g. "models/timber_structures/T_FLOOR.fbx" → "^T_FLOOR"
    depth           : how many parts to place per face
    out_path        : path for the output NMS JSON file
    """
    from shapely.ops import unary_union

    # ------------------------------------------------------------------ #
    # 1. Load target mesh
    # ------------------------------------------------------------------ #
    print(f"[depth_fit] Loading mesh: {target_mesh} …")
    target = target_mesh
    print(f"[depth_fit] {len(target.faces)} faces, {len(target.vertices)} vertices")

    # ------------------------------------------------------------------ #
    # 2. Pre-load every part: cross-section polygon + metadata
    # ------------------------------------------------------------------ #
    class PartInfo:
        def __init__(self, object_id: str):
            self.object_id = object_id

    parts = []   # list of dicts

    for fbx_path in part_fbx_paths:
        obj_id   = "^" + Path(fbx_path).stem
        print(f"[depth_fit] Loading part {obj_id} …")

        fbx_mesh     = fbx_to_trimesh(fbx_path)
        part_max_dim = get_mesh_dimensions(fbx_mesh)["max_dim"]
        cross_poly, drop_axis = _extract_largest_cross_section(fbx_path)

        minx, miny, maxx, maxy = cross_poly.bounds
        cross_size = max(maxx - minx, maxy - miny)   # natural 2-D size

        parts.append({
            "part_info":    PartInfo(obj_id),
            "cross_poly":   cross_poly,
            "part_max_dim": part_max_dim,
            "cross_size":   cross_size,
            "drop_axis":    drop_axis,  # which axis is the flat side perpendicular to
        })
        print(f"[depth_fit]   cross_area={cross_poly.area:.4f}  "
              f"cross_size={cross_size:.4f}  part_max_dim={part_max_dim:.4f}  axis={drop_axis}")

    # ------------------------------------------------------------------ #
    # 3. Iterate over every face
    # ------------------------------------------------------------------ #
    placed_parts = []

    for face_idx, face in enumerate(target.faces):
        face_verts = target.vertices[face]          # (3, 3)

        # Local coordinate frame for this face
        normal = target.face_normals[face_idx]
        if np.linalg.norm(normal) < 1e-10:
            continue
        normal = normal / np.linalg.norm(normal)
        u, v   = _build_frame(normal)
        P0     = face_verts.mean(axis=0)

        # Project face to 2-D
        remaining = _face_to_shapely(face_verts, u, v, P0)
        if remaining.is_empty or remaining.area < 1e-10:
            continue

        # -------------------------------------------------------------- #
        # 4. Depth loop: pick best part, place it, subtract, repeat
        # -------------------------------------------------------------- #
        for d in range(depth):
            if remaining.is_empty or remaining.area < 1e-10:
                break

            best_score        = 0.0
            best_params       = None
            best_part         = None
            best_placed_poly  = None

            for part in parts:
                score, s, theta, tx, ty = _fit_poly_in_poly(
                    remaining, part["cross_poly"]
                )
                if score > best_score:
                    best_score  = score
                    best_params = (s, theta, tx, ty)
                    best_part   = part

                    # Reconstruct the placed polygon for later use
                    tmp = scale(part["cross_poly"],
                                xfact=s, yfact=s, origin='centroid')
                    tmp = rotate(tmp, theta, origin='centroid',
                                 use_radians=False)
                    tmp = translate(tmp, xoff=tx, yoff=ty)
                    best_placed_poly = tmp

            if best_part is None or best_score < 1e-10:
                break   # nothing fits any more

            s, theta, tx, ty = best_params

            # ---------------------------------------------------------- #
            # 5. Compute 3-D placement vectors for NMS JSON
            # ---------------------------------------------------------- #
            cx2d = best_placed_poly.centroid.x
            cy2d = best_placed_poly.centroid.y
            position_3d = P0 + cx2d * u + cy2d * v

            # Scale: s scales cross_poly (cross_size units) in STL space.
            # Normalise to FBX model size then apply game-unit factor (*100).
            scale_factor = (s * best_part["cross_size"]
                            / best_part["part_max_dim"] * 100)

            # Get rotation matrix to align part's natural orientation to face normal
            drop_axis = best_part["drop_axis"]
            rot_matrix = _rotation_matrix_from_axis_to_vec(drop_axis, normal)

            # At = surface normal pointing outward
            at_vec = normal

            # Up = in-plane forward (rotated by theta), scaled by scale_factor, 
            # then further rotated to align with the part's orientation
            theta_rad = np.radians(theta)
            up_vec_base = np.cos(theta_rad) * u + np.sin(theta_rad) * v
            up_vec = (rot_matrix @ up_vec_base) * scale_factor

            placed_parts.append({
                "part_info": best_part["part_info"],
                "position":  position_3d.tolist(),
                "up":        up_vec.tolist(),
                "at":        at_vec.tolist(),
            })

            # Subtract placed shape from remaining empty space
            remaining = remaining.difference(best_placed_poly)
            if not remaining.is_valid:
                remaining = remaining.buffer(0)

        if (face_idx + 1) % 50 == 0 or face_idx == 0:
            print(f"[depth_fit] Face {face_idx + 1}/{len(target.faces)} done, "
                  f"total placed so far: {len(placed_parts)}")

    print(f"[depth_fit] Finished. Total placed parts: {len(placed_parts)}")
    write_nms_json(placed_parts, output_path)