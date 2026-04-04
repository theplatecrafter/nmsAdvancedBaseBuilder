import subprocess
import trimesh
from pathlib import Path
import numpy as np
import os
import tempfile




def meshify(path: str,scale: float = 1.0,rotation: np.ndarray = None,translation: np.ndarray = None) -> trimesh.Trimesh:
    """
    Loads a mesh from the given path, which can be STL,  FBX, or OBJ.
    For FBX, it uses assimp to convert to OBJ first.
    """
    ext = Path(path).suffix.lower()
    if ext == ".stl":
        loaded_mesh = trimesh.load(path, force="mesh")
    elif ext == ".fbx":
        loaded_mesh = fbx_to_trimesh(path)
    elif ext == ".obj":
        loaded_mesh = trimesh.load(path, force="mesh")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Apply transformations if provided
    if rotation is not None:
        loaded_mesh.apply_transform(trimesh.transformations.rotation_matrix(rotation[0], [1, 0, 0]))
        loaded_mesh.apply_transform(trimesh.transformations.rotation_matrix(rotation[1], [0, 1, 0]))
        loaded_mesh.apply_transform(trimesh.transformations.rotation_matrix(rotation[2], [0, 0, 1]))

    if translation is not None:
        loaded_mesh.apply_translation(translation)

    if scale != 1.0:
        loaded_mesh.apply_scale(scale)

    return loaded_mesh

def meshfile(mesh:trimesh.Trimesh, path: str):
    ext = Path(path).suffix.lower()
    if ext == ".stl":
        mesh.export(path)
    elif ext == ".obj":
        mesh.export(path)
    else:
        raise ValueError(f"Unsupported file extension for export: {ext}")

def simplify_mesh(mesh: trimesh.Trimesh, target_face_count: int) -> trimesh.Trimesh:
    if mesh.faces.shape[0] <= target_face_count:
        return mesh

    # 1. Define the path to the compiled binary
    # Make sure this points to the ACTUAL executable file
    tool_path = Path(__file__).parent / "tools/Fast-Quadric-Mesh-Simplification/src.cmd/simplify"
    
    if not tool_path.exists():
        raise FileNotFoundError(f"Simplification tool not found at {tool_path}")

    # 2. Use temporary files to communicate with the C++ tool
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, "input.obj")
        output_path = os.path.join(tmp_dir, "output.obj")

        # Export current mesh to OBJ for the tool to read
        mesh.export(input_path)

        # 3. Calculate reduction fraction (the tool expects a ratio, e.g., 0.1)
        # Note: Check if your version of 'simplify' takes target count or ratio.
        # Most versions of this specific tool take a ratio (0.0 to 1.0).
        ratio = target_face_count / mesh.faces.shape[0]

        # 4. Execute the binary using subprocess
        # Command format: ./simplify input.obj output.obj ratio
        cmd = [str(tool_path), input_path, output_path, str(ratio)]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            # 5. Load the simplified mesh back into trimesh
            simplified_mesh = trimesh.load(output_path)
            print(f"Simplification complete: {mesh.faces.shape[0]} → {simplified_mesh.faces.shape[0]} faces")
            return simplified_mesh
        except subprocess.CalledProcessError as e:
            print(f"Error during simplification: {e.stderr}")
            raise


def _build_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane axes (u, v) for the plane with the given normal."""
    n = normal / np.linalg.norm(normal)
    arb = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(n, arb)) > 0.85:
        arb = np.array([1.0, 0.0, 0.0])
    u = arb - np.dot(arb, n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v
 

def get_mesh_dimensions(mesh: trimesh.Trimesh):
    """
    Returns the width, height, and depth of a trimesh object.
    """
    # 1. Validation
    if mesh is None or len(mesh.vertices) == 0:
        return {"x": 0, "y": 0, "z": 0}

    # 2. Get the extents (the absolute size in X, Y, and Z)
    # extents returns a numpy array: [size_x, size_y, size_z]
    size_x, size_y, size_z = mesh.extents

    # 3. Return as a clean dictionary for easy access
    return {
        "width_x": size_x,
        "height_y": size_y,
        "depth_z": size_z,
        "max_dim": max(mesh.extents),
        "volume": mesh.volume if mesh.is_watertight else "N/A"
    }


def fbx_to_trimesh(fbx_path: str) -> trimesh.Trimesh:
    """
    Uses the system 'assimp' command to convert FBX to OBJ,
    then loads it into Trimesh.
    """
    import tempfile

    fbx_path = str(Path(fbx_path).resolve())

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_obj = os.path.join(tmpdir, "converted.obj")

        result = subprocess.run(
            ["assimp", "export", fbx_path, temp_obj],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"assimp failed (code {result.returncode}):\n"
                f"{result.stderr.decode(errors='replace')}"
            )

        # assimp sometimes writes e.g. "converted.obj.obj" — find whatever it made
        candidates = list(Path(tmpdir).glob("*.obj"))
        if not candidates:
            raise FileNotFoundError(
                f"assimp ran but produced no OBJ in {tmpdir}.\n"
                f"stdout: {result.stdout.decode(errors='replace')}\n"
                f"stderr: {result.stderr.decode(errors='replace')}"
            )

        mesh = trimesh.load(str(candidates[0]), force="mesh")

    return mesh.dump(concatenate=True) if isinstance(mesh, trimesh.Scene) else mesh


def convert_mesh(
    mesh: trimesh.Trimesh,
    target_format: str
) -> trimesh.Trimesh:
    """
    Convert a trimesh.Trimesh object to a different format and back.
    
    This is useful for converting mesh geometry between different representations.
    The mesh is exported to a temporary file in the target format, then reimported.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh object
    target_format : str
        Target format (lowercase, no dot). Options:
        - 'stl': STL (binary)
        - 'obj': Wavefront OBJ
        - 'ply': Stanford PLY
        - 'gltf': GL Transmission Format (text)
        - 'glb': GL Transmission Format (binary)
        - 'dae': COLLADA
        - '3mf': 3D Manufacturing Format
        - 'off': OFF format
        - 'step': STEP CAD format
    
    Returns
    -------
    trimesh.Trimesh
        The converted mesh object
    
    Raises
    ------
    ValueError
        If the target format is not supported
    """
    supported_formats = ['stl', 'obj', 'ply', 'gltf', 'glb', 'dae', '3mf', 'off', 'step']
    
    target_format = target_format.lower().lstrip('.')
    
    if target_format not in supported_formats:
        raise ValueError(
            f"Unsupported mesh format: {target_format}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export to temporary file
        temp_path = os.path.join(tmpdir, f"converted.{target_format}")
        mesh.export(temp_path)
        
        # Reimport and return
        converted_mesh = trimesh.load(temp_path, force="mesh")
        
        # Handle Scene objects (merge into single mesh if needed)
        if isinstance(converted_mesh, trimesh.Scene):
            converted_mesh = converted_mesh.dump(concatenate=True)
        
        return converted_mesh