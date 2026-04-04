"""
core/json_to_stl.py
===================
Converts NMS base JSON output into an STL file by loading all the placed parts,
applying transformations (position, rotation, scale), and merging them.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import trimesh
    TRIMESH_OK = True
except ImportError:
    TRIMESH_OK = False

try:
    from pyassimp import load as assimp_load
    ASSIMP_OK = True
except ImportError:
    ASSIMP_OK = False


def _load_mesh(fbx_path: Path, log: Callable = print) -> "trimesh.Trimesh | None":
    """Load an FBX/OBJ file into a Trimesh, using pyassimp for FBX files."""
    if not TRIMESH_OK:
        log(f"[mesh] trimesh not available")
        return None
    
    fbx_path = Path(fbx_path)
    
    # Handle FBX files with pyassimp
    if fbx_path.suffix.lower() == ".fbx":
        if not ASSIMP_OK:
            log(f"[mesh] Assimp not available for {fbx_path.name}")
            return None
        
        try:
            with assimp_load(str(fbx_path)) as scene:
                meshes = []
                
                def extract_meshes(node):
                    # node.meshes contains direct mesh objects
                    for mesh in node.meshes:
                        if len(mesh.vertices) > 0:
                            vertices = np.array(mesh.vertices)
                            faces = np.array(mesh.faces) if len(mesh.faces) > 0 else None
                            
                            if faces is not None:
                                trimesh_obj = trimesh.Trimesh(
                                    vertices=vertices,
                                    faces=faces,
                                    process=False
                                )
                                meshes.append(trimesh_obj)
                    
                    for child in node.children:
                        extract_meshes(child)
                
                extract_meshes(scene.rootnode)
                
                if meshes:
                    result = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
                    log(f"[mesh] Loaded {fbx_path.name}: {len(result.vertices)} vertices, {len(result.faces)} faces")
                    return result
                else:
                    log(f"[mesh] No valid geometry in {fbx_path.name}")
                    return None
                
        except Exception as e:
            log(f"[mesh] Error loading {fbx_path.name}: {type(e).__name__}: {e}")
            return None
    
    # Handle other formats with trimesh
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = trimesh.load(str(fbx_path), force="mesh")
        
        if isinstance(loaded, trimesh.Scene):
            meshes = [g for g in loaded.geometry.values() if len(g.vertices) > 0]
            if meshes:
                loaded = trimesh.util.concatenate(meshes)
            else:
                log(f"[mesh] Scene has no valid geometries in {fbx_path.name}")
                return None
        
        if isinstance(loaded, trimesh.Trimesh):
            log(f"[mesh] Loaded {fbx_path.name}: {len(loaded.vertices)} vertices, {len(loaded.faces)} faces")
            return loaded
        else:
            log(f"[mesh] {fbx_path.name} is not a valid Trimesh (type: {type(loaded).__name__})")
            return None
    except Exception as e:
        log(f"[mesh] Error loading {fbx_path.name}: {type(e).__name__}: {e}")
        return None


def _build_rotation_matrix(up: np.ndarray, at: np.ndarray) -> np.ndarray:
    """
    Build a 3×3 rotation matrix from up and at vectors.
    
    up: direction vector (may be scaled)
    at: forward direction vector (normalized)
    
    The three basis vectors form:
    - Z axis (up): normalized up vector
    - X axis (right): perpendicular to up and at
    - Y axis (forward): at vector
    """
    # Normalize up vector (remove scale)
    up_norm = up / np.linalg.norm(up)
    at_norm = at / np.linalg.norm(at)
    
    # Compute right vector (cross product)
    right = np.cross(at_norm, up_norm)
    right = right / np.linalg.norm(right)
    
    # Recompute at to ensure orthogonality
    at_corrected = np.cross(up_norm, right)
    
    # Build rotation matrix (columns are basis vectors)
    R = np.column_stack([right, at_corrected, up_norm])
    return R


def json_to_stl(
    json_path: str | Path,
    models_dir: str | Path,
    output_path: str | Path,
    log: Callable = print,
) -> Path:
    """
    Convert an NMS base JSON file to STL by loading and transforming parts.
    
    Parameters
    ----------
    json_path   : path to the output JSON file from the fitter
    models_dir  : path to the models folder containing FBX files
    output_path : path where the STL file will be written
    log         : logging callback
    
    Returns
    -------
    Path to the output STL file
    """
    if not TRIMESH_OK:
        raise RuntimeError("trimesh not installed. Run: pip install trimesh[all]")
    
    json_path = Path(json_path)
    models_dir = Path(models_dir)
    output_path = Path(output_path)
    
    log(f"[json_to_stl] Reading {json_path.name}…")
    
    with open(json_path, "r") as f:
        objects = json.load(f)
    
    log(f"[json_to_stl] {len(objects)} objects to process")
    
    # Build a cache of loaded meshes
    mesh_cache = {}
    combined_meshes = []
    skipped = 0
    
    for obj_idx, obj in enumerate(objects):
        obj_id = obj.get("ObjectID", "UNKNOWN")
        position = np.array(obj.get("Position", [0, 0, 0]))
        up_raw = np.array(obj.get("Up", [0, 1, 0]))
        at_raw = np.array(obj.get("At", [1, 0, 0]))
        
        # Extract scale from up vector magnitude
        scale = np.linalg.norm(up_raw)
        if scale < 1e-8:
            scale = 1.0
        
        # Skip BASE_FLAG (it's just an anchor point)
        if obj_id == "^BASE_FLAG":
            continue
        
        # Load the mesh if not cached
        if obj_id not in mesh_cache:
            # Try to find the mesh file - search for multiple formats
            base_name = obj_id.lstrip("^")
            mesh = None
            fbx_path = None
            
            # Try different extensions
            for ext in [".fbx", ".obj", ".glb", ".gltf"]:
                candidate_name = base_name + ext
                for candidate in models_dir.rglob(candidate_name):
                    fbx_path = candidate
                    mesh = _load_mesh(fbx_path, log=log)
                    if mesh is not None:
                        break
                
                if mesh is not None:
                    break
            
            if mesh is None:
                log(f"[json_to_stl] ⚠ {obj_id}: Could not load mesh (tried {base_name}.*)")
                skipped += 1
                continue
            
            mesh_cache[obj_id] = mesh
        else:
            mesh = mesh_cache[obj_id]
        
        # Clone the mesh for transformation
        mesh_copy = mesh.copy()
        
        # Apply scale
        mesh_copy.apply_scale(scale)
        
        # Apply rotation
        R = _build_rotation_matrix(up_raw, at_raw)
        mesh_copy.apply_transform(
            np.vstack([
                np.column_stack([R, position]),
                [0, 0, 0, 1]
            ])
        )
        
        combined_meshes.append(mesh_copy)
    
    if len(combined_meshes) == 0:
        raise ValueError("No parts could be loaded from the JSON file")
    
    log(f"[json_to_stl] Combining {len(combined_meshes)} meshes…")
    final_mesh = trimesh.util.concatenate(combined_meshes)
    
    # Write to STL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_mesh.export(str(output_path))
    
    log(f"[json_to_stl] Written → {output_path}")
    if skipped > 0:
        log(f"[json_to_stl] ⚠ {skipped} parts skipped (mesh load failed)")
    
    return output_path


def add_to_app_ui(app_instance):
    """
    Add a 'JSON → STL' tab to the app notebook.
    Call this from app.py to add the visualization tool.
    """
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter import filedialog, messagebox
    
    class JsonToStlTool(ttk.Frame):
        def __init__(self, parent: ttk.Notebook, models_dir: Path, **kwargs):
            super().__init__(parent, **kwargs)
            self.columnconfigure(0, weight=1)
            self.models_dir = models_dir
            
            self._build_ui()
        
        def _build_ui(self):
            pad = {"padx": 10, "pady": 5}
            
            # Input JSON
            inp = ttk.LabelFrame(self, text=" Input JSON ", padding=10)
            inp.grid(row=0, column=0, sticky="ew", **pad)
            inp.columnconfigure(1, weight=1)
            
            ttk.Label(inp, text="JSON File:").grid(row=0, column=0, sticky="w")
            self._json_var = tk.StringVar()
            ttk.Entry(inp, textvariable=self._json_var).grid(
                row=0, column=1, sticky="ew", padx=(6, 4))
            ttk.Button(inp, text="Browse…", width=9,
                       command=self._browse_json).grid(row=0, column=2)
            
            # Output STL
            out = ttk.LabelFrame(self, text=" Output STL ", padding=10)
            out.grid(row=1, column=0, sticky="ew", **pad)
            out.columnconfigure(1, weight=1)
            
            ttk.Label(out, text="STL File:").grid(row=0, column=0, sticky="w")
            self._out_var = tk.StringVar()
            ttk.Entry(out, textvariable=self._out_var).grid(
                row=0, column=1, sticky="ew", padx=(6, 4))
            ttk.Button(out, text="Browse…", width=9,
                       command=self._browse_output).grid(row=0, column=2)
            
            # Convert button
            ttk.Button(self, text="▶   Convert to STL",
                       command=self._convert).grid(
                row=2, column=0, sticky="ew", padx=10, pady=(8, 4))
            
            # Log
            self.rowconfigure(3, weight=1)
            log_frame = ttk.LabelFrame(self, text=" Log ", padding=5)
            log_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=(2, 10))
            log_frame.columnconfigure(0, weight=1)
            log_frame.rowconfigure(0, weight=1)
            
            self._log_box = tk.Text(log_frame, height=8, state="disabled",
                                    font=("Courier", 9), wrap="none")
            log_sb = ttk.Scrollbar(log_frame, orient="vertical",
                                   command=self._log_box.yview)
            self._log_box.configure(yscrollcommand=log_sb.set)
            self._log_box.grid(row=0, column=0, sticky="nsew")
            log_sb.grid(row=0, column=1, sticky="ns")
        
        def _log(self, msg: str):
            self._log_box.configure(state="normal")
            self._log_box.insert(tk.END, msg + "\n")
            self._log_box.see(tk.END)
            self._log_box.configure(state="disabled")
        
        def _browse_json(self):
            path = filedialog.askopenfilename(
                title="Select JSON file",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if path:
                self._json_var.set(path)
        
        def _browse_output(self):
            path = filedialog.asksaveasfilename(
                title="Save STL as",
                defaultextension=".stl",
                filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
            )
            if path:
                self._out_var.set(path)
        
        def _convert(self):
            json_path = self._json_var.get()
            out_path = self._out_var.get()
            
            if not json_path or not out_path:
                messagebox.showerror("Missing input", "Please specify both JSON and output paths")
                return
            
            try:
                json_to_stl(json_path, self.models_dir, out_path, log=self._log)
                messagebox.showinfo("Success", f"STL written to:\n{out_path}")
            except Exception as e:
                self._log(f"ERROR: {e}")
                messagebox.showerror("Error", str(e))
    
    return JsonToStlTool
