# NMS Base Builder

Convert any 3D mesh (STL / OBJ) into a No Man's Sky base JSON using
real NMS base-part geometries as building blocks.

---

## Setup

### 1. System dependency (for FBX support)
```bash
sudo apt-get install libassimp-dev python3-pyassimp
```

If your part models are already in OBJ, GLB, or GLTF format you can
skip this step — trimesh loads those natively.

### 2. Python dependencies (inside your venv)
```bash
pip install -r requirements.txt
```

### 3. Place your part models
Drop all your NMS base-part FBX (or OBJ/GLB) files into the `models/`
folder.  The filename stem becomes the NMS ObjectID:

```
models/
  BUILDABLE_FLOOR.fbx    →   "^BUILDABLE_FLOOR"
  BUILDABLE_WALL.fbx     →   "^BUILDABLE_WALL"
  U_PARAGON.fbx          →   "^U_PARAGON"
  ...
```

### 4. Run
```bash
python app.py
```

---

## How the algorithm works

### Step 1 — Part geometry extraction
Each FBX file is loaded and its **bounding box** is computed.
The **smallest dimension** becomes the part's *face axis*
(the direction it presses against a surface):

```
Panel lying flat   →  thin in Y  →  face axis = [0, 1, 0]
Wall panel         →  thin in Z  →  face axis = [0, 0, 1]
```

### Step 2 — Input mesh voxelisation
The STL is voxelised at the chosen *voxel size* (default 4.0 NMS units).
Only the **surface shell** is kept (voxels with at least one empty face-
neighbour).  The nearest-face normal from the original mesh is recorded
for each surface voxel so the fitter knows which way "outward" is.

### Step 3 — Greedy set-cover fitting

```
while uncovered voxels remain AND placed < max_parts:
    seed = first uncovered voxel
    for each selected part:
        for each of the part's 3 axis orientations (±X, ±Y, ±Z):
            R = rotation that aligns this axis with seed's normal
            candidates = nearby uncovered voxels within reach radius
            in_footprint = candidates whose rotated local coords
                           fall inside the part's 2D footprint rectangle
            score = count(in_footprint)
                  − curvature_penalty  (if enabled)
    place best (part, orientation)
    mark all in_footprint as covered
```

**Curvature penalty:** when the surface normals inside a part's
footprint vary a lot (high curvature), large parts are penalised,
making the algorithm prefer smaller, better-fitting parts in those
regions — useful for tight curves like a dragon's neck or wing tips.

### Step 4 — NMS JSON export
Every placed part becomes one object in the NMS base JSON array with:
- `Position` = world-space centre of the placed voxel
- `Up`        = outward surface normal (= the part's face direction)
- `At`        = a consistent forward vector in the surface plane

---

## Tuning guide

| Problem | Try |
|---------|-----|
| Too many parts (>19 k) | Increase voxel size (e.g. 6.0 or 8.0) |
| Shape too blocky | Decrease voxel size (e.g. 2.0) |
| Model too small/large | Adjust Scale (e.g. 0.05 for mm-source STL) |
| Lots of uncovered voxels | Add more / different parts, or raise max_parts |
| Curved areas look bad | Enable curvature penalty, add smaller part variants |

---

## Importing into NMS

1. Open **NMS Save Editor** (github.com/goatfungus/NMSSaveEditor)
2. Load your save file
3. Navigate to your target base → **BaseObject** array
4. Replace / merge with the generated JSON array
5. Save and load in-game

---

## File structure

```
nms_builder/
├── app.py                  Main UI
├── requirements.txt
├── README.md
├── models/                 ← put your FBX/OBJ part files here
└── core/
    ├── parts_loader.py     FBX → PartInfo geometry extraction
    ├── mesh_processor.py   STL → voxelised surface shell
    ├── surface_fitter.py   Greedy set-cover fitting algorithm
    └── nms_exporter.py     NMS JSON builder + writer
```