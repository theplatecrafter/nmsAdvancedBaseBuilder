from quick_fit import T_FLOOR_mesh_to_json
from depth_fit import depth_fit_nms_to_json
from json_to_stl import json_to_stl
from functions import meshify, simplify_mesh, meshfile


stl_path = "mesh/cow.obj"
stl = simplify_mesh(meshify(stl_path), 100)
out = "out/test"
fbx = ["models/timber_structures/T_FLOOR.fbx","models/timber_structures/T_TRIFLOOR.fbx","models/timber_structures/T_WALLDIAGONAL.fbx"]

T_FLOOR_mesh_to_json(stl,out + "_T_FLOOR.json")
depth_fit_nms_to_json(stl,fbx,out + "_DEPTH_FIT.json",1)


json_to_stl(out + "_T_FLOOR.json", "models", out +"_T_FLOOR.stl")
json_to_stl(out + "_DEPTH_FIT.json", "models", out +"_DEPTH_FIT.stl")
meshfile(stl, out + "_SIMPLIFIED.obj")