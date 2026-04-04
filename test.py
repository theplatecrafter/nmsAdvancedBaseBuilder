from quick_fit import T_FLOOR_mesh_to_json
from depth_fit import depth_fit_nms_to_json
from json_to_stl import json_to_stl
from functions import meshify, simplify_mesh, meshfile


stl_path = "mesh/simplify_simplify_drogon_whole_without_base.stl"
stl = meshify(stl_path)
#stl = simplify_mesh(stl, 200)
out = "out/test"
fbx = ["models/timber_structures/T_FLOOR.fbx","models/timber_structures/T_TRIFLOOR.fbx","models/timber_structures/T_WALLDIAGONAL.fbx"]

T_FLOOR_mesh_to_json(stl,out + "_T_FLOOR.json")
json_to_stl(out + "_T_FLOOR.json", "models", out +"_T_FLOOR.stl")

depth_fit_nms_to_json(stl,fbx,out + "_DEPTH.json",1)
json_to_stl(out + "_DEPTH.json", "models", out +"_DEPTH.stl")




meshfile(stl, out + "_SIMPLIFIED.obj")