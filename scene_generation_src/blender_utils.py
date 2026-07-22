import hashlib
import bpy, platform
import sys
import os
import json
import math, bisect
from math import radians, cos, sin
from math import fabs as absf
from mathutils import Vector, Euler, Matrix
import random
from bpy_extras.object_utils import world_to_camera_view
from pathlib import Path

# ==== Logging utility ====
MY_LOG_PATH = os.path.join(os.environ.get("BENCHMARKING_DATA_CACHE", "."), "blender_utils.log")
def myprint(*args, **kwargs):
    os.makedirs(os.path.dirname(MY_LOG_PATH), exist_ok=True)
    with open(MY_LOG_PATH, "a", encoding="utf-8") as f:
        print(*args, **kwargs, file=f)

def load_records_from_jsonl(mapping_path: str) -> list[dict]:
    """Load mapping.jsonl (one JSON per line). Return list of records."""
    if not os.path.isfile(mapping_path):
        print(f"ERROR: mapping.jsonl not found: {mapping_path}")
        return []
    records = []
    with open(mapping_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
                records.append(rec)
            except Exception:
                print(f"WARN: JSON parse failed at line {ln}")
                continue
    return records

def _default_scene_config_path() -> str:
    candidates = []
    explicit = os.environ.get("BENCHMARKING_SCENE_CONFIG")
    if explicit:
        candidates.append(explicit)
    script_dir = os.environ.get("BENCHMARKING_SCRIPT_FOLDER")
    if script_dir:
        candidates.extend([
            os.path.join(script_dir, "scene_generation_src/config/scene_config.json"),
            os.path.join(script_dir, "config/scene_config.json"),
        ])
    package_root = Path(__file__).resolve().parent
    repo_root = package_root.parent
    candidates.extend([
        str(package_root / "config/scene_config.json"),
        str(repo_root / "scene_generation_src/config/scene_config.json"),
        str(repo_root / "config/scene_config.json"),
    ])
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError("No scene_config.json found. Pass --config or set BENCHMARKING_SCENE_CONFIG.")


def _copy_section(flat: dict, section: dict | None, keys: list[str]) -> None:
    if not section:
        return
    for key in keys:
        if key in section:
            flat[key] = section[key]


def _asset_entry_to_flat_config(flat: dict, assets: dict, source_key: str, path_key: str, option_key: str) -> None:
    entry = assets.get(source_key)
    if not entry:
        return
    if isinstance(entry, str):
        flat[path_key] = entry
        return
    flat[path_key] = entry.get("path", "")
    if "filters" in entry:
        flat[option_key] = entry["filters"]


def _normalize_scene_config(scene_type: str, data):
    if isinstance(data, list):
        running_config = None
        path_config = None
        for obj in data:
            if obj.get("scene_type") == scene_type:
                running_config = obj
            elif obj.get("scene_type") == "asset_path":
                path_config = obj
        if running_config is None:
            raise ValueError(f"No valid config '{scene_type}' found in scene_config.json")
        return running_config, path_config or {}

    if not isinstance(data, dict):
        raise ValueError("scene_config.json must be either a list or an object.")
    scenes = data.get("scenes", {})
    if scene_type not in scenes:
        raise ValueError(f"No valid config '{scene_type}' found in scene_config.json")

    scene = scenes[scene_type]
    flat = {"scene_type": scene_type}
    flat.update(data.get("defaults", {}))
    if scene.get("folder_name"):
        flat["folder_name"] = scene["folder_name"]

    assets = scene.get("assets", {})
    _asset_entry_to_flat_config(flat, assets, "A", "A_item_path", "A_object_options")
    _asset_entry_to_flat_config(flat, assets, "B", "B_item_path", "B_object_options")
    _asset_entry_to_flat_config(flat, assets, "C", "C_item_path", "C_object_options")
    _asset_entry_to_flat_config(flat, assets, "ground_material", "GMAT_path", "ground_options")
    _asset_entry_to_flat_config(flat, assets, "fence", "fence_path", "fence_options")
    _asset_entry_to_flat_config(flat, assets, "anchor", "anchor_path", "anchor_options")
    _asset_entry_to_flat_config(flat, assets, "outdoor_art", "outdoor_path", "outdoor_options")
    _asset_entry_to_flat_config(flat, assets, "sky", "sky_path", "sky_options")
    _asset_entry_to_flat_config(flat, assets, "surrounding", "surrounding_path", "surrounding_options")

    _copy_section(flat, scene.get("placement"), [
        "RANDOM_PLACING_AB", "A_ON_B_LAYOUT", "linear_layout", "only_active_a",
        "KEEP_CENTER", "KEEP_CENTER_VALUE", "GRID_N", "N_ASSET_A", "N_ASSET_B",
        "N_ASSET_C", "K_FENCE", "K_OUTDOORART", "K_ANCHOR", "four_angles_deg",
        "host_class",
    ])
    _copy_section(flat, scene.get("environment"), [
        "ground_size", "FENCE_CLEARANCE", "OUTDOORART_CLEARANCE", "OUTDOOR_FIRST",
        "FOCUS_ON_OUTDOORART", "EXCLUDE_NAMES", "BUILDING_FENCE_SIDE_OPTION",
        "BUILDING_OUTDOOR_ART_SIDE_OPTION", "surrounding_clearance",
        "surrounding_margin_between", "surrounding_facing_center",
    ])
    _copy_section(flat, scene.get("camera"), [
        "fit_coverage", "crop_image_fit_coverage", "normal_cam_deg", "crop_cam_deg",
        "VIEW_FROM_CENTER", "CENTERVIEW_HEIGHT", "VIEW_INDICES",
    ])
    _copy_section(flat, scene.get("generation"), [
        "NUMBER_OF_SCENES_TO_GENERATE", "VIEWS_PER_SCENE", "SEED_BASE",
        "RENDER_SAMPLES", "RENDER_RESOLUTION",
    ])
    return flat, data.get("asset_paths", {})


def loading_config(scene_type, path_return=False, config_path: str | None = None):
    config_dir = config_path or _default_scene_config_path()
    with open(config_dir, "r", encoding="utf-8") as f:
        data = json.load(f)
    running_config, path_config = _normalize_scene_config(scene_type, data)
    
    if path_return:
        return running_config, path_config
    return running_config

# ====device selection====
def enable_best_cycles_device(scene: bpy.types.Scene, prefer: str = "AUTO", verbose: bool = True):
    """
    Auto-pick and enable the best Cycles compute backend + GPU devices.
    Returns (device_mode, backend, enabled_devices) e.g. ("GPU", "METAL", ["Apple M2 Max"])
    Fallbacks to CPU if no GPU is available.
    """
    # Only relevant for Cycles
    try:
        if scene.render.engine != 'CYCLES':
            return ("N/A", "N/A", [])
    except Exception:
        pass

    prefs = bpy.context.preferences
    if 'cycles' not in prefs.addons:
        if verbose: myprint("[cycles] addon not found, using CPU")
        scene.cycles.device = 'CPU'
        return ("CPU", "NONE", [])

    cprefs = prefs.addons['cycles'].preferences

    # Backend priority by OS
    sys = platform.system()
    order_by_os = {
        "Darwin":  ["METAL"],                      # macOS (Apple/AMD) -> Metal
        "Windows": ["OPTIX", "CUDA", "HIP", "ONEAPI"],
        "Linux":   ["OPTIX", "CUDA", "HIP", "ONEAPI"],
    }
    order = order_by_os.get(sys, ["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"])

    # If user requested a specific backend, try it first
    if prefer and prefer != "AUTO":
        order = [prefer] + [b for b in order if b != prefer]

    used_backend = None
    used_devices = []

    for backend in order:
        try:
            cprefs.compute_device_type = backend     # select backend
            cprefs.get_devices()                     # refresh device list
        except Exception:
            continue

        # enable all non-CPU devices on this backend; disable CPU
        gpu_devices = [d for d in getattr(cprefs, "devices", []) if getattr(d, "type", "") != 'CPU']
        if not gpu_devices:
            continue

        for d in cprefs.devices:
            d.use = (d in gpu_devices)               # only enable GPUs

        scene.cycles.device = 'GPU'
        used_backend = backend
        used_devices = [f"{d.name} ({d.type})" for d in gpu_devices]
        break

    if not used_backend:
        scene.cycles.device = 'CPU'
        if verbose: myprint("[cycles] No GPU devices found, using CPU.")
        return ("CPU", "NONE", [])

    if verbose:
        myprint(f"[cycles] Using GPU via {used_backend}: {', '.join(used_devices)}")

    return ("GPU", used_backend, used_devices)



# ------------------------------
# Utility: scene reset & helpers
# ------------------------------
def purge_scene():
    # delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

    # remove orphan data blocks (run a few times)
    for _ in range(3):
        try:
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        except Exception:
            pass

    # ensure a world exists
    if not bpy.data.worlds:
        w = bpy.data.worlds.new("World")
    else:
        w = bpy.data.worlds[0]
    bpy.context.scene.world = w


def new_camera(name="PreviewCam"):
    cam_data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam_data.lens = 50  # mm
    cam_data.clip_start = 0.01
    cam_data.clip_end = 10000
    return cam

def add_light(
    name,
    type_="AREA",               # "SUN" | "AREA" | "POINT" | "SPOT"
    energy=1000.0,             # For SUN this is intensity, for others it's power
    location=(0, 0, 0),
    rotation=None,             # Optional Euler rotation (rx, ry, rz) in radians
    size=1.0,                  # AREA side length; POINT/SPOT shadow softness
    shape='SQUARE',            # AREA: 'SQUARE' | 'RECTANGLE' | 'DISK' | 'ELLIPSE'
    angle_deg=0.5,             # SUN shadow angle (degrees)
    spot_size_deg=45.0,        # SPOT cone angle (degrees)
    spot_blend=0.15,           # SPOT edge softness
    color=None                 # (r,g,b) 0..1
):
    # Create a new light datablock
    light_data = bpy.data.lights.new(name=name, type=type_.upper())
    light_data.energy = energy

    t = type_.upper()
    if t == "AREA":
        # Area light: rectangular or square emitter
        light_data.shape = shape
        light_data.size = size
        # For RECTANGLE you can also set light_data.size_y separately
    elif t == "SUN":
        # Sun light: directional, no falloff with distance
        light_data.angle = math.radians(angle_deg)
    elif t == "POINT":
        # Point light: isotropic falloff
        light_data.shadow_soft_size = size
    elif t == "SPOT":
        # Spot light: cone-shaped emitter
        light_data.shadow_soft_size = size
        light_data.spot_size = math.radians(spot_size_deg)
        light_data.spot_blend = spot_blend

    if color is not None:
        # Set RGB color (convert to 4D internally)
        light_data.color = color

    # Wrap into an Object so it can be placed in the scene
    light_obj = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = Vector(location)
    if rotation is not None:
        light_obj.rotation_euler = rotation

    return light_obj

def add_area_light(name, location, power=200.0, size=1.0):
    return add_light(name, type_='AREA', energy=power, location=location, size=size, shape='SQUARE')

def add_sun_light(name, location, rotation, power=5.0, angle_deg=0.5):
    return add_light(name, type_="SUN", energy=power, location=location, angle_deg=angle_deg, rotation=rotation)


def aim_at(obj, target):
    direction = (Vector(target) - obj.location).normalized()
    # Camera looks down -Z, Y is up in Blender
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()


#  old method for getting all mesh objects
def all_mesh_objects():
    return [o for o in bpy.context.scene.objects if o.type == 'MESH']

# new method for getting all renderable objects
# def all_renderable_objects(include_empty_parents=False):
#     """
#     Return all mesh-like objects in the scene.
#     If include_empty_parents=True, also include Empties that have mesh descendants.
#     """
#     objs = []
#     for o in bpy.context.scene.objects:
#         if o.type in {"MESH", "CURVE", "SURFACE", "META"}:
#             objs.append(o)
#         elif include_empty_parents and o.type == "EMPTY":
#             # Check if this empty has mesh descendants
#             has_mesh_child = any(
#                 c.type in {"MESH", "CURVE", "SURFACE", "META"}
#                 for c in bpy.data.objects
#                 if c.parent == o
#             )
#             if has_mesh_child:
#                 objs.append(o)
#     return objs

def all_renderable_objects(include_empty_parents=True, include_collection_instances=True):
    """
    Return all mesh-like objects in the scene.
    If include_empty_parents=True, also include Empties that have mesh descendants.
    If include_collection_instances=True, also include Empties that instance Collections.
    """
    res = []
    for o in bpy.context.scene.objects:
        if o.type in {"MESH", "CURVE", "SURFACE", "META"}:
            res.append(o)
            continue
        if include_collection_instances and o.type == "EMPTY" \
           and getattr(o, "instance_type", None) == "COLLECTION" \
           and getattr(o, "instance_collection", None):
            res.append(o)
            continue
        if include_empty_parents and o.type == "EMPTY":
            has_mesh_child = any(
                c.type in {"MESH", "CURVE", "SURFACE", "META"}
                for c in bpy.context.scene.objects if c.parent == o
            )
            if has_mesh_child:
                res.append(o)
    return res

def get_bounds(objs):
    if not objs:
        return (Vector((0, 0, 0)), Vector((1, 1, 1)))
    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    for o in objs:
        try:
            for c in o.bound_box:
                v = o.matrix_world @ Vector(c)
                mins.x = min(mins.x, v.x)
                mins.y = min(mins.y, v.y)
                mins.z = min(mins.z, v.z)
                maxs.x = max(maxs.x, v.x)
                maxs.y = max(maxs.y, v.y)
                maxs.z = max(maxs.z, v.z)
        except Exception:
            pass
    size = maxs - mins
    center = mins + size * 0.5
    return center, size

#  Advanced bounds that consider evaluated geometry, instances, etc.
def get_visual_bounds(objs, include_children=True):
    """
    Compute world-space bounds of what you actually *see*:
    - Uses evaluated depsgraph (modifiers/geometry nodes applied).
    - Converts CURVE/SURFACE/META to mesh for accurate bounds.
    - Expands COLLECTION instances (Empty with instance_collection).
    - Optionally walks scene parent/child hierarchy.

    Returns (center: Vector, size: Vector).
    """
    RENDERABLE_TYPES = {"MESH", "CURVE", "SURFACE", "META"}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    INF = 1e30
    mins = Vector((+INF, +INF, +INF))
    maxs = Vector((-INF, -INF, -INF))
    any_point = False
    visited = set()  # avoid double-counting exact same datablock

    def _accumulate_minmax(p):
        nonlocal mins, maxs, any_point
        mins.x = min(mins.x, p.x); mins.y = min(mins.y, p.y); mins.z = min(mins.z, p.z)
        maxs.x = max(maxs.x, p.x); maxs.y = max(maxs.y, p.y); maxs.z = max(maxs.z, p.z)
        any_point = True

    def _add_object(o, inherited_mat=None):
        # prevent revisiting exact same object instance
        if o.as_pointer() in visited:
            return
        visited.add(o.as_pointer())

        o_eval = o.evaluated_get(depsgraph)

        # Parent transform chain handling for embedded collection members
        mat = (inherited_mat @ o.matrix_world) if inherited_mat is not None else o_eval.matrix_world

        t = getattr(o_eval, "type", None)

        # Case 1: Real geometry types
        if t in RENDERABLE_TYPES:
            # Prefer converting to evaluated mesh for accurate (mods/GN) bounds
            try:
                me = o_eval.to_mesh()
            except Exception:
                me = None

            if me is not None and getattr(me, "vertices", None):
                for v in me.vertices:
                    _accumulate_minmax(mat @ v.co)
                o_eval.to_mesh_clear()
            else:
                # Fallback to evaluated bound_box if mesh conversion failed
                if getattr(o_eval, "bound_box", None):
                    for c in o_eval.bound_box:
                        _accumulate_minmax(mat @ Vector(c))

        # Case 2: Empty that instances a Collection (BlenderKit common)
        elif t == "EMPTY" and getattr(o_eval, "instance_type", None) == 'COLLECTION' and o_eval.instance_collection:
            coll = o_eval.instance_collection
            for child in coll.objects:
                # child.matrix_world is relative to the collection; compose with the instance transform
                _add_object(child, inherited_mat=mat)

        # Optional: also walk scene parenting chain (your root -> children structure)
        if include_children:
            for child in (c for c in bpy.data.objects if c.parent == o):
                _add_object(child, inherited_mat=None)  # child has its own matrix_world

    for o in objs:
        _add_object(o)

    if not any_point:
        # Nothing measurable; return zero-sized bounds
        return Vector((0, 0, 0)), Vector((0, 0, 0))

    size = maxs - mins
    center = mins + 0.5 * size
    return center, size


# ------------------------------
# Focused object AABB update utility
# ------------------------------
def _aabb_from_center_size_after_z_rot(center: Vector, size: Vector, deg: float):
    """
    Given an object's center and un-rotated size (extent), compute its world AABB
    after rotating the object about its own center around +Z by `deg` degrees.
    Returns (bmin, bmax). # 0, 1, 2 are x,y,z
    """
    sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

    # half extents before rotation
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5

    # rotate around Z: only XY extents change
    t = radians(deg)
    ct, st = cos(t), sin(t)

    # AABB half extents after rotation (axis-aligned)
    hx_p = absf(hx * ct) + absf(hy * st)
    hy_p = absf(hx * st) + absf(hy * ct)
    hz_p = hz  # unchanged

    bmin = Vector((cx - hx_p, cy - hy_p, cz - hz_p))
    bmax = Vector((cx + hx_p, cy + hy_p, cz + hz_p))
    return bmin, bmax

def _aabb_from_center_size(center: Vector, size: Vector):
    """Axis-aligned AABB from center/size without any rotation."""
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    return center - Vector((hx, hy, hz)), center + Vector((hx, hy, hz))

def _merge_aabb(bmin1: Vector, bmax1: Vector, bmin2: Vector, bmax2: Vector):
    """Union of two AABBs. 0, 1, 2 are x,y,z"""
    bmin = Vector((
        min(bmin1[0], bmin2[0]),
        min(bmin1[1], bmin2[1]),
        min(bmin1[2], bmin2[2]),
    ))
    bmax = Vector((
        max(bmax1[0], bmax2[0]),
        max(bmax1[1], bmax2[1]),
        max(bmax1[2], bmax2[2]),
    ))
    return bmin, bmax

def update_focused_object_AABB(current: dict, new: dict, deg: float) -> dict:
    """
    current: {"center": Vector, "size": Vector}  # existing scene AABB (center/size)
    new:     {"center": Vector, "size": Vector}  # object center/size BEFORE rotation
    deg:     float, rotation around Z (degrees), applied about the object's center

    Return updated {"center": Vector, "size": Vector} for the *scene* AABB.
    """
    # AABB for the incoming (possibly rotated) object
    new_bmin, new_bmax = _aabb_from_center_size_after_z_rot(new["center"], new["size"], deg)

    # If current is empty (size == (0,0,0)), initialize with new
    # 0, 1, 2 are x,y,z
    cur_size = current["size"]
    if (absf(cur_size[0]) + absf(cur_size[1]) + absf(cur_size[2])) == 0.0:
        merged_bmin, merged_bmax = new_bmin, new_bmax
    else:
        # Current scene AABB corners
        cur_bmin, cur_bmax = _aabb_from_center_size(current["center"], current["size"])
        merged_bmin, merged_bmax = _merge_aabb(cur_bmin, cur_bmax, new_bmin, new_bmax)

    merged_center = (merged_bmin + merged_bmax) * 0.5
    merged_size   = (merged_bmax - merged_bmin)
    return {"center": merged_center, "size": merged_size}

# ------------------------------
# Asset loading
# ------------------------------


def append_materials_from_blend(blend_path):
    mats = []
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        if data_from.materials:
            data_to.materials = list(data_from.materials)
    for m in data_to.materials or []:
        if m:
            mats.append(m)
    return mats

def _ensure_tiling_on_material(mat, tile_x=1.0, tile_y=1.0, use_uv=True):
    """Ensure proper tiling on the material by adding Texture Coordinate and Mapping nodes."""
    if not mat or not getattr(mat, "use_nodes", False):
        return
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    # find or create TexCoord and Mapping nodes
    texcoord = next((n for n in nodes if n.type == 'TEX_COORD'), None)
    if not texcoord:
        texcoord = nodes.new('ShaderNodeTexCoord')
        texcoord.location = (-1200, 0)

    mapping = next((n for n in nodes if n.type == 'MAPPING'), None)
    if not mapping:
        mapping = nodes.new('ShaderNodeMapping')
        mapping.vector_type = 'POINT'
        mapping.location = (-1000, 0)
        src = texcoord.outputs.get('UV') if use_uv and texcoord.outputs.get('UV') else texcoord.outputs.get('Generated')
        if src:
            links.new(src, mapping.inputs['Vector'])

    # Set tiling
    mapping.inputs['Scale'].default_value[0] = float(tile_x)
    mapping.inputs['Scale'].default_value[1] = float(tile_y)
    mapping.inputs['Scale'].default_value[2] = 1.0

    # Connect Mapping to all texture nodes
    tex_like_types = {'TEX_IMAGE','TEX_NOISE','TEX_MUSGRAVE','TEX_VORONOI','TEX_WAVE','TEX_MAGIC','TEX_BRICK'}
    for n in list(nodes):
        if n.type in tex_like_types:
            vec_input = n.inputs.get('Vector')
            if vec_input:
                for l in list(vec_input.links):
                    links.remove(l)
                links.new(mapping.outputs['Vector'], vec_input)
            # Change image texture to Repeat to avoid Clip/Extend stretching
            if n.type == 'TEX_IMAGE' and hasattr(n, 'extension'):
                n.extension = 'REPEAT'



def append_objects_and_collections_from_blend(blend_path):
    objs = []
    cols = []
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        if data_from.objects:
            data_to.objects = list(data_from.objects)
        if hasattr(data_from, 'collections') and data_from.collections:
            data_to.collections = list(data_from.collections)
    # Link objects
    for o in getattr(data_to, 'objects', []) or []:
        if o is None:
            continue
        try:
            bpy.context.collection.objects.link(o)
            objs.append(o)
        except RuntimeError:
            # may already be linked
            objs.append(o)
    # Link collections (and ensure their objects are visible in scene)
    for c in getattr(data_to, 'collections', []) or []:
        if c is None:
            continue
        try:
            bpy.context.scene.collection.children.link(c)
        except RuntimeError:
            pass
        for o in c.objects:
            try:
                if o.name not in bpy.context.scene.objects:
                    bpy.context.collection.objects.link(o)
            except Exception:
                pass
            if o not in objs:
                objs.append(o)
    return objs


# ------------------------------
# Utility: Duplicate hierarchy with linked data
# ------------------------------
def duplicate_hierarchy_linked(root, name_suffix="_DUP"):
    """
    Duplicate 'root' and ALL of its descendants.
    - Keeps children structure
    - Links mesh data (memory-cheap), but creates new Object datablocks
    """
    # original -> duplicate map (optional to keep, handy for debugging)
    mapping = {}

    def _dup(obj, new_parent=None):
        new = obj.copy()
        # link mesh/curve/etc data (EMPTY has data=None, that's fine)
        if getattr(obj, "data", None):
            new.data = obj.data
        bpy.context.scene.collection.objects.link(new)

        # parent to the new parent and preserve local transform
        if new_parent:
            new.parent = new_parent
            new.matrix_parent_inverse = obj.matrix_parent_inverse.copy()

        # copy local transform explicitly (safer across Blender versions)
        new.location = obj.location.copy()
        new.rotation_mode = obj.rotation_mode
        if obj.rotation_mode == 'QUATERNION':
            new.rotation_quaternion = obj.rotation_quaternion.copy()
        elif obj.rotation_mode == 'AXIS_ANGLE':
            new.rotation_axis_angle = obj.rotation_axis_angle[:]  # tuple copy
        else:
            new.rotation_euler = obj.rotation_euler.copy()
        new.scale = obj.scale.copy()

        mapping[obj] = new

        # recurse over children
        for child in (o for o in bpy.data.objects if o.parent == obj):
            _dup(child, new)

        return new

    new_root = _dup(root, None)
    new_root.name = f"{root.name}{name_suffix}"
    return new_root


def Load_locate_collection(blend_file, object_name_in_scene, 
                           location=(0,0,0), collection_name=None,
                           link=False):
    # 1) use libraries.load to link the collection
    with bpy.data.libraries.load(blend_file, link=link) as (data_from, data_to):
        if not data_from.collections:
            myprint("No collections found in file.")
            return None
        target = collection_name or data_from.collections[0]
        data_to.collections = [target]

    linked_coll = data_to.collections[0]

    if not link:
        inst = bpy.data.objects.new(object_name_in_scene, None)
        inst.instance_type = 'COLLECTION'
        inst.instance_collection = linked_coll
        inst.empty_display_type = 'PLAIN_AXES'
        inst.location = location
        bpy.context.scene.collection.objects.link(inst)
        return inst
    else:
        # 2) Create an Empty instance and point it to the collection
        inst = bpy.data.objects.new(object_name_in_scene, None)
        inst.instance_type = 'COLLECTION'
        inst.instance_collection = linked_coll
        inst.empty_display_type = 'PLAIN_AXES'
        inst.location = location

        bpy.context.scene.collection.objects.link(inst)
        return inst



# === Helper: Make camera look at a point ===
def look_at(cam, target):
    direction = target - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()


# === Compute camera height and radius for desired coverage ===
def cam_height_and_radius_for_coverage(r_target: float, halfz_target: float, 
                                       coverage: float, fov_x: float, 
                                       fov_y: float, pitch_deg: float,
                                       min_cam_height: float = 1.8,
                                       margin: float = 1.1):
    """Compute camera height and radius to fit r_target at desired coverage."""
    r_xy = max(r_target * max(coverage, 1e-3) * margin, 1e-6)
    r_z  = max(halfz_target * max(coverage, 1e-3) * margin, 1e-6)
    theta_rad = math.radians(pitch_deg)

    # compute required D to fit r in XY and Z using fov
    D_xy = r_xy / math.tan(0.5 * fov_x)
    D_z  = r_z  / math.tan(0.5 * fov_y)

    # get the larger D_req
    D_req = max(D_xy, D_z)
    radius = max(D_req * math.cos(theta_rad), 2*r_target)
    height = max(D_req * math.sin(theta_rad), min_cam_height)
    return radius, height




# === Place camera above scene, looking roughly at center ===
def place_camera_above_scene(center=Vector((0, 0, 0)),
                              view_index=0,
                              radius=5.0,
                              height=6.0,
                              jitter=0.2,
                              cam_name="Normal",
                              target_offset=Vector((0, 0, 0))):
    """
    Place a camera at a random position above the scene, pointing roughly toward the center.

    - center: The scene center to look at
    - view_index: If >=0, pick one of 4 cardinal views (0-3); if <0, pick random angle
    - radius: Distance from the center (horizontal)
    - height: Height above the ground
    - jitter: Random offset to make the view more natural
    - cam_name: Name of the created camera object
    - target_offset: Offset the target a bit to simulate imperfect framing
    """

    # Create camera
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.name = cam_name
    scene = bpy.context.scene
    scene.camera = cam

    # Random angle around center
    angle = view_index/10 * (math.pi / 2) if view_index/10 >= 0 else random.uniform(0, 2 * math.pi)
    x = center.x + radius * math.cos(angle) + random.uniform(-jitter, jitter)
    y = center.y + radius * math.sin(angle) + random.uniform(-jitter, jitter)
    z = center.z + height + random.uniform(-jitter, jitter)
    cam.location = Vector((x, y, z))

    # Look at (center + offset)
    look_target = center + target_offset
    look_at(cam, look_target)
    cam["target_center"] = look_target
    # Optional: Wider lens to include more content
    cam.data.lens = 35  # or reduce to 24 for ultra wide
    cam.data.sensor_width = 36

    cam.data.clip_start = 0.01
    cam.data.clip_end   = 1000.0

    # debugging
    # myprint(f"Placed camera at {tuple(round(v, 2) for v in cam.location)} looking at {tuple(round(v, 2) for v in look_target)}")
    return cam

# === Place center camera with slight jitter ===
def _look_at_with_pitch_xy(cam, target_xy: Vector, down_deg: float = 10.0):
    """
    let the camera look at target_xy (x,y,0) with a downward pitch of down_deg degrees.
    0 deg = horizontal, +deg = look down
    """
    cam_pos = cam.location
    flat_target = Vector((target_xy.x, target_xy.y, cam_pos.z))   # horizontal plane target point
    forward_xy = (flat_target - cam_pos).normalized()             # horizontal forward vector
    theta = math.radians(down_deg)
    # horizontal forward rotated down by theta
    forward = forward_xy * math.cos(theta) + Vector((0, 0, -1)) * math.sin(theta)
    look_at(cam, cam_pos + forward)
    return cam_pos + forward

def place_center_camera(cam, 
                        height: float,
                        jitter_xy: float = 0.1,
                        cam_name_prefix: str = "CenterCam",
                        AABB: dict = None,
                        view_index: int = 0,
                        down_deg: float = 10.0,):
    """ Place camera at center with slight jitter."""

    jx = random.uniform(-jitter_xy, jitter_xy)
    jy = random.uniform(-jitter_xy, jitter_xy)
    cam.location = Vector((jx, jy, height))

    if AABB is not None:
        c = AABB["center"]; s = AABB["size"]
        cx, cy = float(c.x), float(c.y)
        hx, hy = 0.5*float(s.x), 0.5*float(s.y)
        # ：+Y=front, +X=right, -Y=back, -X=left
        targets_xy = {
            35: Vector((0 - hx, 0 + hy, 0.0)),  # front-left
            30: Vector((0,      0 + hy, 0.0)),  # front  (+Y)
            25: Vector((0 + hx, 0 + hy, 0.0)),  # front-right
            20: Vector((0 + hx, 0,      0.0)),  # right  (+X)
            15: Vector((0 + hx, 0 - hy, 0.0)),  # back-right   (-X)
            10: Vector((0,      0 - hy, 0.0)),  # back   (-Y)
            5:  Vector((0 - hx, 0 - hy, 0.0)),  # back-left
            0:  Vector((0 - hx, 0,      0.0)),  # left   (-X)
        }
        t_xy = targets_xy.get(view_index, targets_xy[0])
    else:
        # if no AABB, just look in cardinal directions from origin
        yaw = {0:0.0, 1:-math.pi/4, 2:-2*math.pi/4, 3:-3*math.pi/4, 
               4:math.pi, 5:3*math.pi/4, 6:math.pi/2, 7:1*math.pi/4}[view_index % 8]
        t_xy = Vector((cam.location.x + math.cos(yaw),
                       cam.location.y + math.sin(yaw),
                       0.0))

    target_center_ = _look_at_with_pitch_xy(cam, t_xy, down_deg=down_deg)

    cam.name = f"{cam_name_prefix}_{view_index}"
    cam["scene_setting"] = f"centerview_{view_index}"
    cam["target_center"] = target_center_
    cam.data.lens = 18


# ------------------------------
# Default models / materials Generation
# ------------------------------
def create_colored_object(obj_type="sphere",
                          location=(0, 0, 0),
                          size=1.0,
                          depth=2.0,
                          color_name="red",
                          metallic_type="non_metal",
                          object_name="MyObject"):
    """
    Creates a basic shape (sphere, cube, cylinder) at given location, with specified
    color, roughness and metallic properties.

    - obj_type: "sphere", "cube", "cylinder"
    - location: (x, y, z)
    - size: overall size or radius
    - depth: height/depth (only used for cylinder)
    - color_name: color from predefined palette
    - metallic_type: "non_metal", "metal"
    - object_name: name assigned to object
    """

    # === Predefined color palette ===
    predefined_colors = {
        "red":     (1.0, 0.0, 0.0),
        "yellow/gold":  (1.0, 1.0, 0.0),
        "green":   (0.0, 1.0, 0.0),
        "blue":    (0.0, 0.0, 1.0),
        "purple":  (0.4, 0.2, 0.6),
        "black":   (0.0, 0.0, 0.0),
        "white/silver":   (1.0, 1.0, 1.0)
    }

    if color_name not in predefined_colors:
        raise ValueError(f"Color '{color_name}' not in predefined palette.")

    color_rgb = predefined_colors[color_name] + (1.0,)  # Add alpha

    # === Roughness and Metallic presets ===
    roughness_presets = {
        "smooth": 0.1,
        "rough": 0.9
    }

    metallic_presets = {
        "non_metal": 0.0,
        "metal": 1.0
    }


    if metallic_type not in metallic_presets:
        raise ValueError(f"Invalid metallic type '{metallic_type}'.")
    
    # Determine roughness based on metallic type
    if metallic_type == "metal":
        roughness_type = "smooth"  # metals are usually smooth
    else:
        roughness_type = "rough"   # non-metals are usually rough

    # === Add the mesh ===
    x, y, z = location
    # === Create Geometry ===
    if obj_type == "cube":
        bpy.ops.mesh.primitive_cube_add(size=size * 2, location=(x, y, z))
    elif obj_type == "uv_sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=(x, y, z))
    elif obj_type == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=size, depth=depth, location=(x, y, z))
    elif obj_type == "cone":
        bpy.ops.mesh.primitive_cone_add(radius1=size, depth=depth, location=(x, y, z))
    elif obj_type == "torus":
        bpy.ops.mesh.primitive_torus_add(major_radius=size, minor_radius=size * 0.3, location=(x, y, z))
    else:
        raise ValueError(f"Unsupported object type: {obj_type}")

    obj = bpy.context.object
    obj.name = object_name

    # === Create Material ===
    mat = bpy.data.materials.new(name=f"Mat_{object_name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color_rgb
        bsdf.inputs["Roughness"].default_value = roughness_presets[roughness_type]
        bsdf.inputs["Metallic"].default_value = metallic_presets[metallic_type]

    obj.data.materials.append(mat)

    # === Attach metadata ===
    obj["color_name"] = color_name
    obj["roughness_type"] = roughness_type
    obj["metallic_type"] = metallic_type
    obj["shape"] = obj_type

    # for debugging
    # print(f"Created {obj_type} '{object_name}' at {location} with color={color_name}, roughness={roughness_type}, metallic={metallic_type}")

    return obj



# --------------------------------------------------------------------
# Calculations for camera placement to cover a scene
# --------------------------------------------------------------------

def camera_fov_xy(cam: bpy.types.Object, scene: bpy.types.Scene):
    """Return (fov_x, fov_y) in radians for current camera data & render aspect."""
    return cam.data.angle_x, cam.data.angle_y
    camd = cam.data
    # Sensor fit & aspect
    resx = scene.render.resolution_x * scene.render.resolution_percentage / 100.0
    resy = scene.render.resolution_y * scene.render.resolution_percentage / 100.0
    aspect = resx / resy if resy else 1.0

    sensor_fit = camd.sensor_fit  # 'AUTO' | 'HORIZONTAL' | 'VERTICAL'
    sensor_w = camd.sensor_width
    sensor_h = camd.sensor_height

    # Decide effective sensor dimension along each axis
    # (match Blender's logic roughly)
    if sensor_fit == 'VERTICAL':
        # vertical fit: fov_y fixed by sensor_h, fov_x follows aspect
        fov_y = 2.0 * math.atan((sensor_h * 0.5) / camd.lens)
        fov_x = 2.0 * math.atan((sensor_h * aspect * 0.5) / camd.lens)
    elif sensor_fit == 'HORIZONTAL':
        # horizontal fit: fov_x fixed by sensor_w, fov_y follows aspect
        fov_x = 2.0 * math.atan((sensor_w * 0.5) / camd.lens)
        fov_y = 2.0 * math.atan((sensor_w / aspect * 0.5) / camd.lens)
    else:
        # AUTO: approximate by aspect; Blender switches based on aspect >= 1
        if aspect >= 1.0:
            fov_x = 2.0 * math.atan((sensor_w * 0.5) / camd.lens)
            fov_y = 2.0 * math.atan((sensor_w / aspect * 0.5) / camd.lens)
        else:
            fov_y = 2.0 * math.atan((sensor_h * 0.5) / camd.lens)
            fov_x = 2.0 * math.atan((sensor_h * aspect * 0.5) / camd.lens)
    return fov_x, fov_y


# def scene_center_and_radius(objs, exclude_names=frozenset()):
#     """Compute (center, radius_xy) from a set of objects, ignoring excluded ones."""
#     use = [o for o in objs if o.type == 'MESH' and o.name not in exclude_names]
#     if not use:
#         return Vector((0,0,0)), 1.0
#     center, size = get_bounds(use)
#     r = 0.5 * max(size.x, size.y)
#     return center, max(r, 1e-4)  # guard against zero
def scene_center_and_radius(objs, exclude_names=frozenset()):
    """Compute (center, radius_xy) from a set of objects, ignoring excluded ones."""
    keep = [o for o in objs if o and (o.name not in exclude_names)]
    meshes = [o for o in keep if o.type in {"MESH", "CURVE", "SURFACE", "META"}]
    if meshes:
        center, size = get_bounds(meshes)
        r = 0.5 * max(size.x, size.y)
        return center, max(r, 1e-4)

    # Fallback： use visual bounds for non-mesh objects
    center, size = get_visual_bounds(keep, include_children=True)
    r = 0.5 * max(size.x, size.y)
    return center, max(r, 1e-4)


def required_radius_for_coverage(cam: bpy.types.Object,
                                 scene: bpy.types.Scene,
                                 target_radius_xy: float,
                                 height: float,
                                 coverage: float = 1.0,
                                 margin: float = 1.1) -> float:
    """
    Solve radius so that a circle/sphere of radius=coverage*target_radius_xy
    centered at 'center' fits inside the camera frustum, given camera height.

    coverage: 1.0 -> fit-all; <1.0 -> close-up (show fraction of the scene)
    margin:   safety scale to avoid tight crop (e.g., 1.05~1.2)
    """
    r = max(target_radius_xy * max(coverage, 1e-3) * margin, 1e-6)
    fov_x, fov_y = camera_fov_xy(cam, scene)
    fov_min = min(fov_x, fov_y)
    # distance from camera to center required:
    D_req = r / math.tan(0.5 * fov_min)
    # convert to horizontal radius on the ring given a fixed height
    horiz = max(D_req*D_req - height*height, 0.0)
    return math.sqrt(horiz)

# ======== BBOX-BASED 2D PROJECTION & OCCLUSION ESTIMATION ========
def _get_depsgraph():
    """Obtain evaluated depsgraph robustly."""
    try:
        return bpy.context.evaluated_depsgraph_get()
    except AttributeError:
        return bpy.context.depsgraph

# World-space bbox center using original object.
def _world_bbox_center_eval(obj) -> Vector:
    """World-space bbox center using evaluated object (modifiers applied)."""
    deps, obj_eval, object_mesh, model_matrix_world = _eval_mesh(obj)
    corners = [model_matrix_world @ Vector(c) for c in obj_eval.bound_box]
    return sum(corners, Vector((0, 0, 0))) / 8.0

def _view_dir_xy_from_point(world_point, cam):
    """
    Same convention as U.view_dir_xy:
    u: left(-)->right(+)  v: bottom(-)->top(+)  w: behind(-)->in-front(+)
    """
    M = cam.matrix_world.to_3x3()
    right   = (M @ Vector((1, 0, 0))).normalized()
    up      = (M @ Vector((0, 1, 0))).normalized()
    forward = (M @ Vector((0, 0,-1))).normalized()
    d = (world_point - cam.location)
    if d.length == 0:
        d = forward.copy()
    d.normalize()
    return float(d.dot(right)), float(d.dot(up)), float(d.dot(forward))

def view_dir_xy(obj, cam):
    """
    Return (u, v, w) based on camera axes:
      u: left(-) -> right(+),   v: bottom(-) -> top(+),   w: behind(-) -> in-front(+).
    Works for all objects (even off-screen or behind camera).
    """
    c = _world_bbox_center_eval(obj)
    return _view_dir_xy_from_point(c, cam)

def _project_bbox_ndc_fast(obj, cam, scene):
    """Internal: project evaluated bbox corners to NDC, and check front/behind."""
    deps, obj_eval, object_mesh, model_matrix_world = _eval_mesh(obj)
    cam_inv = cam.matrix_world.inverted()
    xs, ys = [], []
    in_front = 0
    for c in obj_eval.bound_box:
        wp = model_matrix_world @ Vector(c)
        pv = world_to_camera_view(scene, cam, wp)  # returns x,y in [0,1] if inside; can be out of range
        xs.append(pv.x); ys.append(pv.y)
        p_cam = cam_inv @ wp
        if p_cam.z < 0.0:
            in_front += 1
    ndc_min = (min(xs), min(ys))
    ndc_max = (max(xs), max(ys))
    return ndc_min, ndc_max, (in_front == 0)  # all_behind?

def _render_size(scene):
    W = int(scene.render.resolution_x * scene.render.resolution_percentage / 100.0)
    H = int(scene.render.resolution_y * scene.render.resolution_percentage / 100.0)
    return W, H


def _bbox_from_ndc_to_coco(ndc_min, ndc_max, W, H):
    """Clamp NDC bbox to [0,1]^2 and convert to COCO [x,y,w,h] (top-left origin).
       Return None if no intersection."""
    nx0, ny0 = ndc_min; nx1, ny1 = ndc_max
    if nx1 <= 0.0 or nx0 >= 1.0 or ny1 <= 0.0 or ny0 >= 1.0:
        return None
    cx0 = max(0.0, min(1.0, nx0)); cy0 = max(0.0, min(1.0, ny0))
    cx1 = max(0.0, min(1.0, nx1)); cy1 = max(0.0, min(1.0, ny1))
    x  = int(round(cx0 * W))
    y  = int(round((1.0 - cy1) * H))   # NDC y=0 bottom -> COCO y=0 top
    w  = int(round((cx1 - cx0) * W))
    h  = int(round((cy1 - cy0) * H))
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]

def _bbox_coco(obj, cam, scene):
    """
    COCO bbox for object in this view, or None if completely off-screen
    (or entirely behind camera). Fast, bbox-corners based, not occlusion-aware.
    """
    ndc_min, ndc_max, all_behind = _project_bbox_ndc_fast(obj, cam, scene)
    if all_behind:
        return None
    W, H = _render_size(scene)
    return _bbox_from_ndc_to_coco(ndc_min, ndc_max, W, H)

def in_view_check(obj, cam, scene) -> bool:
    """
    True if object's projected bbox intersects the image rect and it's not fully behind camera.
    """
    ndc_min, ndc_max, all_behind = _project_bbox_ndc_fast(obj, cam, scene)
    if all_behind:
        return False
    nx0, ny0 = ndc_min; nx1, ny1 = ndc_max
    return not (nx1 <= 0.0 or nx0 >= 1.0 or ny1 <= 0.0 or ny0 >= 1.0)

def occlusion_rate(obj, cam, scene, max_samples: int = 800, seed: int | None = None):
    """
    Estimate occlusion ratio in [0..1] via area-weighted triangle sampling + ray_cast.
    Returns None if the object is not in view (quick bbox gate).
    Notes:
    - Samples are drawn proportional to triangle area (uniform over surface).
    - For each sample: require facing the camera, in front of camera, and inside image bounds.
      Then cast a ray from camera toward the sample; if any closer hit exists -> occluded.
      Otherwise visible.
    - Robust against vertex-edge gaps/self-precision issues by:
        * sampling *inside* triangles (barycentric), not at vertices
        * offsetting ray origin past near clip
        * stopping ray slightly *before* the sample point
    """
    # 0) quick reject by projected bbox
    if not in_view_check(obj, cam, scene):
        return None

    deps, obj_eval, object_mesh, model_matrix_world = _eval_mesh(obj)  # ensure evaluated mesh exists
    mw = model_matrix_world # alias

    if not object_mesh:
        return 0.0
    if not hasattr(object_mesh, "loop_triangles") or len(object_mesh.loop_triangles) == 0:
        object_mesh.calc_loop_triangles()

    verts = object_mesh.vertices
    tris_world: list[tuple[Vector, Vector, Vector, Vector, float]] = []
    cumsum = []
    total_area = 0.0

    # 1) build world-space triangles with area + world normal
    for tri in object_mesh.loop_triangles:
        i0, i1, i2 = tri.vertices
        v0 = mw @ verts[i0].co
        v1 = mw @ verts[i1].co
        v2 = mw @ verts[i2].co
        e1 = v1 - v0
        e2 = v2 - v0
        area = (e1.cross(e2)).length * 0.5
        if area <= 1e-12:
            continue
        n_world = e1.cross(e2).normalized()
        total_area += area
        cumsum.append(total_area)
        tris_world.append((v0, v1, v2, n_world, area))

    if total_area <= 1e-12:
        # obj_eval.to_mesh_clear()
        if hasattr(obj_eval, "to_mesh_clear"):
            try: obj_eval.to_mesh_clear()
            except: pass
        return 0.0

    rng = random.Random(seed if seed is not None else (hash(obj.name) & 0xFFFFFFFF))
    cam_origin = cam.location
    cam_inv = cam.matrix_world.inverted()

    consider = 0
    visible = 0

    # epsilons
    near_eps = max(getattr(cam.data, "clip_start", 0.01) * 1.5, 1e-4)
    stop_before = 1e-5  # stop a hair before the sample point

    # 2) draw samples and test visibility
    for _ in range(max_samples):
        t = rng.uniform(0.0, total_area)
        idx = bisect.bisect_left(cumsum, t)
        v0, v1, v2, n_world, _a = tris_world[idx]

        # barycentric sampling: sqrt trick for uniform in triangle
        r1 = rng.random()
        r2 = rng.random()
        sr1 = math.sqrt(r1)
        P = (1.0 - sr1) * v0 + sr1 * (1.0 - r2) * v1 + sr1 * r2 * v2

        # facing camera? (ignore backfaces to reduce false negatives)
        to_cam = (cam_origin - P).normalized()
        if n_world.dot(to_cam) <= 0.0:
            continue

        # in front of camera?
        P_cam = cam_inv @ P
        if P_cam.z >= 0.0:
            consider += 1 # if behind camera, still count as considered
            continue

        # inside image bounds?
        pv = world_to_camera_view(scene, cam, P)
        if pv.x < 0.0 or pv.x > 1.0 or pv.y < 0.0 or pv.y > 1.0:
            consider += 1 # if outside image, still count as considered
            continue

        consider += 1
        dir_vec = P - cam_origin
        dist = dir_vec.length
        if dist <= 1e-6:
            continue
        dir_n = dir_vec / dist

        # shift origin forward to avoid near-plane/self hits; stop just before P
        origin = cam_origin + dir_n * near_eps
        max_dist = max(0.0, dist - stop_before)

        hit, loc, normal, face_index, hit_obj, _hit_m = scene.ray_cast(deps, origin, dir_n, distance=max_dist)
        if not hit:
            # no closer geometry -> visible
            visible += 1
        else:
            # if first hit is this object (evaluated or original), still visible
            same = (getattr(hit_obj, "original", hit_obj) == obj) or (hit_obj == obj_eval)
            if same:
                visible += 1

    # obj_eval.to_mesh_clear()
    if hasattr(obj_eval, "to_mesh_clear"):
        try: obj_eval.to_mesh_clear()
        except: pass

    if consider == 0:
        # bbox gate said in-view but no valid samples; treat as no occlusion
        return 0.0

    occ = 1.0 - (visible / consider)
    return max(0.0, min(1.0, occ))

# 
def left_right_of_view(horizontal_value, eps=0):
    """
    Return "left", "right", or "center" based on object's horizontal position
    relative to camera view direction.
    """
    # horizontal_value < 0: left; horizontal_value > 0: right; horizontal_value ~= 0: center
    if horizontal_value < -eps:
        side_lr = "left"
    elif horizontal_value > eps:
        side_lr = "right"
    else:
        side_lr = "center"
    return side_lr

# Internal: get evaluated mesh, depsgraph, model matrix
def _eval_mesh(obj):
    deps = _get_depsgraph()
    obj_eval = obj.evaluated_get(deps)
    model_matrix_world = obj_eval.matrix_world
    geom_types = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}

    object_mesh = None
    if getattr(obj_eval, "type", None) in geom_types:
        try:
            # 3.x get evaluated mesh
            object_mesh = obj_eval.to_mesh()
        except Exception:
            object_mesh = None

    return deps, obj_eval, object_mesh, model_matrix_world



# ------------------------------For collection instance handling------------------------------
def _gather_render_members_under_root(root_obj, include_hidden_render=False):
    """
    Collect renderable children under a parent (usually an Empty that groups an imported asset).
    This does NOT modify visibility; we only filter by type and hide_render flag.
    """
    if root_obj is None:
        return []
    # direct children only (the asset importer in this file parents new objects to a root Empty)
    members = [o for o in bpy.data.objects
               if getattr(o, "parent", None) == root_obj
               and o.type not in {'LIGHT', 'CAMERA'}]
    if not include_hidden_render:
        members = [o for o in members if not getattr(o, "hide_render", False)]
    return members

def collection_world_aabb_from_root(root_obj, include_hidden_render=False):
    """
    World-space AABB of an imported asset grouped under a root (Empty).
    Use evaluated depsgraph so Modifiers / GN / Instances are baked.
    Returns (min_vec3, max_vec3) or (None, None) if nothing usable.
    """
    center, size = get_visual_bounds([root_obj], include_children=True)
    # if size is (0,0,0), treat as empty
    if (abs(size.x) + abs(size.y) + abs(size.z)) <= 1e-12:
        return (None, None)
    # center/size -> bmin/bmax
    hx, hy, hz = size.x * 0.5, size.y * 0.5, size.z * 0.5
    bmin = Vector((center.x - hx, center.y - hy, center.z - hz))
    bmax = Vector((center.x + hx, center.y + hy, center.z + hz))
    return (bmin, bmax)

def _ndc_from_world_aabb(aabb_min, aabb_max, cam, scene):
    """
    Project the 8 AABB corners to NDC. Return (ndc_min, ndc_max, all_behind)
    where ndc_* are (x,y) in NDC (can be outside [0,1]), and all_behind=True
    means all corners have positive camera-space z (i.e., behind camera).
    """
    if aabb_min is None:  # empty
        return (None, None, True)

    cam_inv = cam.matrix_world.inverted()
    xs, ys = [], []
    in_front = 0

    # 8 corners
    for x in (aabb_min.x, aabb_max.x):
        for y in (aabb_min.y, aabb_max.y):
            for z in (aabb_min.z, aabb_max.z):
                wp = Vector((x, y, z))
                pv = world_to_camera_view(scene, cam, wp)
                xs.append(pv.x); ys.append(pv.y)
                p_cam = cam_inv @ wp
                if p_cam.z < 0.0:
                    in_front += 1

    ndc_min = (min(xs), min(ys))
    ndc_max = (max(xs), max(ys))
    all_behind = (in_front == 0)
    return ndc_min, ndc_max, all_behind

def _bbox_from_ndc_to_coco_clamped(ndc_min, ndc_max, W, H):
    """Clamp NDC bbox to [0,1]^2 and convert to COCO [x,y,w,h]; None if no intersection."""
    nx0, ny0 = ndc_min; nx1, ny1 = ndc_max
    if nx1 <= 0.0 or nx0 >= 1.0 or ny1 <= 0.0 or ny0 >= 1.0:
        return None
    cx0 = max(0.0, min(1.0, nx0)); cy0 = max(0.0, min(1.0, ny0))
    cx1 = max(0.0, min(1.0, nx1)); cy1 = max(0.0, min(1.0, ny1))
    x  = int(round(cx0 * W))
    y  = int(round((1.0 - cy1) * H))   # NDC -> image top-left
    w  = int(round((cx1 - cx0) * W))
    h  = int(round((cy1 - cy0) * H))
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]



def view_dir_xy_collection(root_obj, cam):
    """Camera-relative (u,v,w) using the AABB center under a parent/root."""
    aabb = collection_world_aabb_from_root(root_obj)
    if aabb[0] is None:
        return 0.0, 0.0, -1.0
    c = (aabb[0] + aabb[1]) * 0.5
    return _view_dir_xy_from_point(c, cam)

def bbox_coco_collection(root_obj, cam, scene):
    """COCO bbox of a parent-grouped asset using its world AABB (depsgraph evaluated)."""
    bmin, bmax = collection_world_aabb_from_root(root_obj)
    ndc_min, ndc_max, all_behind = _ndc_from_world_aabb(bmin, bmax, cam, scene)
    if all_behind or ndc_min is None:
        return None
    W, H = _render_size(scene)
    return _bbox_from_ndc_to_coco_clamped(ndc_min, ndc_max, W, H)

def in_view_check_collection(root_obj, cam, scene) -> bool:
    """In-view test based on projected AABB intersection."""
    bmin, bmax = collection_world_aabb_from_root(root_obj)
    ndc_min, ndc_max, all_behind = _ndc_from_world_aabb(bmin, bmax, cam, scene)
    if all_behind or ndc_min is None:
        return False
    nx0, ny0 = ndc_min; nx1, ny1 = ndc_max
    return not (nx1 <= 0.0 or nx0 >= 1.0 or ny1 <= 0.0 or ny0 >= 1.0)



def _tris_from_object(o, deps=None, inherited_mat=None, require_mesh=True):
    """
    Build world-space triangles from an object.
    - If 'o' is a mesh-like object in depsgraph: use evaluated mesh.
    - If 'o' is not evaluated (e.g., collection member not linked to scene),
      try o.to_mesh() directly (modifiers won't apply, but OK for occlusion).
    - Apply 'inherited_mat' (e.g., instance Empty matrix) if provided.
    Returns: list of (v0, v1, v2)
    """
    tris = []
    try:
        o_eval = o.evaluated_get(deps) if (deps is not None and hasattr(o, "evaluated_get")) else o
    except Exception:
        o_eval = o

    # Try to_mesh on whatever we have
    me = None
    try:
        me = o_eval.to_mesh()
    except Exception:
        pass

    if not me:
        # not a mesh, just bail out
        if require_mesh:
            return tris
        else:
            return tris

    # Compose transform: inherited_mat @ o_eval.matrix_world if possible
    try:
        M = (inherited_mat @ o_eval.matrix_world) if (inherited_mat is not None) else o_eval.matrix_world
    except Exception:
        M = getattr(o_eval, "matrix_world", None)
        if inherited_mat is not None and M is not None:
            M = inherited_mat @ M

    try:
        me.transform(M)
    except Exception:
        pass

    verts = getattr(me, "vertices", None)
    polys = getattr(me, "polygons", None)
    if not verts or not polys:
        try:
            o_eval.to_mesh_clear()
        except Exception:
            pass
        return tris

    for poly in polys:
        idx = poly.vertices
        if len(idx) < 3:
            continue
        v0 = verts[idx[0]].co.copy()
        for i in range(1, len(idx) - 1):
            v1 = verts[idx[i]].co.copy()
            v2 = verts[idx[i+1]].co.copy()
            tris.append((v0, v1, v2))

    try:
        o_eval.to_mesh_clear()
    except Exception:
        pass
    return tris

# ------------------------------2D Sampling Utilities------------------------------
def random_anchor_in_square(span: float, obj_diameter: float, rng=random, exclusion_size: float = 0.0):
    """
    Uniform sample in square [-H,H]^2 excluding central square [-E,E]^2,
    where H = span/2 - obj_diameter/2  (keep object fully inside)
          E = min(exclusion, H-1e-9)   (cap to avoid degenerate)
    """
    H = 0.5 * (span - obj_diameter)
    if H <= 0:
        # object is too large; just return center
        print("[Warning] random_anchor_in_square: object diameter too large for span.")
        return 0.0, 0.0

    E = (exclusion_size + obj_diameter) * 0.5
    E = min(max(0.0, E), H - 1e-9)

    # Areas of the four rectangles (two groups with equal members):
    # LR group: x in [-H,-E] U [E,H], y in [-H,H]
    # TB group: x in [-E,E],      y in [-H,-E] U [E,H]
    area_lr = 2 * (H - E) * (2 * H)      # left + right
    area_tb = 2 * (2 * E) * (H - E)      # top + bottom
    total   = area_lr + area_tb
    if total <= 0:
        # no exclusion effectively
        return rng.uniform(-H, H), rng.uniform(-H, H)

    if rng.random() < (area_lr / total):
        # Left/Right rectangles
        if rng.random() < 0.5:
            x = rng.uniform(-H, -E)      # left
        else:
            x = rng.uniform( E,  H)      # right
        y = rng.uniform(-H, H)
    else:
        # Top/Bottom rectangles
        x = rng.uniform(-E, E)
        if rng.random() < 0.5:
            y = rng.uniform(-H, -E)      # bottom
        else:
            y = rng.uniform( E,  H)      # top
    return x, y



def in_view_check_collection_strict(root_obj, cam, scene, min_pixels=400):
    bmin, bmax = collection_world_aabb_from_root(root_obj)
    ndc_min, ndc_max, all_behind = _ndc_from_world_aabb(bmin, bmax, cam, scene)
    if all_behind or ndc_min is None:
        return False
    W, H = _render_size(scene)
    # compute pixel area
    x0 = max(0, int(round(ndc_min[0] * W))); y0 = max(0, int(round((1.0 - ndc_max[1]) * H)))
    x1 = min(W, int(round(ndc_max[0] * W))); y1 = min(H, int(round((1.0 - ndc_min[1]) * H)))
    area = max(0, x1 - x0) * max(0, y1 - y0)
    return area >= int(min_pixels)


GEOM_TYPES = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}

def has_geometry(obj_eval) -> bool:
    GEOM_TYPES = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}
    """Check if the object is a valid geometry type."""
    return getattr(obj_eval, "type", None) in GEOM_TYPES

# new bbox tight for collection instance. last version is too loose
def bbox_coco_collection_tight(root_obj, cam, scene, require_infront=True):
    GEOM_TYPES = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}
    deps = _get_depsgraph()
    cam_inv = cam.matrix_world.inverted()
    W, H = _render_size(scene)

    xs, ys = [], []
    xs_in, ys_in   = [], [] # new version is to calculate only the points that are in the view
    visited = set()  # prevent double-counting same object

    def _accum_obj(o, inherited_M=None):
        ptr = o.as_pointer()
        if ptr in visited:
            return
        visited.add(ptr)
        # process (Empty -> Collection)
        if o.type == 'EMPTY' and getattr(o, "instance_type", None) == 'COLLECTION' and getattr(o, "instance_collection", None):
            M_inst = (o.matrix_world if inherited_M is None else inherited_M @ o.matrix_world)
            # notice: here we do NOT check hide_render for instance children
            for child in o.instance_collection.objects:
                _accum_obj(child, inherited_M=M_inst)
            return

        # Also recurse into ordinary Empty's children (to match get_visual_bounds behavior)
        if o.type == 'EMPTY':
            for child in bpy.data.objects:
                if getattr(child, "parent", None) == o:
                    _accum_obj(child, inherited_M=inherited_M)
            return

        if o.type not in GEOM_TYPES:
            return
        
        o_eval = o.evaluated_get(deps)
        M_obj = o_eval.matrix_world
        M = M_obj if inherited_M is None else (inherited_M @ M_obj)

        me = None
        try:
            try:
                me = o_eval.to_mesh(preserve_all_data_layers=False, depsgraph=deps)
            except TypeError:
                me = o_eval.to_mesh()

            if me is None:
                return

            for v in me.vertices:
                wp = M @ v.co
                if require_infront and (cam_inv @ wp).z >= 0.0:  # Blender camera: in front => z < 0
                    continue
                pv = world_to_camera_view(scene, cam, wp)
                xs.append(pv.x); ys.append(pv.y) # old version
                if 0.0 <= pv.x <= 1.0 and 0.0 <= pv.y <= 1.0:
                    xs_in.append(pv.x); ys_in.append(pv.y) # new version

        finally:
            if me is not None and hasattr(o_eval, "to_mesh_clear"):
                o_eval.to_mesh_clear()


    # gather all render members
    members = _gather_render_members_under_root(root_obj, include_hidden_render=False)
    if members:
        for m in members:
            _accum_obj(m, inherited_M=None)
    else:
        _accum_obj(root_obj, inherited_M=None)

    if not xs or not ys:
        return None

    # use in-view points if possible
    xs = xs_in if xs_in else xs
    ys = ys_in if ys_in else ys

    nx0, nx1 = min(xs), max(xs)
    ny0, ny1 = min(ys), max(ys)

    # clamp to image rect
    if nx1 <= 0.0 or nx0 >= 1.0 or ny1 <= 0.0 or ny0 >= 1.0:
        return None
    cx0 = max(0.0, min(1.0, nx0)); cy0 = max(0.0, min(1.0, ny0))
    cx1 = max(0.0, min(1.0, nx1)); cy1 = max(0.0, min(1.0, ny1))

    x = int(round(cx0 * W))
    y = int(round((1.0 - cy1) * H))   # NDC -> image top-left
    w = int(round((cx1 - cx0) * W))
    h = int(round((cy1 - cy0) * H))
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def _current_scene_token(self):
    return int(bpy.context.scene.get("bvh_cache_token", 0))

def invalidate_bvh_cache(self):
    self._occ_cache["scene_tris"] = None
    self._occ_cache["scene_bvh"]  = None
    self._occ_cache["self_cache"].clear()

def _ensure_scene_bvh(self):
    token = self._current_scene_token()
    if self._occ_cache["scene_bvh"] is not None and self._occ_cache["token"] == token:
        return
    # rebuild scene BVH, if not present or token changed
    tris_scene = []
    deps = bpy.context.evaluated_depsgraph_get()

    def _add_tris_scene_from_obj(ob, inherited_mat=None):
        if getattr(ob, "hide_render", False):
            return
        if ob.type == 'EMPTY' and getattr(ob, "instance_type", None) == 'COLLECTION' and getattr(ob, "instance_collection", None):
            for child in ob.instance_collection.objects:
                _add_tris_scene_from_obj(child, inherited_mat=(inherited_mat @ ob.matrix_world if inherited_mat else ob.matrix_world))
            return
        if ob.type not in {'MESH', 'CURVE', 'SURFACE', 'META'}:
            return
        tris = _tris_from_object(ob, deps=deps, inherited_mat=inherited_mat)
        tris_scene.extend(tris)

    for ob in bpy.context.scene.objects:
        _add_tris_scene_from_obj(ob, inherited_mat=None)

    # reconstruct BVH
    from mathutils.bvhtree import BVHTree
    flat = [v for tri in tris_scene for v in tri]
    faces = [(3*i, 3*i+1, 3*i+2) for i in range(len(tris_scene))]
    scene_bvh = BVHTree.FromPolygons(flat, faces) if faces else None

    self._occ_cache["scene_tris"] = tris_scene
    self._occ_cache["scene_bvh"]  = scene_bvh
    self._occ_cache["token"]      = token

def _get_or_build_self_cache(self, root_obj):
    """build or get cached self-triangle data for root_obj."""
    key = int(root_obj.as_pointer())
    entry = self._occ_cache["self_cache"].get(key)
    if entry is not None:
        return entry

    # print(f"[self_cache] build start | root={getattr(root_obj, 'name', None)} | type={getattr(root_obj, 'type', None)} | hide_render={getattr(root_obj, 'hide_render', None)}")

    # build cache entry for root_obj, including expanded instances
    deps = bpy.context.evaluated_depsgraph_get()
    tris_self = []
    total_area = 0.0
    cumsum = []

    def _add_tris_self_from_members(members, inherited_mat=None):
        nonlocal tris_self, total_area, cumsum
        # print(f"[self_cache] members={len(members)} | inherited_mat={'yes' if inherited_mat is not None else 'no'}")
        for ob in members:
            # print(f"[self_cache]   ob={ob.name} | type={ob.type} | hide_render={getattr(ob, 'hide_render', None)}")
            if getattr(ob, "hide_render", False):
                continue
            if ob.type == 'EMPTY' and getattr(ob, "instance_type", None) == 'COLLECTION' and getattr(ob, "instance_collection", None):
                # print(f"[self_cache]   -> expand collection instance: {ob.name}")
                _add_tris_self_from_members(ob.instance_collection.objects, inherited_mat=(inherited_mat @ ob.matrix_world if inherited_mat else ob.matrix_world))
                continue
            if ob.type == 'EMPTY':
                children = [c for c in bpy.data.objects if getattr(c, "parent", None) == ob]
                if children:
                    # print(f"[self_cache]   -> expand empty children: {ob.name} | children={len(children)}")
                    _add_tris_self_from_members(children, inherited_mat=inherited_mat)
                continue
            if ob.type not in {'MESH', 'CURVE', 'SURFACE', 'META'}:
                continue
            tris = _tris_from_object(ob, deps=deps, inherited_mat=inherited_mat)
            # print(f"[self_cache]   -> tris_from_object: {ob.name} | tris={len(tris)}")
            for (v0, v1, v2) in tris:
                e1 = v1 - v0; e2 = v2 - v0
                area = (e1.cross(e2)).length * 0.5
                if area <= 1e-12:
                    continue
                tris_self.append((v0, v1, v2, area))
                total_area += area
                cumsum.append(total_area)

    members = _gather_render_members_under_root(root_obj, include_hidden_render=False)
    if members:
        _add_tris_self_from_members(members, inherited_mat=None)
    else:
        if root_obj.type == 'EMPTY' and getattr(root_obj, "instance_type", None) == 'COLLECTION' and getattr(root_obj, "instance_collection", None):
            # print(f"[self_cache] no members; expand root collection: {root_obj.name}")
            _add_tris_self_from_members(root_obj.instance_collection.objects, inherited_mat=root_obj.matrix_world)
        # else:
        #     print(f"[self_cache] no members and root not a collection instance: {getattr(root_obj, 'name', None)}")

    from mathutils.bvhtree import BVHTree
    tris_only = [(v0, v1, v2) for (v0, v1, v2, _a) in tris_self]
    flat = [v for tri in tris_only for v in tri]
    faces = [(3*i, 3*i+1, 3*i+2) for i in range(len(tris_only))]
    bvh_self = BVHTree.FromPolygons(flat, faces) if faces else None

    # print(f"[self_cache] build done | tris_self={len(tris_self)} | total_area={total_area} | bvh_self={'yes' if bvh_self else 'no'}")
    entry = {"tris_self": tris_self, "cumsum": cumsum, "total_area": total_area, "bvh_self": bvh_self}
    self._occ_cache["self_cache"][key] = entry
    return entry

# reliable version with caching
def occlusion_rate_collection_surface_cached(self, bbox, root_obj, cam, scene, max_samples=800, seed=None):
    # print(root_obj.name)
    # quick AABB reject
    if not bbox:
        # print("[occlusion_rate_collection_surface_cached] Invalid bbox input.")
        return None

    # make sure scene BVH is ready
    self._ensure_scene_bvh()
    scene_bvh  = self._occ_cache["scene_bvh"]
    tris_scene = self._occ_cache["scene_tris"]

    # get/build self cache
    s = self._get_or_build_self_cache(root_obj)
    tris_self  = s["tris_self"]; cumsum = s["cumsum"]; total_area = s["total_area"]; bvh_self = s["bvh_self"]
    if total_area <= 1e-12 or bvh_self is None or scene_bvh is None:
        # if total_area <= 1e-12:
        #     print(f"[occlusion_rate_collection_surface_cached] No usable surface in self (total_area={total_area}).")
        # elif bvh_self is None:
        #     print("[occlusion_rate_collection_surface_cached] No usable BVH in self.")
        # elif scene_bvh is None:
        #     print("[occlusion_rate_collection_surface_cached] No usable BVH in scene.")
        # else:
        #     print("[occlusion_rate_collection_surface_cached] No usable surface in self or scene.")

        # print("[occlusion_rate_collection_surface_cached] No usable surface in self or scene.")
        return 0.0

    rng = random.Random(seed if seed is not None else (hash(root_obj.name) & 0xFFFFFFFF))
    cam_origin = cam.location
    cam_inv = cam.matrix_world.inverted()

    consider = 0
    visible  = 0
    eps_depth = 1e-5
    near_eps  = max(getattr(cam.data, "clip_start", 0.01) * 1.5, 1e-4)

    for _ in range(max_samples):
        t = rng.uniform(0.0, total_area)
        idx = bisect.bisect_left(cumsum, t)
        v0, v1, v2, _area = tris_self[idx]

        # sampling + frustum and front/back check (same as original function)
        r1 = rng.random(); r2 = rng.random()
        sr1 = math.sqrt(r1)
        P = (1.0 - sr1) * v0 + sr1 * (1.0 - r2) * v1 + sr1 * r2 * v2

        P_cam = cam_inv @ P
        if P_cam.z >= 0.0:
            consider += 1
            continue
        pv = world_to_camera_view(scene, cam, P)
        if pv.x < 0.0 or pv.x > 1.0 or pv.y < 0.0 or pv.y > 1.0:
            consider += 1
            continue

        consider += 1
        dir_vec = P - cam_origin
        dist = dir_vec.length
        if dist <= 1e-6:
            continue
        dir_n = dir_vec / dist
        origin = cam_origin + dir_n * near_eps

        # first hit self, then hit scene, compare the order
        hit_self  = bvh_self.ray_cast(origin, dir_n)
        if hit_self[0] is None:
            visible += 1
            continue
        d_self = (hit_self[0] - origin).length

        hit_scene = scene_bvh.ray_cast(origin, dir_n)
        if hit_scene[0] is None:
            visible += 1
            continue
        d_scene = (hit_scene[0] - origin).length

        if d_scene + eps_depth < d_self:
            pass
        else:
            visible += 1

    # print(f"consider: {consider}, visible: {visible}")
    if consider == 0:
        return 0.0
    occ = 1.0 - (visible / consider)
    return max(0.0, min(1.0, occ))


def set_sky_hdri(self, exr_path: str):
    """Set world background to a sky HDRI from given .exr path."""

    exr_path = bpy.path.abspath(exr_path)
    if not os.path.isfile(exr_path):
        print(f"[set_sky_hdri] File not found: {exr_path}")
        return

    scene = self.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world

    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links

    # Background node
    bg = nodes.get("Background")
    if bg is None:
        bg = nodes.new(type="ShaderNodeBackground")

    # Environment Texture node
    env = nodes.get("Environment Texture")
    if env is None:
        env = nodes.new(type="ShaderNodeTexEnvironment")
        env.name = "Environment Texture"

    # Connect env to background's Color
    # (First remove old Color input connections)
    for link in list(links):
        if link.to_node == bg and link.to_socket == bg.inputs['Color']:
            links.remove(link)
    links.new(env.outputs['Color'], bg.inputs['Color'])

    # Load HDRI image
    env.image = bpy.data.images.load(exr_path)
    print(f"[set_sky_hdri] Loaded HDRI: {exr_path}")


def make_seed(base_seed: int,  scene_i: int) -> int:
    s = f"{base_seed}|{scene_i}"
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) 


def bbox3d_projected_wireframe_collection(root_obj, cam, scene, require_infront=True):
    """
    Project the object's world AABB (3D bbox) into the current camera view and
    return a wireframe in 2D pixel coordinates.

    Returns:
      {
        "corners_px": [(x,y) or None] * 8,   # pixel coords in image space (top-left origin)
        "edges": [(i,j), ...]               # valid edges (both endpoints not None)
      }
      or None if no valid bbox.
    """
    # 1) Get world-space AABB (min/max) for the asset under this root empty.
    bmin, bmax = collection_world_aabb_from_root(root_obj)
    if bmin is None:
        return None

    # 2) Define the 8 corners in a fixed, consistent order.
    #    Bottom face (zmin): 0-1-2-3, Top face (zmax): 4-5-6-7
    corners_world = [
        Vector((bmin.x, bmin.y, bmin.z)),  # 0
        Vector((bmax.x, bmin.y, bmin.z)),  # 1
        Vector((bmax.x, bmax.y, bmin.z)),  # 2
        Vector((bmin.x, bmax.y, bmin.z)),  # 3
        Vector((bmin.x, bmin.y, bmax.z)),  # 4
        Vector((bmax.x, bmin.y, bmax.z)),  # 5
        Vector((bmax.x, bmax.y, bmax.z)),  # 6
        Vector((bmin.x, bmax.y, bmax.z)),  # 7
    ]

    # 3) Project each corner to NDC, then convert to pixel coords.
    W, H = _render_size(scene)
    cam_inv = cam.matrix_world.inverted()

    corners_px = [None] * 8
    for i, wp in enumerate(corners_world):
        # If requested, ignore corners behind the camera (Blender: in front => camera-space z < 0).
        if require_infront and (cam_inv @ wp).z >= 0.0:
            corners_px[i] = None
            continue

        pv = world_to_camera_view(scene, cam, wp)  # NDC-ish: x,y in screen space (may be outside [0,1])
        x_px = pv.x * W
        y_px = (1.0 - pv.y) * H  # convert to top-left origin
        corners_px[i] = (x_px, y_px)

    # 4) Standard 12 edges of a box (by corner indices).
    edges_all = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]

    # 5) Keep only edges whose endpoints are both valid (not None).
    edges = [(a, b) for (a, b) in edges_all if corners_px[a] is not None and corners_px[b] is not None]

    # If nothing usable remains, return None.
    if not edges:
        return None

    return {"corners_px": corners_px, "edges": edges}
