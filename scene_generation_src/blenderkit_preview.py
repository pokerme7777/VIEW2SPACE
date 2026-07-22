# Blender headless preview renderer for BlenderKit assets
# Usage example:
# my_blender ./src/blenderkit_preview.py -- \
#   --source_folder "../kit_cache/processed_materials" \
#   --material_bool true \
#   --res 768 \
#   --samples 32

from ast import arg
import bpy
import sys
import os
import json
import math
from mathutils import Vector
# from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import blender_utils 

# ------------------------------
# CLI args
# ------------------------------


def log(msg):
    line = f"[preview] {msg}"
    print(line)
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    except Exception:
        pass

# ------------------------------
# Utility: scene reset & helpers
# ------------------------------
def world_cfg(ENGINE, FILM_TRANSPARENT, RES, SAMPLES):
    scene = bpy.context.scene
    scene.render.engine = ENGINE
    scene.render.film_transparent = FILM_TRANSPARENT
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100

    if ENGINE == 'CYCLES':
        scene.cycles.device = 'CPU'  # headless-safe
        scene.cycles.samples = SAMPLES
        scene.cycles.use_denoising = True
    else:
        scene.eevee.taa_samples = SAMPLES
        scene.eevee.use_gtao = True

    # Color management – keep defaults; ensure no extreme looks
    try:
        scene.view_settings.view_transform = scene.view_settings.view_transform  # no-op to avoid missing keys across versions
        scene.view_settings.look = 'None'
    except Exception:
        pass

    # Softer background light if not transparent
    if not FILM_TRANSPARENT and scene.world:
        scene.world.use_nodes = True
        nt = scene.world.node_tree
        bsdf = next((n for n in nt.nodes if n.type == 'BACKGROUND'), None)
        if bsdf:
            bsdf.inputs[1].default_value = 1.0  # strength


# ------------------------------
# IO helpers
# ------------------------------
def load_records_from_jsonl(mapping_path: str) -> list[dict]:
    """Load mapping.jsonl (one JSON per line). Return list of records."""
    return blender_utils.load_records_from_jsonl(mapping_path)


def update_preview_in_jsonl(mapping_path: str, stem: str, preview_png: str) -> None:
    """Set rec['preview_img'] = absolute preview_png for the asset matching <stem>."""
    if not os.path.isfile(mapping_path):
        return
    abs_png = os.path.abspath(preview_png)
    changed = False
    buf = []
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                buf.append(line)
                continue
            try:
                rec = json.loads(s)
            except Exception:
                buf.append(line)
                continue

            # match by file name stem from known fields
            matched = False
            candidates = [
                rec.get("new_file"),
                rec.get("dst"),
                rec.get("src"),
            ]
            for cand in candidates:
                if not isinstance(cand, str):
                    continue
                base = os.path.basename(cand)
                if base.lower().endswith(".blend"):
                    base = os.path.splitext(base)[0]
                else:
                    base = os.path.splitext(base)[0]
                if base == stem:
                    matched = True
                    break

            if matched:
                if rec.get("preview_img") != abs_png:
                    rec["preview_img"] = abs_png
                    changed = True
                buf.append(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                buf.append(line)

    if changed:
        tmp = mapping_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for x in buf:
                f.write(x if isinstance(x, str) else str(x))
        os.replace(tmp, mapping_path)


# ------------------------------
# xyz axes visualization
# ------------------------------
def add_renderable_axes(center, axis_len, thickness=0.02, cone_scale=2.5, emissive_strength=5.0):
    """
    Create renderable XYZ axes at `center` using cylinders + cones with emissive materials.
    Returns a list of created objects.
    """
    import bpy
    from mathutils import Vector

    def make_emissive_material(name, color, strength):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        emit = nt.nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
        emit.inputs["Strength"].default_value = strength
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        return mat

    red = make_emissive_material("AxisX_Emit", (1, 0, 0), emissive_strength)
    green = make_emissive_material("AxisY_Emit", (0, 1, 0), emissive_strength)
    blue = make_emissive_material("AxisZ_Emit", (0, 0, 1), emissive_strength)

    created = []
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=center)  # for grouping
    root = bpy.context.active_object
    root.name = "RenderableAxes_Root"

    def add_axis(dir_vec: Vector, mat, name_prefix):
        # shaft
        bpy.ops.mesh.primitive_cylinder_add(
            radius=thickness, depth=axis_len, enter_editmode=False,
            location=(center.x, center.y, center.z)
        )
        shaft = bpy.context.active_object
        shaft.name = f"{name_prefix}_shaft"
        shaft.data.materials.append(mat)
        # orient and move so it starts at center and points along dir_vec
        shaft.location = center + dir_vec.normalized() * (axis_len * 0.5)
        shaft.rotation_mode = 'QUATERNION'
        shaft.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(dir_vec.normalized())

        # arrow head
        bpy.ops.mesh.primitive_cone_add(
            radius1=thickness * cone_scale, depth=axis_len * 0.2,
            enter_editmode=False
        )
        head = bpy.context.active_object
        head.name = f"{name_prefix}_head"
        head.data.materials.append(mat)
        head.location = center + dir_vec.normalized() * (axis_len + (axis_len * 0.1))
        head.rotation_mode = 'QUATERNION'
        head.rotation_quaternion  = Vector((0, 0, 1)).rotation_difference(dir_vec.normalized())

        # hierarchy
        shaft.parent = root
        head.parent = root
        created.extend([shaft, head])

    from mathutils import Vector
    add_axis(Vector((1, 0, 0)), red, "AxisX")
    add_axis(Vector((0, 1, 0)), green, "AxisY")
    add_axis(Vector((0, 0, 1)), blue, "AxisZ")

    # ensure they render
    for obj in created + [root]:
        obj.hide_render = False
        obj.hide_set(False)

    return [root] + created

# ------------------------------
# Render setups
# ------------------------------

def setup_three_point_lights(center, radius):
    r = max(radius, 0.5)
    key = blender_utils.add_area_light("KeyLight", (center.x + r*1.2, center.y - r*1.8, center.z + r*1.2), power=40, size=r*2.5)
    fill = blender_utils.add_area_light("FillLight", (center.x - r*2.0, center.y + r*1.5, center.z + r*0.8), power=25, size=r*3.0)
    rim = blender_utils.add_area_light("RimLight", (center.x, center.y + r*2.5, center.z + r*1.8), power=20, size=r*2.0)
    blender_utils.aim_at(key, center)
    blender_utils.aim_at(fill, center)
    blender_utils.aim_at(rim, center)



def render_model_preview(blend_path, out_png, ENGINE, FILM_TRANSPARENT, RES, SAMPLES, show_axes=False):
    blender_utils.purge_scene()
    world_cfg(ENGINE, FILM_TRANSPARENT, RES, SAMPLES)

    cam = blender_utils.new_camera()

    # Load objects
    objs = blender_utils.append_objects_and_collections_from_blend(blend_path)
    bpy.context.view_layer.update()


    # Filter mesh objects
    meshes = [o for o in objs if o and o.type == 'MESH']
    if not meshes:
        meshes = blender_utils.all_mesh_objects()

    for o in meshes:
        try:
            o.select_set(False)
            if hasattr(o.data, 'use_auto_smooth'):
                o.data.use_auto_smooth = True
        except Exception:
            pass

    center, size = blender_utils.get_bounds(meshes)
    radius = max(size.x, size.y, size.z) * 0.5

    # --- Optional XYZ visualization ---
    if show_axes:
        axis_len = radius * 1.5
        add_renderable_axes(center, axis_len, thickness=radius * 0.02)
        log("Renderable XYZ axes added.")

    # --- FRONT VIEW camera (look from +Y toward -Y) ---
    # distance proportional to object size
    dist = radius * 1.3
    # cam.location = Vector((center.x + radius * 0.5, center.y - dist, center.z + radius*1))  # slight upward bias
    cam.location = Vector((center.x + radius * 0.001, center.y - radius*0.001, center.z + radius * 0.001))  # slight upward bias
    blender_utils.aim_at(cam, center)

    # setup_three_point_lights(center, radius)
    blender_utils.add_sun_light('KeyLight', (3.0, -3.0, 3.0), rotation=(math.radians(45), 0, math.radians(45)), power=20, angle_deg=0.5)
    # setup_ground_plane(center, radius)

    # Tighten framing via FOV heuristic
    bpy.context.view_layer.update()
    try:
        camd = cam.data
        sensor_h = camd.sensor_height or 24.0
        fov = 2.0 * math.atan((sensor_h * 0.5) / camd.lens)
        max_dim = max(size.x, size.y, size.z)
        needed = (max_dim * 0.6) / math.tan(fov * 0.5)
        current = (center - cam.location).length
        if needed > current:
            direction = (cam.location - center).normalized()
            cam.location = center + direction * needed * 1.2
            blender_utils.aim_at(cam, center)
    except Exception:
        pass

    # Output
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.filepath = out_png

    log(f"Rendering MODEL: {os.path.basename(blend_path)} -> {out_png}")
    bpy.ops.render.render(write_still=True)


def render_material_preview(blend_path, out_png, ENGINE, FILM_TRANSPARENT, RES, SAMPLES):
    blender_utils.purge_scene()
    world_cfg(ENGINE, FILM_TRANSPARENT, RES, SAMPLES)

    cam = blender_utils.new_camera()
    # Camera straight-on slight tilt
    cam.location = Vector((0.0, -3.0, 1.8))
    blender_utils.aim_at(cam, Vector((0.0, 0.0, 0.0)))

    # Load materials
    mats = blender_utils.append_materials_from_blend(blend_path)
    if not mats:
        log(f"WARN: No materials found in {blend_path}")
        return
    mat = mats[0]

    # Make a preview plane
    bpy.ops.mesh.primitive_plane_add(size=2.0, enter_editmode=False, align='WORLD', location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.data.materials.clear()
    plane.data.materials.append(mat)

    # Lights
    # setup_three_point_lights(Vector((0, 0, 0)), radius=1.5)
    blender_utils.add_sun_light('KeyLight', (3.0, -3.0, 3.0), rotation=(math.radians(45), 0, math.radians(45)), power=100, angle_deg=0.5)

    # Output
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.filepath = out_png

    log(f"Rendering MATERIAL: {os.path.basename(blend_path)} -> {out_png}")
    bpy.ops.render.render(write_still=True)


# ------------------------------
# Main
# ------------------------------
def getting_preview(source_folder, output_folder=None, RES=768, material_bool=False, SAMPLES=256,
                    FILM_TRANSPARENT=True, MAX_ASSETS=None, mapping_path="mapping.jsonl",
                    show_axes=False):

    SRC = os.path.abspath(source_folder)
    if output_folder is None:
        output_folder = SRC
    OUT = os.path.abspath(output_folder)

    IS_MATERIAL = material_bool
    ENGINE = "CYCLES"

    global LOG_PATH
    LOG_PATH = os.path.join(SRC, "preview_batch.log")

    if not os.path.isdir(SRC):
        log(f"ERROR: source_folder not found: {SRC}")
        return

    # --- mapping.jsonl is REQUIRED now ---
    if mapping_path is None:
        # default: look for a file named 'mapping.jsonl' in OUT first, then SRC
        candidate1 = os.path.join(OUT, "mapping.jsonl")
        candidate2 = os.path.join(SRC, "mapping.jsonl")
        mapping_path = candidate1 if os.path.isfile(candidate1) else candidate2
    else:
        mapping_path = os.path.join(SRC, "mapping.jsonl")
    records = load_records_from_jsonl(mapping_path)
    if not records:
        log("No mapping records found. Abort.")
        return

    # Build tasks: only missing/invalid previews
    tasks = []
    for rec in records:
        # 1) locate .blend
        bpath = rec.get("dst")
        if not bpath:
            name = rec.get("new_file")
            if name:
                bpath = os.path.join(SRC, name)
        if not bpath or not os.path.isfile(bpath):
            continue

        stem = os.path.splitext(os.path.basename(bpath))[0]

        # 2) decide preview path
        prev = rec.get("preview_img")
        if isinstance(prev, str) and prev.strip():
            out_png = prev if os.path.isabs(prev) else os.path.abspath(os.path.join(OUT, prev))
        else:
            out_png = os.path.join(OUT, f"{stem}.png")

        # 3) skip if preview already exists
        if os.path.isfile(out_png):
            continue

        tasks.append((stem, bpath, out_png))

    if MAX_ASSETS:
        tasks = tasks[:MAX_ASSETS]

    log(f"Found {len(records)} records, {len(tasks)} to render.")
    # for stem, bpath, out_png in tqdm(tasks, desc="Rendering previews", unit="asset", ascii=True):
    for stem, bpath, out_png in tasks: # [debug]
        try:
            if IS_MATERIAL:
                render_material_preview(bpath, out_png, ENGINE, FILM_TRANSPARENT, RES, SAMPLES)
            else:
                render_model_preview(bpath, out_png, ENGINE, FILM_TRANSPARENT, RES, SAMPLES, show_axes)

            # write preview path back
            update_preview_in_jsonl(mapping_path, stem, out_png)

        except Exception as e:
            log(f"ERROR: Failed to render {stem}: {e}")
        finally:
            blender_utils.purge_scene()

    log("Done.")


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="BlenderKit Asset Preview Renderer")
    parser.add_argument('-S', '--name', type=str, required=True,
                        help='Processed asset folder name under processed_asset/.')
    parser.add_argument('--data-root', type=str, default=os.environ.get("BENCHMARKING_DATA_CACHE"),
                        help='Root containing processed_asset/.')
    parser.add_argument('--res', type=int, default=768, help='Preview image resolution.')
    parser.add_argument('--samples', type=int, default=256, help='Cycles samples.')
    parser.add_argument('--material', action='store_true', help='Render assets as material previews.')
    parser.add_argument('--max-assets', type=int, default=None, help='Optional limit for smoke tests.')
    parser.add_argument('--show-axes', action='store_true', help='Render XYZ axes in model previews.')
    args, unknown_args = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])

    BLENDER_FOLDER = args.data_root
    if BLENDER_FOLDER is None:
        raise RuntimeError("Set BENCHMARKING_DATA_CACHE or pass --data-root.")
    getting_preview(
        source_folder=os.path.join(BLENDER_FOLDER, f"processed_asset/{args.name}"),
        output_folder=os.path.join(BLENDER_FOLDER, f"processed_asset/{args.name}_preview"),
        RES=args.res,
        material_bool=args.material,
        SAMPLES=args.samples,
        MAX_ASSETS=args.max_assets,
        show_axes=args.show_axes,
    )
