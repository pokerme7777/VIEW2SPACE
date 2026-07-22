import bpy
import os
import re
import json, ast
import math
import random
import sys
from pathlib import Path
from typing import Callable, Union, Optional
from mathutils import Vector
import itertools
from mathutils import Euler, Matrix



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
# Local utils (from your uploaded modules)
import blender_utils as U  # purge_scene, add_area_light, place_camera_above_scene, create_colored_object, get_bounds, all_mesh_objects
import blender_placing as P  # random_generate_non_overlapping_points, raycast_down_from_world_xy, drop_to_ground
import layout_generation_util as LGU  # different layout generation rules
from surrounding_src import place_surroundings
from random_place_ab import place_asset_random_ab
from getting_meta_util import _setup_pass_outputs, assign_instance_ids

class ScenePipeline:
    """
    A compact, class-based pipeline to generate a basic Blender scene, place objects, lights, and cameras,
    then render from multiple views while recording per-object and per-camera metadata.
    
    Key stages:
      1) setup_scene()  purge scene, create ground, world defaults
      2) generate_objects(specs)  make objects at non-overlapping XY, use raycast for Z
      3) place_lights(config)  add area/sun lights
      4) place_cameras(num_views or explicit view_indices)  reproducible placements
      5) render_all()  render each camera view to image files
      6) save_metadata()  write JSON with object + camera info per render
    """

    def __init__(self,
                 out_dir: str = "../caches",
                 seed: int | None = None,
                 resolution=(640, 480),
                 engine: str = "CYCLES",
                 samples: int = 16,
                 color_management: str = "Filmic",
                 file_format: str = "PNG",
                 use_transparency: bool = False,
                 scene_name: str = "Scene",
                 EXCLUDE_NAMES: list[str] = ["fence"]):
        if seed is not None:
            random.seed(seed)
        self.out_dir = bpy.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.scene = bpy.context.scene
        self.scene.name = scene_name
        self.scene.render.engine = engine
        self.scene.render.resolution_x = resolution[0]
        self.scene.render.resolution_y = resolution[1]
        self.scene.render.resolution_percentage = 100
        self.scene.view_settings.view_transform = color_management
        self.scene.render.image_settings.file_format = file_format
        self.scene.render.image_settings.color_mode = 'RGBA' if use_transparency else 'RGB'
        self.scene.render.film_transparent = use_transparency

        # Cycles settings (if used)
        if engine == "CYCLES":
            self.scene.render.engine = "CYCLES" 
            self.scene.cycles.samples = samples
            self.scene.cycles.feature_set = 'SUPPORTED'
            mode, backend, devices = U.enable_best_cycles_device(self.scene, prefer="AUTO", verbose=True)
            try:
                s = self.scene.cycles
                s.use_adaptive_sampling = True
                # set a low threshold to avoid too much noise; adjust as needed
                if hasattr(s, "adaptive_threshold"):
                    s.adaptive_threshold = 0.02

                # Denoiser: prioritize OPTIX for OPTIX backend, OIDN for others
                if hasattr(s, "use_denoising"):
                    s.use_denoising = True
                if hasattr(s, "denoiser"):
                    # s.denoiser = "OPTIX" if backend == "OPTIX" else "OPENIMAGEDENOISE"
                    if samples >= 256:
                        s.denoiser = "OPTIX" if backend == "OPTIX" else "OPENIMAGEDENOISE"
                    else:
                        s.denoiser = "NONE"  # Non-Local Means for low samples
            except Exception:
                pass
        
        self.cameras: list[bpy.types.Object] = []
        self.renders: list[dict] = []  # per-view record (paths, camera info)
        self.objects: dict[list[(bpy.types.Object, list[bpy.types.Object])]] = {}
        self.scene_center = Vector((0, 0, 0))
        self.ground_keywords = ("_ground", "fence")
        self.object_bound_dict = {} # store each object's bound/size info {"asset_code": Vector(x,y,z)}
        self.focused_objects_AABB = {"center": Vector((0,0,0)), "size": Vector((0,0,0))}
        self._occ_cache = {
            "scene_tris": None,
            "scene_bvh": None,
            "self_cache": {},   # key: int(root_obj.as_pointer())
            "token": -1,        # last seen scene token
        }
        self.EXCLUDE_NAMES = EXCLUDE_NAMES

    def load_records_from_jsonl(self, mapping_path: str) -> list[dict]:
        return U.load_records_from_jsonl(mapping_path)

    # --------------------
    # Stage 1: Base scene
    # --------------------
    def setup_scene(self, create_ground=True, ground_size=30.0, ground_z=0.0,
                    ground_it_records: list[dict] | None = None,
                    ground_material_index: int = 0,
                    ground_tile_size: float = 0.6,
                    place_ground: bool = False,
                    sky_hdri_record: dict | None = None):
        """
        Sets up the base scene.
        If `ground_blend_path` is provided, append materials from that .blend file
        (using your existing append_materials_from_blend function) and assign one to the ground.
        Otherwise, falls back to a simple diffuse grey material.
        """
        U.purge_scene()
        ground_name_must_be = "_ground"
        if ground_it_records is not None:
            ground_blend_path = ground_it_records[0].get("dst")
            ground_tile_size = ground_it_records[0].get("scale", 1.0)

        # Optional: create a simple ground plane for raycast hits and shadows
        if create_ground:
            bpy.ops.mesh.primitive_plane_add(size=ground_size, location=(0, 0, ground_z))
            plane = bpy.context.object
            # make sure it's named uniquely and with '_ground' suffix
            plane.name = ground_name_must_be

            mat = None

            if ground_it_records and os.path.isfile(ground_blend_path):
                try:
                    mats = U.append_materials_from_blend(ground_blend_path)
                    if mats:
                        # pick one (default index 0)
                        idx = min(max(0, ground_material_index), len(mats) - 1)
                        mat = mats[idx]
                        print(f"[setup_scene] Ground material loaded from '{ground_blend_path}': {mat.name}")
                    else:
                        print(f"[setup_scene] No materials found in {ground_blend_path}")
                except Exception as e:
                    print(f"[setup_scene] Failed to append materials from {ground_blend_path}: {e}")

            if mat is None:
                # Give a diffuse grey material
                mat = bpy.data.materials.new("Mat_Ground")
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.8
                    bsdf.inputs["Alpha"].default_value = 0.0
            
            # Assign material to ground plane
            if plane.data.materials:
                plane.data.materials[0] = mat
            else:
                plane.data.materials.append(mat)
            # --- NEW: fix material tiling so textures don't stretch when ground_size > 1 ---
            try:
                # Compute how many times to tile the texture in X/Y directions
                # Example: if ground_size = 30 and each material tile = 1m, we need 30 repeats
                tiles = max(1, int(round(ground_size / max(ground_tile_size, 0.1))))

                # Apply mapping and texture coordinate adjustments to the material
                U._ensure_tiling_on_material(mat, tile_x=tiles, tile_y=tiles, use_uv=True)

                # Ensure the plane has UVs (new planes usually do, but we check to be safe)
                if not plane.data.uv_layers:
                    bpy.context.view_layer.objects.active = plane
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.mesh.select_all(action='SELECT')
                    bpy.ops.uv.smart_project(angle_limit=66.0)
                    bpy.ops.object.mode_set(mode='OBJECT')
            except Exception as e:
                print("[setup_scene] tiling setup failed:", e)

        # elif place_ground:
        #     bpy.ops.mesh.primitive_plane_add(size=ground_size, location=(0, 0, ground_z))
        #     plane = bpy.context.object
        #     # make sure it's named uniquely and with '_ground' suffix
        #     plane.name = ground_name_must_be
        #     blend_path = "<path_to_ground_blend_file>"
        #     grp_list = self._import_asset_from_path(blend_path, name_hint=ground_name_must_be)
            
        #     ground_root = grp_list[0]
        #     ground_root.scale = (0.5, 0.5, 0.5)
        #     self.place_asset_by_xy(ground_root, 0, 0, 0)
        # self.objects[plane.name] = (plane, [])  # ignore this one for general generation!!!!!! VIS only
        if sky_hdri_record and sky_hdri_record.get("dst"):
            U.set_sky_hdri(self, exr_path=sky_hdri_record.get("dst"))

        return self

    # --------------------------------------
    # Stage 2: Object generation and placement
    # --------------------------------------
    def _bound_cache_key(self, asset_code, deg: float | None):
        if deg is None:
            return asset_code
        try:
            deg_key = int(round(float(deg))) % 360
        except Exception:
            return asset_code
        return f"{asset_code}|rot_{deg_key}"

    def get_bound_update_bound_dict(self, record, root, deg: float | None = None):
        asset_code = record.get("asset_code")
        if deg is None:
            try:
                deg = math.degrees(root.rotation_euler.z)
            except Exception:
                deg = None
        cache_key = self._bound_cache_key(asset_code, deg)
        if cache_key in self.object_bound_dict:
            size_vec = self.object_bound_dict[cache_key]
            center = root.location
        else:
            # Measure XY bounds via utility
            center, size_vec = U.get_visual_bounds([root])
            # store in dict
            self.object_bound_dict[cache_key] = size_vec
        return (center, size_vec)
        
    
    def get_bound_update_AABB(self, record, root, deg:float=None, focus_on_object: bool = True):
        """update self.object_bound_dict and self.focused_objects_AABB"""
        center, size_vec = self.get_bound_update_bound_dict(record, root, deg=deg)

        if any(ex_name in record.get("asset_code") for ex_name in self.EXCLUDE_NAMES):
            focus_on_object = False

        if deg is not None and focus_on_object:
            self.focused_objects_AABB = U.update_focused_object_AABB(current=self.focused_objects_AABB,
                                                    new={"center": (root.location.x, root.location.y, root.location.z),
                                                        "size": size_vec},
                                                    deg=deg)
        return (center, size_vec)
                                                    
    def _import_asset_from_path(self, blend_path: str, name_hint: str = "Imported", link_to_scene: bool = True) -> list[bpy.types.Object]:
        """import asset from <blend_path> and append it to the scene, returning the list of newly added objects."""
        inst = U.Load_locate_collection(blend_file=blend_path, object_name_in_scene=name_hint, location=(0,0,0))
        return [inst]
    
    def cleanup_objects(self, temp_handlers):
        """Cleanup temporary imported objects used for measurement."""
        delete_names = []
        for root, children in temp_handlers:
            delete_names.append(root.name)
            for o in children:
                if o and o.name in bpy.data.objects:
                    bpy.data.objects.remove(o, do_unlink=True)
            if root and root.name in bpy.data.objects:
                bpy.data.objects.remove(root, do_unlink=True)

        # also remove from self.objects tracking
        if self.objects:
            for name in delete_names:
                self.objects.pop(name, None)

    def _estimate_max_diameter_from_samples(self, items: list[dict], max_samples: int = 5, scale_key: str = "scale") -> float:
        """
        Import a few assets temporarily to estimate their maximum XY diameter.
        Clean them up after measuring. Uses U.get_bounds to measure. 
        """
        if not items:
            return 1.0  # fallback
        samples = random.sample(items, k=min(max_samples, len(items)))
        diameters = []
        temp_handlers = []

        for it in samples:
            path = it.get("dst")
            if not path:
                continue
            objs = self._import_asset_from_path(path, name_hint="__TMP_MEASURE", link_to_scene=False)
            if not objs:
                continue
            root = objs[0]
            sc = float(it.get(scale_key, 1.0) or 1.0)
            root.scale = (sc, sc, sc)

            _, size_vec = self.get_bound_update_AABB(it, root, deg=None, focus_on_object=False)
                
            diameters.append(float(max(size_vec.x, size_vec.y)))
            temp_handlers.append((root, []))

        # cleanup
        self.cleanup_objects(temp_handlers=temp_handlers)
        return max(diameters) if diameters else 1.0

    def place_asset_by_xy(self, root: bpy.types.Object, cx: float, cy: float, angle_deg: float):

        root.rotation_euler = Euler((0.0, 0.0, math.radians(angle_deg)), 'XYZ')
        root.location = (cx, cy, 50.0)  # start high, will drop down
        # c, s = U.get_visual_bounds([root])

        # Raycast down to find ground Z, then drop precisely (reuse your utils).
        hit, loc, *_ = P.raycast_down_from_world_xy(cx, cy, start_z=20.0, max_dist=200.0)
        z = float(loc.z) if hit else 0.0
        root.location.z = z + 0.1  # slight lift before drop
        P.drop_to_ground(root, z, clearance=0.0)   # uses your existing util

    def save_root_metadata(self, root, obj_name, blend_path, scale_size, angle_deg, asset_code, facing_value=0.0):
        """Helper to tag root with metadata for later JSON export."""
        root["name"] = obj_name
        root["dst"] = blend_path
        root["asset_code"] = asset_code
        root["asset_scale"] = scale_size
        root["rotation_deg"] = angle_deg
        root["location"] = tuple(root.location)
        root["facing"] = facing_value

    def _build_rect_layout(self, grid_n: int, seed: int | None = None,
                      disable_rows_random: bool = True, disable_minmax=(0, 1),
                      row_padding: float = 0.0):
        LGU.build_rect_layout(self, grid_n=grid_n, seed=seed,
                      disable_rows_random=disable_rows_random, disable_minmax=disable_minmax,
                      row_padding=row_padding)
        
    def _build_linear_layout(self, grid_n: int, seed: int | None = None,
                                only_active_a: bool = False,
                                row_padding: float = 0.0):
        LGU.build_linear_layout(self, grid_n=grid_n, seed=seed,
                                only_active_a=only_active_a,
                                row_padding=row_padding)

    def _build_b_on_a_layout(self, grid_n: int, seed: int | None = None,
                                row_padding: float = 0.0):
        LGU.build_b_on_a_layout(self, grid_n=grid_n, seed=seed,
                                row_padding=row_padding)

    def _process_assets_placement(self, K, name_prefix, chosen_cells, record_items):
        for idx, (cx, cy, save_deg) in enumerate(chosen_cells, start=1):
            # pick an item
            it = random.choice(record_items)
            blend_path = it.get("dst")

            # print(f"=======[place_asset_grid] Placing asset '{it.get('asset_code')}' at cell {idx}/{K}...")
            if not blend_path:
                continue
            scale_size = float(it.get("scale", 1.0) or 1.0)

            # import and group
            # obj_name = f"{name_prefix}_{idx:03d}"
            obj_name = self._unique_name(name_prefix, mode=None)

            grp_list = self._import_asset_from_path(blend_path, name_hint=obj_name)
            if not grp_list:
                continue
            root = grp_list[0]
            root.scale = (scale_size, scale_size, scale_size)

            # Discrete rotation around Z, if four_angles_deg is provided, use it. else use save_deg
            if save_deg is not None and self.four_angles_deg is None:
                angle_deg = save_deg
            else:
                try:
                    angle_deg = random.choice(self.four_angles_deg)
                except Exception as e:
                    print(f"Error choosing angle: {e}, please check four_angles_deg value.")

            # place by (cx, cy)
            self.place_asset_by_xy(root, cx, cy, angle_deg)
            
            center, size_vec = self.get_bound_update_AABB(it, root, deg=angle_deg, focus_on_object=True)

            # debug info
            # print(f" visual bbox center: {tuple(center)}")

            # save root metadata
            self.save_root_metadata(root, obj_name, blend_path, scale_size, angle_deg, asset_code=it.get("asset_code"), facing_value=it.get("facing", 0.0))
            # Track as scene object
            self.objects[obj_name] = (root, []) 

    def place_asset_grid(self,
                        A_record_items : list[dict],
                        B_record_items : list[dict] = None,
                        anchor_record: dict | None = None,
                        grid_n: int = 5,
                        place_count_A: int | None = None,
                        place_count_B: int | None = None,
                        keep_center: bool = False,
                        keep_center_value: int = 2,
                        linear_layout: bool = False,
                        only_active_a: bool = False,
                        a_on_b_layout: bool = False,
                        four_angles_deg: list[float] = [0.0, 90.0, 180.0, 270.0],
                        max_asset_diameter: float | None = None,
                        cell_padding: float = 0.0,
                        sample_for_size: bool = True,
                        random_seed: int | None = None,
                        name_prefix: str = "object"):
        """
        Place asset objects on a regular grid.
        - Build a grid of (grid_n x grid_n) cells.
        - Each placement randomly picks: a free cell, an asset entry, and a rotation {0,90,180,270} around Z.
        - Repeat for 'place_count_A' times (default = min(grid_n*grid_n, len(items)) but not exceed grid cells).

        Args:
            A_record_items: List of records for A class assets.
            B_class_jsonl_path: JSONL file for B class assets (if not NONE, then rectangle mode).
            anchor_record: optional dict {"dst": "...", "scale": 1.0 (optional)} for anchor asset to place first.
            keep_center: whether to reserve the center area for anchor placement.
            keep_center_value: number of cells to reserve at center (e.g., 2 means 2x2 cells).
            four_angles_deg: list of discrete rotation angles to choose from.
            grid_n: grid dimension (grid_n x grid_n cells).
            place_count_A: number of assets to place (default: grid_n*grid_n).
            place_count_B: number of B class assets to place (default: same as A).
            max_asset_diameter: if provided, use it directly; otherwise estimate from samples.
            cell_padding: fraction to inflate the cell size (e.g., 0.10 = +10%).
            sample_for_size: whether to import a few assets to estimate the max diameter.
            random_seed: make placements reproducible if provided.
        """
        if random_seed is not None:
            random.seed(random_seed)

        # 1) Load mapping list and init
        self.A_record_items = A_record_items 
        if not self.A_record_items:
            raise RuntimeError(f"No valid entries found in: {A_record_items}")

        self.B_record_items = B_record_items
        self.B_cell_centers = None

        self.keep_center = keep_center

        # 2) Determine cell size
        # Class A assets only for now
        if max_asset_diameter is None and sample_for_size:
            max_asset_diameter = self._estimate_max_diameter_from_samples(self.A_record_items, max_samples=max(1, len(self.A_record_items)))
            print("running this part===========debugging===========")
            print(f"[place_asset_grid] Estimated max asset diameter from samples: {max_asset_diameter:.3f} m")
        if max_asset_diameter is None:
            max_asset_diameter = 1.0  # fallback conservative
        
        # Class B assets are smaller.
        if self.B_record_items:
            B_max_asset_diameter = self._estimate_max_diameter_from_samples(self.B_record_items, max_samples=min(5, len(self.B_record_items)))
            self.B_cell_size = float(B_max_asset_diameter) * (1.0 + float(cell_padding))
            
        # init for anchor version
        anchor_diameter = 0.0 # init anchor diameter
        if anchor_record:
            anchor_path = anchor_record.get("dst")
            anchor_scale = float(anchor_record.get("scale", 1.0) or 1.0)
            anchor_diameter = self._estimate_max_diameter_from_samples([anchor_record], max_samples=1)
            print(f"[place_asset_grid] Anchor asset diameter: {anchor_diameter:.3f} m")


        cell_size = float(max_asset_diameter) * (1.0 + float(cell_padding))
        self.cell_size = cell_size

        # 3) Build grid centers around scene origin (0,0). Rows = Y+, Cols = X+.
        #    Grid spans [(−span/2, −span/2) .. (+span/2, +span/2)] in XY.
        # if anchor_record: anchor task x_times of cells
        
        x_times = math.ceil(anchor_diameter / self.cell_size) ** 2

        if self.B_record_items is None:
            real_grid_n = max(grid_n, math.ceil(math.sqrt(place_count_A + x_times)))
        else:
            real_grid_n = grid_n
        self.span = real_grid_n * self.cell_size

        # saving the space for centeral camera anchor
        # rules on all these buttons
        # if not B record items, then square layout
        # if having A and B record items, then rectangle layouts
        # if having A and B record items, a_on_b_layout is False, then rect layout
        # if having A and B record items, only_active_a is True, then A linear layout
        # if having A and B record items, a_on_b_layout is True, then A on B layout [A is the sqaure layout]
        # if having A and B record items, linear_layout is True, then A on B layout [A is the linear layout]
        if self.B_record_items is None or (a_on_b_layout and not linear_layout):
            x0 = -0.5 * self.span + 0.5 * self.cell_size
            y0 = -0.5 * self.span + 0.5 * self.cell_size
            # save (x, y, deg-None) for each cell center
            self.cell_centers = [(x0 + c * self.cell_size, y0 + r * self.cell_size, None) for r in range(real_grid_n) for c in range(real_grid_n)]
        else:
            if not linear_layout:
                # you will get self.cell_centers and self.B_cell_centers after building rect layout
                self._build_rect_layout(grid_n=real_grid_n, seed=random_seed,
                        disable_rows_random=True, disable_minmax=(0, 0),
                        row_padding=0.0)
            else:
                self._build_linear_layout(grid_n=real_grid_n, seed=random_seed,
                                        only_active_a=only_active_a,
                                        row_padding=0.0)

        if anchor_record:
            anchor_name_suffix = "anchor"
            anchor_name = self._unique_name(anchor_name_suffix, mode=None)
            anchor_list = self._import_asset_from_path(anchor_record.get("dst"), name_hint=anchor_name)
            if anchor_list:
                anchor_root = anchor_list[0]
                anchor_scale_value = anchor_record.get("scale", 1.0)
                anchor_root.scale = (anchor_scale_value, anchor_scale_value, anchor_scale_value)

                # place anchor at random location
                if self.keep_center:
                    # print("keeping center for anchor placement==========")
                    init_anchor_x, init_anchor_y = U.random_anchor_in_square(span=self.span, 
                                                                             obj_diameter=anchor_diameter, 
                                                                             exclusion_size=keep_center_value*self.cell_size)
                else:
                    init_anchor_x = random.uniform(-0.5 * (self.span - anchor_diameter), 0.5 * (self.span - anchor_diameter))
                    init_anchor_y = random.uniform(-0.5 * (self.span - anchor_diameter), 0.5 * (self.span - anchor_diameter))
                # place by (cx, cy)
                anchor_deg = 0.0
                self.place_asset_by_xy(anchor_root, init_anchor_x, init_anchor_y, anchor_deg)

                # save root metadata
                self.save_root_metadata(anchor_root, anchor_name, 
                                        anchor_path, anchor_scale, 
                                        anchor_deg, asset_code=anchor_record.get("asset_code"), 
                                        facing_value=anchor_record.get("facing", 0.0))
                

                # update both object_bound_dict and focused_objects_AABB
                _, size_vec = self.get_bound_update_AABB(anchor_record, anchor_root, deg=anchor_deg, focus_on_object=True)
                
                # save to objects tracking
                self.objects[anchor_name] = (anchor_root, [])

        # remove occupied cells
        occupied_cells = []
        
        for (cx, cy, save_deg) in self.cell_centers:
            if anchor_record and anchor_list:
                if abs(cx - init_anchor_x) < 0.5 * anchor_diameter and abs(cy - init_anchor_y) < 0.5 * anchor_diameter:
                    occupied_cells.append((cx, cy, save_deg))
            if self.keep_center and not B_record_items:
                if abs(cx) < keep_center_value/2 * self.cell_size and abs(cy) < keep_center_value/2 * self.cell_size:
                    occupied_cells.append((cx, cy, save_deg))

        # update available cells
        self.cell_centers = [cell for cell in self.cell_centers if cell not in occupied_cells]


        # 4) Decide how many to place
        total_cells = len(self.cell_centers)
        K = place_count_A if place_count_A is not None else total_cells
        K = max(0, min(K, total_cells))
        # self_real_K = K        

        # 5) Randomly pick K distinct cells
        self.chosen_cells = random.sample(self.cell_centers, k=K)

        # 6) Place one asset per chosen cell
        self.four_angles_deg = four_angles_deg
  
        # process all asset placement for class A
        self._process_assets_placement(K, name_prefix+"_A", self.chosen_cells, self.A_record_items)
        
        if a_on_b_layout and self.B_record_items:
            real_b_grid_n = real_grid_n * 3
            self._build_b_on_a_layout(grid_n=real_b_grid_n, seed=random_seed, row_padding=0.0)

        if self.B_cell_centers:
            K_b = place_count_B if place_count_B is not None else K
            K_b = max(0, min(K_b, len(self.B_cell_centers)))
            # accumulate total placements
            # self_real_K = K + K_b
      
        # process all asset placement for class B if any
        if self.B_cell_centers and K_b > 0:
            self.chosen_b_cells = random.sample(self.B_cell_centers, k=K_b)
            self._process_assets_placement(K_b, name_prefix+"_B", self.chosen_b_cells, self.B_record_items)


        self.side_len = self.span  # default side length if no fence built
        # Update bounds for downstream camera placement
        # self.scene_center, _ = U.get_bounds(U.all_mesh_objects())
        # roots = [tpl[0] for tpl in self.objects.values()]
        # if roots:
        #     self.scene_center, _ = U.get_visual_bounds(roots)
        # else:
        #     self.scene_center = Vector((0, 0, 0))

        return self
    
    def place_asset_random_ab(
        self,
        A_record_items: list[dict],
        B_record_items: list[dict] | None = None,
        place_count_A: int = 1,
        place_count_B: int | None = None,
        max_tries_per_obj: int = 50,
        GRID_N: float = 2.0,
        keep_center: bool = False,
        keep_center_value: int = 2,
        radius_growth: float = 1.2,
        four_angles_deg: list[float] | None = None,
        name_prefix: str = "object",
        ):
        return place_asset_random_ab(self,
        A_record_items,
        B_record_items,
        place_count_A,
        place_count_B,
        max_tries_per_obj,
        GRID_N,
        # radius_growth,
        keep_center,
        keep_center_value,
        radius_growth,
        four_angles_deg,
        name_prefix,
        )

    # ----------------------
    # stage 2b building fence using fence_path
    # ----------------------

        
    def build_fence(self,
                    it_records: list[dict] = None,
                    name_prefix: str = "fence",
                    clearance: float = 0.0,
                    building_fence_side_option: list[str] = ["top", "bottom", "left", "right"],
                    num_side: int = 3) -> bpy.types.Object:
        """
        Build a rectangular fence that wraps the current grid placement.

        Args:
            fence_it_records which contains the .blend file dst, asset_code, scale
                        The segment is assumed to be oriented lengthwise in XY (X or Y).
            name_prefix: Root object name for the fence group.
            clearance: Extra outward offset (meters) so the fence sits outside the grid span.

        Behavior:
            - Uses self.span.
            - Measures the fence segment's XY length (the longer of X/Y bounds).
            - Side length = ceil(self.span / seg_len) * seg_len  (rounded up to integer segments).
            - Places rows of segments on four sides, rotated 0/90/180/270 as needed.
            - Drops each segment to ground via the same raycast used for asset placement.

        Returns:
            The root Empty that parents all fence segments.
        """
        self.fence_side_option = building_fence_side_option
        if not hasattr(self, "span") :
            raise RuntimeError("Grid size unknown. Call place_asset_grid() before build_fence().")

        if it_records is None:
            return self
        
        # 1) init every variable
        current_it_record = it_records[0]
        fence_path = current_it_record.get("dst")
        scale_value = current_it_record.get("scale")

        # 2) Import ONE fence segment to measure; keep it as a template.

        tmpl_name = f"{name_prefix}_SEG_TEMPLATE"
        # avoid the duplicated name.
        tmpl_name = self._unique_name(tmpl_name, mode=None)

        seg_list = self._import_asset_from_path(fence_path, name_hint=tmpl_name, link_to_scene=True)
        if not seg_list:
            raise RuntimeError(f"Failed to import fence from: {fence_path}")
        seg_tmpl = seg_list[0]
        
        seg_tmpl.scale = (scale_value, scale_value, scale_value)
        seg_tmpl.location = (0, 0, -100.0) # make sure it's out of sight and raycast range

        center, size_vec = U.get_visual_bounds([seg_tmpl])
        seg_len = float(max(size_vec.x, size_vec.y)) # 
        seg_thickness = float(min(size_vec.x, size_vec.y))
        self.fence_thickness = seg_thickness

        # center_local = seg_tmpl.matrix_world.inverted() @ Vector((center.x, center.y, center.z))
        # origin_to_center_local = Vector((center_local.x, center_local.y, center_local.z))

        print(f"[build_fence] Measured fence segment length: {seg_len:.3f} m, thickness: {seg_thickness:.3f} m")
        print(f"self.span = {self.span:.3f} m")

        # print("==== Fence debug ====")
        # print("seg_tmpl:", seg_tmpl, "type:", type(seg_tmpl))
        # print("instance_type:", getattr(seg_tmpl, "instance_type", None))
        # print("instance_collection:", getattr(seg_tmpl, "instance_collection", None))
        # print("has mesh data:", bool(seg_tmpl.data))
        # print("child count:", len([o for o in bpy.data.objects if o.parent == seg_tmpl]))

        if seg_len <= 1e-6:
            raise RuntimeError("Measured fence segment length is ~0. Please check the asset orientation/scale.")

        # 3) Each side must be a multiple of seg_len that covers at least 'span'
        #    Note: we add a small safety factor to avoid rounding artifacts when seg_len ~ span.
        # side_len = math.ceil((self.span + 1) / seg_len) * seg_len
        side_len = math.ceil((self.span + 2 * clearance) / seg_len) * seg_len
        n_per_side = int(round(side_len / seg_len))

        # 4) Build a root Empty to parent everything (and keep template hidden).
        fence_root = bpy.data.objects.new(name_prefix, None)
        bpy.context.scene.collection.objects.link(fence_root)


        # 5) Compute positions for the four sides.
        #    We place the fence just outside the grid: half-span + clearance (+ half thickness).
        #    Using half thickness keeps the *inner* face approximately tangent to the grid boundary.
        # half = 0.5 * (self.span + 1)
        half = 0.5 * side_len
        outward = 0.5 * seg_thickness

        # Side definitions: (axis, const, rotation_z_deg)
        # - "X" sides (top/bottom): rows run along X, constant Y = +/-(half + outward), rot = 0 / 180
        # - "Y" sides (left/right): rows run along Y, constant X = +/-(half + outward), rot = 90 / 270
        sides = {
            "top": ("X", +(half + outward),   0.0),   # top    (+Y)
            "bottom": ("X", -(half + outward), 180.0),   # bottom (-Y)
            "right": ("Y", +(half + outward),  90.0),   # right  (+X)
            "left" : ("Y", -(half + outward), 270.0),   # left   (-X)
        }

        # pick side to build fence
        picked_side_names = random.sample(self.fence_side_option, k=min(num_side, len(self.fence_side_option)))

        # picked_sides = random.sample(sides, k=1)  # pick any 1 side to build fence
        self.fence_positions = picked_side_names
        self.side_len = side_len
        picked_sides = [sides[side_name] for side_name in picked_side_names]  # pick any 1 side to build fence

        # Start coord so instances are centered along each side:
        # Along the running axis we span [-side_len/2, +side_len/2] in steps of seg_len
        def _positions_along(length, count):
            start = -0.5 * length + 0.5 * seg_len  # center each segment on its slot
            return [start + i * seg_len for i in range(count)]

        # 6) Lay segments for each side
        for axis, const, rot_deg in picked_sides:
            coords = _positions_along(side_len, n_per_side)
            for t in coords:
                if axis == "X":
                    dup_x, dup_y =  t,  const
                else:
                    dup_x, dup_y = const,  t
                seg_dup = U.duplicate_hierarchy_linked(seg_tmpl)

                # R = seg_dup.rotation_euler.to_matrix()
                # off_world = R @ origin_to_center_local 
                # dup_x = dup_x - off_world.x
                # dup_y = dup_y - off_world.y

                # self.objects[seg_dup.name] = (seg_dup, [])  # ignore this one for general generation!!!!!! VIS only
                self.place_asset_by_xy(seg_dup, dup_x, dup_y, angle_deg=rot_deg)


        temp_handlers = []
        # Measure XY bounds via utility
        seg_tmpl_children = [c for c in bpy.data.objects if getattr(c, "parent", None) == seg_tmpl]
        temp_handlers.append((seg_tmpl, seg_tmpl_children))
        # 7) clean up template
        self.cleanup_objects(temp_handlers=temp_handlers)
        return self

    # ----------------------
    # stage 2c place some arts on top, bottom, left, right out of the grid
    # ----------------------
    def place_outdoor_arts(self,
                            it_records: list[dict] = None,
                            name_prefix: str = "outdoor_art",
                            clearance: float = 5,
                            focus_on_outdoorart: bool = False,
                            available_art_positions: list[str] = ["top", "bottom", "left", "right"]):
        """
        Place art objects outside the grid area at specified positions.
        Args:
            it_records: list of dictionaries containing .blend file paths and other metadata for art objects.
            positions: list of positions to place arts ("top", "bottom", "left", "right").
            clearance: distance from the grid boundary to place the arts.
        """
        self.available_art_positions = available_art_positions
        if it_records is None:
            return self

        if not hasattr(self, "span"):
            raise RuntimeError("Grid size unknown. Call place_asset_grid() before place_outdoor_arts().")

        for current_it_record in it_records:
            # 1) Import ONE art segment to measure; keep it as a template.
            art_paths = current_it_record.get("dst")
            scale_value = current_it_record.get("scale")

            half = 0.5 * (self.span) + clearance
            if hasattr(self, "fence_positions") and hasattr(self, "fence_thickness"):
                half += self.fence_thickness

            # avoid duplicated name
            tmpl_name = self._unique_name(name_prefix, mode=None)

            seg_list = self._import_asset_from_path(art_paths, name_hint=tmpl_name, link_to_scene=True)
            if not seg_list:
                raise RuntimeError(f"Failed to import art from: {art_paths}")
            seg_tmpl = seg_list[0]
            
            seg_tmpl.scale = (scale_value, scale_value, scale_value)
            seg_tmpl.location = (0, 0, -100.0) # make sure it's out of sight and raycast range

            center, size_vec = U.get_visual_bounds([seg_tmpl])
            # seg_len = float(max(size_vec.x, size_vec.y)) # 
            seg_thickness = float(min(size_vec.x, size_vec.y))
            half = 0.5 * (self.side_len + 1)
            outward = 0.5 * seg_thickness + clearance

            self.span = 2 * (half + outward) # update span to include fence clearance

            # all objects are facing -Y in the begining. We need to rotate them accordingly when placing on different sides.
            sides = {
                "top": ("X", +(half + outward),   0.0),   # top    (+Y)
                "bottom": ("X", -(half + outward), 180.0),   # bottom (-Y)
                "right": ("Y", +(half + outward),  270.0),   # right  (+X)
                "left" : ("Y", -(half + outward), 90.0),   # left   (-X)
            }

            picked_sides_name = random.sample(self.available_art_positions, k=1)
            # get picked sides info based on picked sides name
            picked_sides = [sides[side_name] for side_name in picked_sides_name]
            self.available_art_positions = [p for p in self.available_art_positions if p not in picked_sides_name]

            if not picked_sides:
                print("[place_outdoor_arts] No valid positions specified, skipping art placement.")
                return self

            for axis, const, angle_deg in picked_sides:
                seg_dup = U.duplicate_hierarchy_linked(seg_tmpl)
                if axis == "X":
                    dup_x, dup_y =  0.0,  const
                else:
                    dup_x, dup_y = const,  0.0
                self.place_asset_by_xy(seg_dup, dup_x, dup_y, angle_deg=angle_deg)

                center, size_vec = self.get_bound_update_AABB(current_it_record, seg_dup, deg=angle_deg, focus_on_object=focus_on_outdoorart)
                # if focus_on_outdoorart:
                self.objects[seg_dup.name] = (seg_dup, [])

                self.save_root_metadata(seg_dup, 
                                        seg_dup.name, 
                                        art_paths, 
                                        scale_value, 
                                        angle_deg, 
                                        asset_code=current_it_record.get("asset_code"),
                                        facing_value=current_it_record.get("facing", 0.0))


            temp_handlers = []
            # Measure XY bounds via utility
            seg_tmpl_children = [c for c in bpy.data.objects if getattr(c, "parent", None) == seg_tmpl]
            temp_handlers.append((seg_tmpl, seg_tmpl_children))
            # 7) clean up template
            self.cleanup_objects(temp_handlers=temp_handlers)
        return self

    def place_surroundings(self,
                            running_config: dict | None = None,
                            name_prefix: str = "surrounding"):
        return place_surroundings(self, running_config, name_prefix)

    def place_c_on_a_or_b(
        self,
        running_config: dict | None = None,
        place_count_C: int = 5,
        row_padding: float = 0.0,
        max_tries_per_obj: int = 30,
        name_prefix: str = "object_C",
    ):
        C_record_items = running_config.get("C_record_items")
        host_class = running_config.get("host_class")
        if not C_record_items:
            return self
        return LGU.place_c_on_a_or_b(
            self,
            C_record_items,
            place_count_C,
            row_padding,
            max_tries_per_obj,
            host_class,
            name_prefix,
        )

    # ----------------------
    # Stage 3: Lighting setup
    # ----------------------
    def place_lights(self,
                     area_power=300.0,
                     area_size=3.0,
                     area_positions=((4, -4, 6), (-4, 4, 6)),
                     sun_mode=False):
        """Place area lights and sun lights at specified positions."""
        if sun_mode:
            for i, pos in enumerate(area_positions):
                U.add_sun_light(name=f"SunLight_{i}", location=pos, rotation=(0, 0, 0), power=5.0, angle_deg=0.5)
        else:
            for i, pos in enumerate(area_positions):
                U.add_area_light(name=f"AreaLight_{i}", location=pos, power=area_power, size=area_size)
        return self

    # Helper to compute scene center/radius excluding ground-like objects
    def exclude_ground_scene_center_and_radius(self, ignore_ground=True, exclude_names=("_ground", "fence")):
        """
        Compute scene center and radius while excluding objects by name prefixes
        (e.g. 'fence', 'ground', 'sky', etc.).
        Handles both mesh objects and Empty parents.
        """
        #  shortcut if focused_objects_AABB is available
        if self.focused_objects_AABB:
            center = self.focused_objects_AABB["center"]
            size = self.focused_objects_AABB["size"]
            radius = 0.5 * math.sqrt(size.x**2 + size.y**2) # should we consider z?
            # radius = 0.5*math.sqrt(size.x**2 + size.y**2 + size.z**2)
            self._arena_radius = radius
            return center, radius

        exclude_names = exclude_names or set("_ground", "fence")
        all_objs = U.all_renderable_objects(include_empty_parents=True)

        # fallback: use self.objects if available
        if (not all_objs) and hasattr(self, "objects") and self.objects:
            roots = [tpl[0] for tpl in self.objects.values() if tpl and tpl[0]]
            all_objs = [o for o in roots if o]

        exclude_roots = [
            o for o in bpy.context.scene.objects
            if any(o.name.lower().startswith(pfx.lower()) for pfx in exclude_names)
        ]

        excluded = set()
        def gather_descendants(obj):
            for child in (c for c in bpy.data.objects if c.parent == obj):
                excluded.add(child)
                gather_descendants(child)

        # gather all descendants of excluded roots
        for root in exclude_roots:
            excluded.add(root)
            gather_descendants(root)        

        # ignore ground by keyword (legacy)
        all_objs = [o for o in all_objs if o not in excluded]

        # exclude by prefix (case-insensitive)
        if ignore_ground:
            all_objs = [o for o in all_objs if not self._is_ground_name(o.name)]

        if not all_objs:
            raise RuntimeError("No valid objects left for scene center computation.")

        center, radius = U.scene_center_and_radius(all_objs)
        self._arena_radius = radius
        return center, radius

    # -----------------------
    # Stage 4: Camera staging
    # -----------------------
    def place_cameras(self,
                      num_views: int = 2,
                      view_indices: list[int] | None = None,
                      radius: float | None = None,
                      height: float = 6.0,
                      jitter: float = 0.25,
                      cam_name_prefix: str = "Cam",
                      exclude_names: set[str] | None = None,
                      fit_coverage: float = 1.0,        # 1.0 fit-all; <1.0 partial
                      margin: float = 1.0,
                      ignore_ground: bool = True,
                      crop_side: str | None = None,   # 'left' | 'right' | 'random' | None
                      crop_frac: float | None = None,    # fraction of width to keep on cropped side
                      desired_pitch_deg: float = 40.0,
                      centerview_height: float = None, # New optional height when viewing from center
                      view_from_center: bool = False):       # how much to pan target, as fraction of r1):
        """
        Auto-compute camera radius (if not provided) so that selected objects fit
        a desired coverage ratio of the frame. Optionally shift target center for
        subsequent views (second_center) to get a close-up on a sub-region.
        """
        self.cameras.clear()

        # pick distinct view indices (0..3) if requested; otherwise random angles
        if view_indices is None:
            if view_from_center is None:
                chosen = random.sample([0, 10, 20, 30], k=min(num_views, 4))
            else:
                chosen = random.sample([0, 5, 10, 15, 20, 25, 30, 35], k=min(num_views, 8))
        else:
            chosen = view_indices[:num_views]

        tmp_cam = None
        if radius is None:
            # Make a tiny temp camera just to fetch FOV info once
            bpy.ops.object.camera_add()
            tmp_cam = bpy.context.object
            # mirror U.place_camera_above_scene defaults
            tmp_cam.data.lens = 35
            tmp_cam.data.sensor_width = 36
            tmp_cam.data.clip_start = 0.01
            tmp_cam.data.clip_end   = 1000.0
        try:
            # compute center/radius for the first target set
            center1, r1 = self.exclude_ground_scene_center_and_radius(
                ignore_ground=ignore_ground,
                exclude_names=exclude_names
            )

            # set ground z and min cam height
            c_all, s_all = self.focused_objects_AABB["center"], self.focused_objects_AABB["size"]
            
            eye_height = 1.8
            # --- Original ---
            # min_cam_height = max(eye_height + c_all.z, 0.5)

            # --- New ---
            # Either look from objects center or human eye height
            # Choose the lower one
            min_cam_height = max(c_all.z, eye_height/2)

            if s_all.z *3 > max(s_all.x, s_all.y):
                camera_rate = 1 - (max(s_all.x, s_all.y) / (s_all.z * 3))
                center1.z = (c_all.z + 0.5*s_all.z) * camera_rate
            else:
                center1.z = 0

            # get fov
            fov_x, fov_y = U.camera_fov_xy(tmp_cam if tmp_cam else self.scene.camera, self.scene)

            # pick radius1
            if radius is None:
                # fov_x, fov_y = U.camera_fov_xy(tmp_cam if tmp_cam else self.scene.camera, self.scene)
                radius1, height1 = U.cam_height_and_radius_for_coverage(
                    r_target=r1,
                    halfz_target=0.5 * s_all.z,
                    coverage=fit_coverage,
                    fov_x=fov_x,
                    fov_y=fov_y,
                    pitch_deg=desired_pitch_deg,
                    min_cam_height=min_cam_height,
                    margin=margin,
                )
            else:
                radius1 = radius
                height1 = height if height is not None else 6.0

            # Place per-view
            for i, vi in enumerate(chosen):
                center = center1 
                rad = radius1
                h_calculated = height1 
                # rad = radius1 if (second_center is None or i == 0) else radius2
                # h_calculated = height1 if (second_center is None or i == 0) else height2

                cam = U.place_camera_above_scene(
                    center=center,
                    view_index=vi,
                    radius=rad,
                    height=h_calculated,
                    jitter=jitter,
                    cam_name=f"{cam_name_prefix}_{i}",
                    target_offset=Vector((random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15), 0.0)),
                )
                cam["view_index_code"] = int(vi)
                cam["scene_setting"] = "original"

                # —— NEW: optionally crop to left/right side by panning target
                if crop_frac:
                    # calc right/left vector in world XY plane
                    dir_xy = Vector((cam.location.x - center.x, cam.location.y - center.y, 0.0))
                    if dir_xy.length < 1e-6:
                        dir_xy = Vector((1.0, 0.0, 0.0))
                    dir_xy.normalize()
                    # vertical component is zeroed out, so this is purely lateral
                    lateral = Vector((-dir_xy.y, dir_xy.x, 0.0))  

                    sign = 0.0  # init
                    #  3) determine crop side or bird_eye
                    if crop_side:
                        side = crop_side.lower()
                        if side == "random":
                            side = random.choice(["left", "right"])

                        # crop direction: left = -1, right = +1
                        sign = -1.0 if side == "left" else +1.0

                    # compute new target by panning along lateral vector
                    cov = max(1e-4, min(1.0, float(crop_frac)))  # 0..1
                    r_sub = cov * float(r1)

                    # 4) compute sub-coverage radius r_sub
                    inset_frac = 0.05  # 5% pull-in; tweak 0.03~0.08 
                    aim_dist = (float(r1) - 0.5 * r_sub) * (1.0 - inset_frac)
                    new_target = center + lateral * sign * aim_dist

                    # 5) compute required D_req to fit r_sub
                    fov_x, fov_y = U.camera_fov_xy(cam, self.scene)
                    fov_min = min(fov_x, fov_y)
                    D_req   = r_sub / math.tan(0.5 * fov_min)        # required distance to target

                    theta = math.radians(desired_pitch_deg)
                    horiz = max(0.25, D_req * math.cos(theta))       # horizontal radius (give a lower limit to avoid being too close)
                    z_up  = max(0.25, D_req * math.sin(theta))       # camera height (relative to ground/center.z)

                    # 6) reposition camera: keep horizontal orientation dir_xy, don't change orbit quadrant; height according to z_up
                    cam.location.x = new_target.x + dir_xy.x * horiz
                    cam.location.y = new_target.y + dir_xy.y * horiz
                    cam.location.z = center.z + z_up  

                    # re-aim camera
                    U.look_at(cam, new_target)
                    if crop_side:
                        cam["scene_setting"] = "bird_eye_cropped"
                        cam["crop_side"] = side
                    else:
                        cam["scene_setting"] = "bird_eye"
                    cam["target_center"] = new_target


                    
                elif view_from_center:
                    U.place_center_camera(cam, 
                                        height = min_cam_height if not centerview_height else centerview_height, # New
                                        jitter_xy = 0.1,
                                        cam_name_prefix = "CenterCam",
                                        AABB = self.focused_objects_AABB,
                                        view_index = int(vi))
                    cam["scene_setting"] = "center_view"
                self.cameras.append(cam)


        finally:
            # cleanup temp cam if we created one
            if tmp_cam is not None and tmp_cam.name in bpy.context.scene.objects:
                bpy.data.objects.remove(tmp_cam, do_unlink=True)

        # update for downstream
        return self

    def assign_instance_ids(self, start_id: int = 1, include_ground: bool = False):
        """
        Assign unique pass_index to each root object for instance segmentation.
        Uses Blender's Object.pass_index, which is emitted in the IndexOB render pass.
        """
        return assign_instance_ids(self, start_id, include_ground)

    # ----------------------
    # Stage 5: Render outputs
    # ----------------------
    def _setup_pass_outputs(self, out_basepath: str, enable_depth: bool = True, enable_seg: bool = True):
        return _setup_pass_outputs(self, out_basepath, enable_depth, enable_seg)
    

    def render_all(self, basename: str = "view", output_depth_seg: bool = False):
        self.renders.clear()

        for idx, cam in enumerate(self.cameras):
            self.scene.camera = cam
            vi = int(cam.get("view_index_code", idx))
            filepath = os.path.join(self.out_dir, f"{basename}_{vi:02d}")
            fmt = self.scene.render.image_settings.file_format.lower()
            out_path = Path(bpy.path.abspath(filepath + f".{fmt}"))
            out_path = Path(*out_path.parts[-3:]).as_posix()  # relative path for record
            self.scene.render.filepath = filepath

            self._setup_pass_outputs(out_basepath=filepath, enable_depth=output_depth_seg, enable_seg=output_depth_seg)

            bpy.ops.render.render(write_still=True)

            mods = list(getattr(self, "current_modifications", [])) # capture current mods if any
            # record render + camera info
            info = {
                "image_path": out_path,
                "view_index": vi,
                "camera_name": cam.name,
                "camera": self._camera_record(cam),
                "scene_setting": "modified" if mods else cam.get("scene_setting", "original"),
                "modifications": mods,
            }
            if output_depth_seg:
                info["depth_path"] = Path(bpy.path.abspath(filepath + "_depth0001.exr")).parts[-3:]
                info["seg_path"]   = Path(bpy.path.abspath(filepath + "_seg0001.exr")).parts[-3:]

            if cam.get("crop_side"):
                info["crop_side"] = cam.get("crop_side")
            self.renders.append(info)
        return self

    def _unique_name(self, base: str, mode: str=None) -> str:
        """Return a unique object name based on 'base'."""
        if base not in bpy.data.objects:
            return f"{base}"

        suffix = {"move": "_mv", "replace": "_replace", "add": "_add"}.get((mode or "").lower(), "")
        candidate = f"{base}{suffix}"
        if candidate not in bpy.data.objects:
            return candidate
        i = 1
        while f"{base}_{i:03d}" in bpy.data.objects:
            i += 1
        return f"{base}_{i:03d}"
    
    def _is_ground_name(self, name: str) -> bool:
        lname = name.lower()
        return (any(kw in lname for kw in self.ground_keywords) or ("ground" == lname))



    def modify_object(self,
                    mode: str = "replace",
                    move_pad: float = 0.15,
                    border: float = 0.2,
                    max_tries: int = 5,
                    running_config: dict | None = None,
                    min_move_radius: float = 3.0,
                    add_dist_range: tuple[float, float] = (1.5, 4.0)) -> dict:
        """
        Randomly pick one object and apply a modification.
        mode: 'replace' | 'delete' | 'move'
        Returns a dict describing the modification, using tuple pairs not strings:
            {'type':'delete', 'pair': (old_name, None)}
            {'type':'replace', 'pair': (old_name, new_name)}
            {'type':'move',   'pair': (old_name, new_name)}
            {'type':'add',    'pair': (old_name, new_name)}
        Notes:
          - Excludes ground-like names.
          - Excludes objects already modified in this session (tracked by self._modified_names).
        """
        if not hasattr(self, "_modified_names"):
            self._modified_names = set()
        if not hasattr(self, "current_modifications"):
            self.current_modifications = []

        self.invalidate_bvh_cache()
        bpy.context.scene["bvh_cache_token"] = int(bpy.context.scene.get("bvh_cache_token", 0)) + 1

        # candidate pool
        # candidates = [
        #     o for o in list(self.objects)
        #     if o.type == 'MESH' and not self._is_ground_name(o.name) and o.name not in self._modified_names
        # ]
        # candidates = [o for o in self.objects if not self._is_ground_name(o[0].name) and o[0].name not in self._modified_names]
        candidates_names = [name for name in self.objects.keys() if not self._is_ground_name(name) 
                            and name not in self._modified_names and not name.startswith("outdoor_")]
        if not candidates_names:
            return {}

        old_name = random.choice(candidates_names) 
        modify_obj = self.objects[old_name] #(root, children)
        modify_obj_root = modify_obj[0]
        
        if running_config.get("scene_type") =="CLEVR-like":
            allowed_shapes= running_config.get("shapes_options")
            color_choices=running_config.get("color_options")
            metallic_choices=running_config.get("material_options")
            height_range=ast.literal_eval(running_config.get("height_range"))

            # common attributes pulled from custom props
            shape = str(modify_obj.get("shape", "cube"))
            size  = float(modify_obj.get("size", 0.5))
            depth = float(modify_obj.get("depth", 1.0)) 
            color = str(modify_obj.get("color_name", "red"))
            metal = str(modify_obj.get("metallic_type", "non_metal"))

        def get_new_xy():
            available = [c for c in self.cell_centers if c not in self.chosen_cells]
            new_xy = (random.choice(available))
            self.chosen_cells.append(new_xy)
            return new_xy

        def _collect_obstacles():
            obs = []
            for o in self.objects:
                if o.type != 'MESH' or self._is_ground_name(o.name):
                    continue
                r_xy = P.footprint_radius_basic(str(o.get("shape","cube")),
                                        float(o.get("size",0.5))) + move_pad
                obs.append((float(o.location.x), float(o.location.y), r_xy))
            return obs

        if mode == "delete":
            # remove the object from scene and our list
            self.cleanup_objects(temp_handlers=[modify_obj])
            rec = {"type": "delete", "pair": (old_name, None)}

        elif mode == "replace":

            new_name = self._unique_name(old_name, mode="replace")  # keep lineage but unique

            if running_config.get("scene_type") =="CLEVR-like":
                # same size/height; random new shape (and depth if the new shape uses it)
                for max_attempts in range(5):
                    new_shape = random.choice(allowed_shapes)
                    new_color = random.choice(color_choices)
                    new_metal = random.choice(metallic_choices)
                    # ensure at least one attribute changes
                    if new_shape != shape or new_color != color or new_metal != metal:
                        break

                new_depth = random.uniform(*height_range) if new_shape in ("cube","cylinder","cone","torus") else depth
                loc = tuple(modify_obj.location)
                
                new_obj = U.create_colored_object(
                    obj_type=new_shape,
                    location=loc,
                    size=size,
                    depth=new_depth,
                    color_name=new_color,
                    metallic_type=new_metal,
                    object_name=new_name,
                )
                new_obj["shape"] = new_shape
                new_obj["size"] = size
                new_obj["depth"] = new_depth
                new_obj["color_name"] = new_color
                new_obj["metallic_type"] = metal

                # swap in our tracking list
                try:
                    self.objects.remove(modify_obj)
                except Exception:
                    pass
                bpy.data.objects.remove(modify_obj, do_unlink=True)
                self.objects.append(new_obj)
                self.cleanup_objects(temp_handlers=[modify_obj])
            else:
                
                # pick an item
                for _ in range(5):
                    it = random.choice(self.A_record_items)
                    blend_path = it.get("dst")
                    if blend_path != modify_obj_root.get("dst"):
                        break  # avoid same asset
                scale_size = float(it.get("scale", 1.0) or 1.0)

                # import and group
                grp_list = self._import_asset_from_path(blend_path, name_hint=new_name)
                new_root = grp_list[0]
                new_root.scale = (scale_size, scale_size, scale_size)

                # place at same XY, raycast for Z
                modify_obj_cx, modify_obj_cy = modify_obj_root.location.x, modify_obj_root.location.y
                modify_obj_angle_deg = modify_obj_root.get("rotation_deg", 0.0)
                #remove the old object from scene then place the new one
                self.cleanup_objects(temp_handlers=[modify_obj])
                self.place_asset_by_xy(new_root, modify_obj_cx, modify_obj_cy, modify_obj_angle_deg)

                # update focused AABB
                center, size_vec = self.get_bound_update_AABB(it, new_root, deg=modify_obj_angle_deg)

                # save root metadata
                self.save_root_metadata(new_root, new_name, blend_path, 
                                        scale_size, modify_obj_angle_deg, 
                                        asset_code=it.get("asset_code"),
                                        facing_value=it.get("facing", 0.0))
                # Track as scene object
                children = [c for c in bpy.data.objects if getattr(c, "parent", None) == new_root]
                self.objects[new_name] = (new_root, children)

                
            # remove the object from scene and our list
            rec = {"type": "replace", "pair": (old_name, new_name)}

        elif mode == "move":
            new_name = self._unique_name(old_name, mode="move")

            if running_config.get("scene_type") =="CLEVR-like":
                # keep same attributes, but re-sample XY using variable-radius logic against existing obstacles
                # 1) collect obstacles (all other meshes)
                obstacles = _collect_obstacles()

                # 2) target radius for this object
                r_target = P.footprint_radius_basic(shape, size) + move_pad

                # 3) sample a new (x,y) avoiding obstacles, within arena
                arena = max(self._arena_radius + move_pad*1.5, 5.0)

                ok = False
                for _ in range(10):
                    x, y = P.sample_point_avoiding_obstacles(
                        r_target=r_target, obstacles=obstacles, arena_radius=arena,
                        border=border, max_tries=max_tries,
                    )
                    if math.hypot(x - modify_obj.location.x, y - modify_obj.location.y) >= min_move_radius:
                        ok = True; break
                if not ok:
                    # failed to find a valid new location; just move in +X direction by min_move_radius
                    x, y = modify_obj.location.x + min_move_radius, modify_obj.location.y

                # 4) precise Z via raycast
                hit, loc, normal, index, hit_obj, mat = P.raycast_down_from_world_xy(x, y, start_z=50.0, max_dist=200.0)
                z = loc.z if hit else 0.0

                new_obj = U.create_colored_object(
                    obj_type=shape,
                    location=(x, y, z + size),
                    size=size,
                    depth=depth,
                    color_name=color,
                    metallic_type=metal,
                    object_name=new_name,
                )
                P.drop_to_ground(new_obj, z, clearance=0.0)
                new_obj["shape"] = shape
                new_obj["size"] = size
                new_obj["depth"] = depth
                new_obj["color_name"] = color
                new_obj["metallic_type"] = metal

                try:
                    self.objects.remove(modify_obj)
                except Exception:
                    pass
                bpy.data.objects.remove(modify_obj, do_unlink=True)
                self.objects.append(new_obj)
            else:
                # pick a new cell
                new_obj_cx, new_obj_cy, save_deg = get_new_xy()
                if save_deg:
                    modify_obj_angle_deg = save_deg
                else:
                    modify_obj_angle_deg = modify_obj_root.get("rotation_deg", 0.0)  
                self.place_asset_by_xy(modify_obj_root, new_obj_cx, new_obj_cy, modify_obj_angle_deg)

                # update focused AABB
                center, size_vec = self.get_bound_update_AABB(modify_obj_root, modify_obj_root, deg=modify_obj_angle_deg)
                # save root metadata
                self.save_root_metadata(modify_obj_root, new_name, modify_obj_root.get("dst"),\
                                         modify_obj_root.get("asset_scale"), modify_obj_angle_deg, 
                                         asset_code=modify_obj_root.get("asset_code"),
                                         facing_value=modify_obj_root.get("facing", 0.0))
                # Track as scene object
                children = [c for c in bpy.data.objects if getattr(c, "parent", None) == modify_obj_root]
                self.objects[new_name] = (modify_obj_root, children)

                # also remove from self.objects tracking
                self.objects.pop(old_name, None)

            rec = {"type": "move", "pair": (old_name, new_name)}

        elif mode == "add":
            if running_config.get("scene_type") =="CLEVR-like":
                # randomly pick an existing object as target, and place a new object nearby
                new_shape = random.choice(allowed_shapes)
                new_depth = random.uniform(*height_range) if new_shape in ("cube","cylinder","cone","torus") else depth
                r_target_new = P.footprint_radius_basic(new_shape, size) + move_pad
                obstacles = _collect_obstacles()  # all existing objects are obstacles
                arena = max(self._arena_radius, 5.0)

                # sample a new (x,y) around the target obj, within add_dist_range
                d_min, d_max = add_dist_range
                ok = False
                for _ in range(max_tries):
                    x, y = P.sample_point_avoiding_obstacles(
                        r_target=r_target_new, obstacles=obstacles, arena_radius=arena,
                        border=border, max_tries=max_tries,
                    )
                    d = math.hypot(x - modify_obj.location.x, y - modify_obj.location.y)
                    if d_min <= d <= d_max:
                        ok = True; break
                if not ok:
                    # failed to find a valid new location; just move in +X direction by min_move_radius
                    x, y = modify_obj.location.x + d_min, modify_obj.location.y

                hit, loc, *_ = P.raycast_down_from_world_xy(x, y, start_z=50.0, max_dist=200.0)
                z = float(loc.z) if hit else 0.0

                new_name = self._unique_name("object", mode="add")
                new_obj = U.create_colored_object(
                    obj_type=new_shape, location=(x, y, z + size), size=size, depth=new_depth,
                    color_name=color, metallic_type=metal, object_name=new_name,
                )
                P.drop_to_ground(new_obj, z, clearance=0.0)
                new_obj["shape"] = new_shape; new_obj["size"] = size
                new_obj["depth"] = new_depth; new_obj["color_name"] = color
                new_obj["metallic_type"] = metal

                self.objects.append(new_obj)
            else:
                # randomly pick an item
                new_name = self._unique_name(old_name, mode="add")
                new_it = random.choice(self.A_record_items)
                new_blend_path = new_it.get("dst")
                scale_size = float(new_it.get("scale", 1.0) or 1.0)
                # import and group
                grp_list = self._import_asset_from_path(new_blend_path, name_hint=new_name)
                new_root = grp_list[0]
                new_root.scale = (scale_size, scale_size, scale_size)

                # Discrete rotation around Z
                if self.four_angles_deg is None:
                    self.four_angles_deg = [0.0, 90.0, 180.0, 270.0]
                angle_deg = random.choice(self.four_angles_deg)

                # place by (cx, cy)
                new_obj_cx, new_obj_cy, save_deg = get_new_xy()
                if save_deg:
                    angle_deg = save_deg
                self.place_asset_by_xy(new_root, new_obj_cx, new_obj_cy, angle_deg)

                center, size_vec = self.get_bound_update_AABB(new_it, new_root, deg=angle_deg)
                # save root metadata
                self.save_root_metadata(new_root, new_name, new_blend_path, 
                                        scale_size, angle_deg, asset_code=new_it.get("asset_code"),
                                        facing_value=new_it.get("facing", 0.0))
                # Track as scene object
                children = [c for c in bpy.data.objects if getattr(c, "parent", None) == new_root]
                self.objects[new_name] = (new_root, children)


            rec = {"type": "add", "pair": (old_name, new_name)}
            # mark target as modified
            self._modified_names.add(old_name)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # mark as modified this session to avoid double-modifying
        self._modified_names.add(old_name)
        # cache record for metadata
        self.current_modifications.append(rec)
        # refresh bounds for later camera placement
        # self.scene_center, size_vec = U.get_bounds(U.all_mesh_objects())
        self.scene_center, size_vec = U.get_visual_bounds(U.all_mesh_objects())

        return rec

    # ----------------------
    # Stage 7: Annotations/JSON
    # ----------------------
    def occlusion_rate_collection_surface_cached(self, bbox, root_obj, cam, scene, max_samples=800, seed=None):
        return U.occlusion_rate_collection_surface_cached(self, bbox, root_obj, cam, scene, max_samples, seed)
    def _get_or_build_self_cache(self, root_obj):
        return U._get_or_build_self_cache(self, root_obj)
    def _ensure_scene_bvh(self):
        U._ensure_scene_bvh(self)
    def invalidate_bvh_cache(self):
        U.invalidate_bvh_cache(self)
    def _current_scene_token(self):
        return U._current_scene_token(self)

    def save_metadata(self, filename: str = "metadata.json", append_mode: bool = False):
        """
        Save scene- and view-level metadata:
        - Scene-level: image_size, scene_center, objects (fixed attrs), lights
        - View-level: camera info, COCO bboxes per object (None if outside),
                        vertical_order (bottom->top), horizontal_order (left->right),
                        vertical_counts [below, inside, above],
                        horizontal_counts [left, inside, right]
        Notes:
        - Uses bbox corners (fast). Not occlusion-aware.
        - Only iterates self.objects (the pipeline-created meshes), so ground plane
            from setup_scene() is naturally excluded.
        """
        scene = self.scene
        
        W, H = U._render_size(scene)
        out_path = os.path.join(self.out_dir, filename)

        data = {}
        old_views = []


        if append_mode and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            old_views = data.get("views", [])
            # ensure structure
            if not isinstance(data.get("objects"), list):
                data["objects"] = []
            if not isinstance(data.get("views"), list):
                data["views"] = []
            old_views = data["views"]
        else:
            data = {
                "image_size": [W, H],
                "scene_center": tuple(self.scene_center),
                "N_objects": len(self.objects),
                "objects": [],
                "grid_size": self.cell_size,
                "lights": [],
                "views": [],
            }

        # mesh_objs = [o for o in self.objects if o.type == 'MESH']
        roots = [o[0] for o in self.objects.values()]

        existing_names = {
            o.get("name") for o in data.get("objects", [])
            if isinstance(o, dict) and "name" in o
        }

        # Objects: fixed attributes, independent of view
        for obj in roots:
            if obj.name in existing_names:
                continue
            rot_deg = obj.get("rotation_deg")
            size_key = self._bound_cache_key(obj.get("asset_code"), rot_deg)
            if size_key not in self.object_bound_dict:
                size_key = obj.get("asset_code")
            rec = {
                "name": obj.name,
                "asset_code": obj.get("asset_code"),
                "dst": obj.get("dst"),
                "asset_scale": obj.get("asset_scale"),
                "rotation_deg": rot_deg,
                "location": tuple(obj.location),
                "size_object": tuple(self.object_bound_dict.get(size_key, Vector((0, 0, 0)))),
                "facing": obj.get("facing"),
            }
            data["objects"].append(rec)


        if not append_mode:
            # Lights
            for o in scene.objects:
                if o.type == 'LIGHT' and getattr(o, "data", None):
                    L = o.data
                    light_rec = {
                        "name": o.name,
                        "type": L.type,  # 'AREA' | 'SUN' | 'POINT' | 'SPOT'
                        "location": tuple(o.location),
                        "energy": float(getattr(L, "energy", 0.0)),
                        "color": tuple(getattr(L, "color", (1.0, 1.0, 1.0))),
                    }
                    if L.type == 'AREA':
                        light_rec["size"] = float(getattr(L, "size", 0.0))
                        light_rec["shape"] = getattr(L, "shape", "SQUARE")
                    elif L.type == 'SUN':
                        light_rec["angle"] = float(getattr(L, "angle", 0.0))
                    elif L.type == 'SPOT':
                        light_rec["spot_size"] = float(getattr(L, "spot_size", 0.0))
                        light_rec["spot_blend"] = float(getattr(L, "spot_blend", 0.0))
                    data["lights"].append(light_rec)

        # ---------- View-level (per camera) ----------
        new_views = []
        for view_idx, r in enumerate(self.renders):
            cam = next((c for c in self.cameras if c.name == r["camera_name"]), None) or scene.camera

            bbox_map = {}
            wireframe_map = {}  # NEW: per-object 3D bbox wireframe projected to this view
            visibility = {}
            x_pairs, y_pairs = [], []
            v_counts = {"below": 0, "inside": 0, "above": 0}
            h_counts = {"left": 0, "inside": 0, "right": 0}


            for root_collection in roots:
                name = root_collection.name

                # 1) bbox (COCO)
                # bbox = U.bbox_coco_collection(root_collection, cam, scene)
                bbox = U.bbox_coco_collection_tight(root_collection, cam, scene)

                bbox_map[name] = bbox

                # 1a) 3d bbox in world coords (for occlusion caching)
                wf = U.bbox3d_projected_wireframe_collection(root_collection, cam, scene)
                wireframe_map[name] = wf

                # 2) direction coords (all objects must be sortable)
                u, v, w = U.view_dir_xy_collection(root_collection, cam)
                x_pairs.append((u, name))  # left(-) -> right(+)
                y_pairs.append((v, name))  # bottom(-) -> top(+)

                # 3) in-view & occlusion & left/right
                # in_view = U.in_view_check_collection(root_collection, cam, scene)
                # in_view = U.in_view_check_collection_strict(root_collection, cam, scene, min_pixels=400)
                in_view = bbox is not None

                # occ = U.occlusion_rate_collection_surface(root_collection, cam, scene, max_samples=400) if in_view else 0.0
                occ = self.occlusion_rate_collection_surface_cached(bbox,root_collection, cam, scene, max_samples=200) if in_view else 0.0
                # occ = 0.0
                side_lr = U.left_right_of_view(u)
                visibility[name] = {"in_view": bool(in_view), "occlusion": None if occ is None else float(occ), "side_lr": side_lr}

                # 4) counts: if not in view, use sign of u/v to decide side; if in_view -> inside
                if in_view:
                    v_counts["inside"] += 1
                    h_counts["inside"] += 1
                else:
                    v_counts["below" if v < 0.0 else "above"] += 1
                    h_counts["left" if u < 0.0 else "right"] += 1

            # 5) orders (include ALL objects, even off-screen/behind)
            x_pairs.sort(key=lambda t: t[0])  # left -> right
            y_pairs.sort(key=lambda t: t[0])  # bottom -> top
            horizontal_order = [name for _, name in x_pairs]
            vertical_order   = [name for _, name in y_pairs]

            view_rec = {
                "view_index": r["view_index"],
                "image_path": r["image_path"],
                "camera_name": r["camera_name"],
                "camera": r["camera"],
                "scene_setting": r.get("scene_setting", "original"),
                "modifications": r.get("modifications", []),
                "bboxes": bbox_map,
                "bbox3d_wireframes": wireframe_map,  # NEW
                "vertical_order": vertical_order,         # bottom -> top (all objs)
                "horizontal_order": horizontal_order,     # left   -> right (all objs)
                "vertical_counts": [v_counts["below"], v_counts["inside"], v_counts["above"]],
                "horizontal_counts": [h_counts["left"], h_counts["inside"], h_counts["right"]],
                "visibility": visibility,
            }
            new_views.append(view_rec)
        
        # append and write
        data["views"] = old_views + new_views if append_mode else new_views

        # write file
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return out_path


    # ----------------------
    # Helpers
    # ----------------------
    def cleanup_memory(self):
        """
        Thoroughly clean up memory after each scene generation.
        Call this at the end of each scene to prevent memory accumulation.
        """
        # 1. Clear instance caches
        self.invalidate_bvh_cache()
        self.objects.clear()
        self.cameras.clear()
        self.renders.clear()
        self.object_bound_dict.clear()
        self.focused_objects_AABB = {"center": Vector((0,0,0)), "size": Vector((0,0,0))}

        if hasattr(self, '_modified_names'):
            self._modified_names.clear()
        if hasattr(self, 'current_modifications'):
            self.current_modifications.clear()
        if hasattr(self, 'chosen_cells'):
            self.chosen_cells.clear()
        if hasattr(self, 'cell_centers'):
            self.cell_centers.clear()

        # 2. Run orphans_purge multiple times to catch nested orphans
        for _ in range(5):
            try:
                bpy.ops.outliner.orphans_purge(
                    do_local_ids=True,
                    do_linked_ids=True,
                    do_recursive=True
                )
            except Exception:
                pass

        # 3. Manually remove unused data blocks
        # Meshes
        for block in list(bpy.data.meshes):
            if block.users == 0:
                bpy.data.meshes.remove(block)
        # Materials
        for block in list(bpy.data.materials):
            if block.users == 0:
                bpy.data.materials.remove(block)
        # Images
        for block in list(bpy.data.images):
            if block.users == 0:
                bpy.data.images.remove(block)
        # Collections
        for block in list(bpy.data.collections):
            if block.users == 0:
                bpy.data.collections.remove(block)
        # Node groups
        for block in list(bpy.data.node_groups):
            if block.users == 0:
                bpy.data.node_groups.remove(block)
        # Textures
        for block in list(bpy.data.textures):
            if block.users == 0:
                bpy.data.textures.remove(block)
        # Actions
        for block in list(bpy.data.actions):
            if block.users == 0:
                bpy.data.actions.remove(block)
        # Armatures
        for block in list(bpy.data.armatures):
            if block.users == 0:
                bpy.data.armatures.remove(block)
        # Curves
        for block in list(bpy.data.curves):
            if block.users == 0:
                bpy.data.curves.remove(block)
        # Libraries (linked data)
        for block in list(bpy.data.libraries):
            if block.users == 0:
                bpy.data.libraries.remove(block)

    def _camera_record(self, cam: bpy.types.Object) -> dict:
        pos = tuple(cam.location)
        # Approximate heading in degrees: angle of look-vector projected to XY plane
        # We aim the camera at scene_center; use vector from cam to center
        v = (Vector(cam["target_center"]) - cam.location)
        heading = math.degrees(math.atan2(v.y, v.x))  # [-180, 180]
        # Normalize to [0, 360)
        heading = (heading + 360.0) % 360.0
        return {
            "position": pos,
            "heading_deg": heading,
            "target": tuple(cam["target_center"]) if "target_center" in cam else None,
            "lens_mm": cam.data.lens if hasattr(cam.data, "lens") else None,
        }

def scene_done(scene_dir: Path, expected_views: int | None = None) -> bool:
    """
    Consider a scene 'done' if metadata.json exists.
    Optionally, also check that at least 'expected_views' images are present.
    """
    meta = scene_dir / "metadata.json"
    if not meta.exists():
        return False
    if expected_views is not None:
        # count files that look like rendered stills (png, jpg, exr, etc.)
        images = list(scene_dir.glob("*.png")) + list(scene_dir.glob("*.jpg")) + list(scene_dir.glob("*.exr"))
        return len(images) >= expected_views
    return True

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def generate_many_scenes(
    base_out_dir: str = "../caches",
    num_scenes: int = 20,
    views_per_scene: Union[int, Callable[[int], int]] = 2,
    start_index: int = 1,
    # seed_base: int = 1000, # final seed per scene = seed_base + i
    resolution=(800, 600), # Render resolution per view
    # resolution=(1920, 1080), # Render resolution per view
    # samples: int = 16, # Cycles samples
    samples: int = 256, # Cycles samples
    running_config: dict | None = None,
):
    """
    Generate multiple random scenes. Each scene goes into its own subfolder:
        <base_out_dir>/scene_001, scene_002, ...

    Parameters
    ----------
    base_out_dir : str
        Blender path; use '//' prefix to be relative to the .blend location.
    num_scenes : int
        How many scenes to produce in total.
    views_per_scene : int or callable(i)->int
        Number of camera views to render per scene. Can be a fixed int or a function
        that returns an int for scene index i (1-based).
    start_index : int
        Start numbering scenes from this index (default 1).
    seed_base : int
        Base seed; final seed per scene = seed_base + i to get reproducible randomness.
    resolution : (int, int)
        Render resolution per view.
    samples : int
        Cycles samples.

    """
    # Resolve base output directory to absolute path
    base_abs = Path(bpy.path.abspath(base_out_dir))
    ensure_dir(base_abs)

    produced = 0
    i = start_index
    last_index = start_index + num_scenes 

    # Load running config to get records
    A_record_items = running_config.get("A_record_items", None)
    B_record_items = running_config.get("B_record_items", None)
    if A_record_items is None and B_record_items is None:
        raise ValueError("A_record_items and B_record_items must be provided in running_config.")
    
    ground_material_records = running_config.get("ground_material_records", None)
    fence_records = running_config.get("fence_records", None)
    outdoorart_records = running_config.get("outdoorart_records", None)
    anchor_records = running_config.get("anchor_records", None)
    sky_records = running_config.get("sky_records", None)

    # Parameters for asset placement
    GRID_N = running_config.get("GRID_N")
    KEEP_CENTER_VALUE=running_config.get("KEEP_CENTER_VALUE", None)
    N_ASSET_A = running_config.get("N_ASSET_A")
    N_ASSET_B = running_config.get("N_ASSET_B", 0)
    N_ASSET_C = running_config.get("N_ASSET_C", 5)
    EXCLUDE_NAMES = running_config.get("EXCLUDE_NAMES", [])

    seed_base = running_config.get("SEED_BASE", 1000)
    # random.seed(seed_base)
    k_anchor_range = running_config.get("K_ANCHOR", [])
    k_fence_range = running_config.get("K_FENCE", [])
    k_outdoorart_range = running_config.get("K_OUTDOORART", [])


    # buttons
    keep_center = running_config.get("KEEP_CENTER", False)
    focus_on_outdoorart_button = running_config.get("FOCUS_ON_OUTDOORART", False)
    view_from_center = running_config.get("VIEW_FROM_CENTER", False)
    a_on_b_layout_button = running_config.get("A_ON_B_LAYOUT", False)
    is_training_scene = "training" in running_config.get("scene_type")

    for i in range(start_index, last_index):

        random.seed(U.make_seed(seed_base, i))
        # Decide view count for this scene
        k = views_per_scene(i) if callable(views_per_scene) else int(views_per_scene)

        # Per-scene subdir
        scene_dir = base_abs / f"scene_{i:03d}"
        ensure_dir(scene_dir)
        
        # Skip if already done
        if scene_done(scene_dir, expected_views=k):
            U.myprint(f"[Skip ] scene_{i:03d} already exists with metadata and images.")
            i += 1
            # if i % 5 == 0:
            # Training scene grows slower
            if running_config.get("RANDOM_PLACING_AB"):
                continue
            
            if is_training_scene:
                if i % 10 == 0:
                    N_ASSET_A += 1
                    N_ASSET_B += random.randint(0,1)
                    N_ASSET_C += random.randint(1,2)
            else:
                if i % 2 == 0:
                    N_ASSET_A += 1
                    N_ASSET_B += random.randint(0,1)
                    N_ASSET_C += random.randint(1,2)

            # ---New---
            for _ in range(3):
                if KEEP_CENTER_VALUE:
                    if N_ASSET_A >= GRID_N**2 - KEEP_CENTER_VALUE**2:
                        GRID_N+=1
                        # KEEP_CENTER_VALUE+=1
                    else: break
            continue

        # Decide random counts for this scene
        k_anchor = min(random.randint(k_anchor_range[0], k_anchor_range[1]) if k_anchor_range else 0, 1)
        k_fence = min(random.randint(k_fence_range[0], k_fence_range[1]) if k_fence_range else 0, 4)
        k_outdoorart = min(random.randint(k_outdoorart_range[0], k_outdoorart_range[1]) if k_outdoorart_range else 0, 4)

        # print(f"number of outdoor art to place: {k_outdoorart}")
        # print(f"number of fence to build: {k_fence}")
        # print(f"number of anchor to place: {k_anchor}")
        # print(f"linear layout: {running_config.get('linear_layout', None)}")
        # print(f"only active a: {running_config.get('only_active_a', None)}")
        # print("==============================")

        # random get qualified records
        picked_ground_records = random.sample(ground_material_records, k=1) if ground_material_records else None
        picked_fence_records = random.sample(fence_records, k=1) if fence_records else None
        picked_outdoorart_records = random.sample(outdoorart_records, k=k_outdoorart) if outdoorart_records else None
        picked_anchor_records = random.sample(anchor_records, k=k_anchor) if anchor_records else None
        picked_sky_record = random.choice(sky_records) if sky_records else None

        # Build a fresh pipeline with a per-scene seed
        pipe = (ScenePipeline(
                    out_dir=str(scene_dir),
                    seed=seed_base + i,
                    resolution=resolution,
                    samples=samples,
                    engine="CYCLES",
                    file_format="PNG",
                    use_transparency=False,
                    scene_name=f"Scene_{i:03d}",
                    EXCLUDE_NAMES=EXCLUDE_NAMES
                )
                # .setup_scene(create_ground=False, ground_blend_path=picked_ground_path, place_ground=True)
                .setup_scene(create_ground=True, 
                             ground_it_records=picked_ground_records, 
                             ground_size=running_config["ground_size"],
                             sky_hdri_record=picked_sky_record)
               )
        if not running_config.get("RANDOM_PLACING_AB"):
            pipe.place_asset_grid(
                    A_record_items=A_record_items,
                    B_record_items=B_record_items,
                    anchor_record=picked_anchor_records[0] if picked_anchor_records else None, # using outdoor for testing now
                    grid_n=GRID_N,
                    place_count_A=N_ASSET_A,
                    place_count_B=N_ASSET_B,
                    max_asset_diameter=None,
                    four_angles_deg=running_config.get("four_angles_deg", None),
                    cell_padding=0.0,
                    sample_for_size=True,
                    random_seed=seed_base + i,
                    keep_center=keep_center,
                    keep_center_value=running_config.get("KEEP_CENTER_VALUE", 2),
                    linear_layout=running_config.get("linear_layout", None),
                    only_active_a=running_config.get("only_active_a", None),
                    a_on_b_layout=a_on_b_layout_button,
                    )
        else: 
            pipe.place_asset_random_ab(
                    A_record_items=A_record_items,
                    B_record_items=B_record_items,
                    GRID_N=GRID_N,
                    place_count_A=N_ASSET_A,
                    place_count_B=N_ASSET_B,
                    keep_center=keep_center,
                    keep_center_value=running_config.get("KEEP_CENTER_VALUE", 2),
                    four_angles_deg=running_config.get("four_angles_deg", None)
                    )
        # Place fence and outdoor art in order
        fence_clearance = running_config.get("FENCE_CLEARANCE", 0.3)
        outdoorart_clearance = running_config.get("OUTDOORART_CLEARANCE", 1.0)
        view_indices = running_config.get("VIEW_INDICES", None)

        # Place fence and outdoor art in order
        if not running_config.get("OUTDOOR_FIRST", False):
            pipe.build_fence(it_records=picked_fence_records, 
                             clearance=fence_clearance, num_side=k_fence, 
                             building_fence_side_option=running_config.get("BUILDING_FENCE_SIDE_OPTION", ["top", "bottom", "left", "right"]))
            pipe.place_outdoor_arts(it_records=picked_outdoorart_records, clearance=outdoorart_clearance, focus_on_outdoorart=focus_on_outdoorart_button,
                                    available_art_positions=running_config.get("BUILDING_OUTDOOR_ART_SIDE_OPTION", ["top", "bottom", "left", "right"]))

        else:
            pipe.place_outdoor_arts(it_records=picked_outdoorart_records, 
                                    clearance=outdoorart_clearance,focus_on_outdoorart=focus_on_outdoorart_button,
                                    available_art_positions=running_config.get("BUILDING_OUTDOOR_ART_SIDE_OPTION", ["top", "bottom", "left", "right"]))
            pipe.build_fence(it_records=picked_fence_records, 
                             clearance=fence_clearance, num_side=k_fence, 
                             building_fence_side_option=running_config.get("BUILDING_FENCE_SIDE_OPTION", ["top", "bottom", "left", "right"]))
        
        pipe.place_surroundings(running_config=running_config)
        pipe.place_c_on_a_or_b(running_config=running_config, place_count_C=N_ASSET_C)
        
        pipe.place_lights(area_power=500.0, area_size=2.5, sun_mode=True)
        pipe.place_cameras(num_views=k, fit_coverage=running_config["fit_coverage"], 
                            desired_pitch_deg=running_config["normal_cam_deg"],
                            exclude_names=EXCLUDE_NAMES,
                            view_indices=view_indices,
                            view_from_center=False)  # <-- per-scene views here

        #
        pipe.assign_instance_ids(include_ground=True)
        # Render and save metadata
        pipe.render_all(basename="view")
        meta_path = pipe.save_metadata(filename="metadata.json")

        if view_from_center:
            pipe.place_cameras(num_views=k, fit_coverage=running_config["fit_coverage"], 
                                desired_pitch_deg=running_config["normal_cam_deg"],
                                exclude_names=EXCLUDE_NAMES,
                                centerview_height=running_config.get("CENTERVIEW_HEIGHT"), # New option
                                view_from_center=view_from_center)  # <-- per-scene views here
            pipe.render_all(basename="centerview")
            meta_path = pipe.save_metadata(filename="metadata.json", append_mode=True)

        # # birdeye view full
        # pipe.place_cameras(num_views=1, view_indices=[0], crop_frac=1.1, 
        #                     desired_pitch_deg=90)
        # # render image and save metadata
        # pipe.render_all(basename="view_birdeye")
        # meta_path = pipe.save_metadata(filename="metadata.json", append_mode=True)

        if not focus_on_outdoorart_button:
            # random pick an exclusion set for camera placement (to avoid looking at empty ground)
            exist_cams_index = [cam.get("view_index_code", 0) for cam in pipe.cameras]
            # selecte the view not in exist_cams_index as the
            selected_view = [vi for vi in [00,10,20,30] if vi not in exist_cams_index]
            if not selected_view:
                selected_view = [random.choice([00,10,20,30])]
            pipe.place_cameras(num_views=1, view_indices=selected_view, crop_frac=running_config["crop_image_fit_coverage"], 
                            crop_side=random.choice(["left","right"]), desired_pitch_deg=running_config["crop_cam_deg"])
            # render image and save metadata
            pipe.render_all(basename="view_crop")
            meta_path = pipe.save_metadata(filename="metadata.json", append_mode=True)

        # # modify_object example (optional)
        # modified_mode = random.choice(["replace", "delete", "move", "add"])
        # pipe.modify_object(mode=modified_mode, running_config=running_config)
        # # pipe.modify_object(mode="move", running_config=running_config)
        #  # re-place cameras at fixed views (2,3,4) after modification
        # pipe.place_cameras(num_views=1, view_indices=[2,3,4], fit_coverage=running_config["modified_image_fit_coverage"], desired_pitch_deg=running_config["normal_cam_deg"])
        # pipe.render_all(basename="view_mod")
        # meta_path = pipe.save_metadata(filename="metadata.json", append_mode=True)

        # U.myprint(f"[Done ] scene_{i:03d} -> {scene_dir} | metadata: {meta_path}")

        # Clean up memory after each scene to prevent memory leak
        pipe.cleanup_memory()
        del pipe

        produced += 1
        i += 1

        # if i % 5 == 0:
        if running_config.get("RANDOM_PLACING_AB"):
            continue

        if is_training_scene:
            if i % 10 == 0:
                N_ASSET_A += 1
                N_ASSET_B += random.randint(0,1)
                N_ASSET_C += random.randint(1,2)
        else:
            if i % 2 == 0:
                N_ASSET_A += 1
                N_ASSET_B += random.randint(0,1)
                N_ASSET_C += random.randint(1,2)

        # ---Old---
        # for _ in range(3):
        #         if N_ASSET_A*2 > GRID_N**2:
        #             GRID_N+=1
        #         else: break
        #     continue
        # ---New---
        for _ in range(3):
            if KEEP_CENTER_VALUE:
                if N_ASSET_A >= GRID_N**2 - KEEP_CENTER_VALUE**2:
                    GRID_N+=1
                    # KEEP_CENTER_VALUE+=1
                else: break

    U.myprint(f"Batch complete. Produced {produced} new scene(s) under: {base_abs}")

# =========================
# Example: run this in Blender's Scripting panel
# =========================

def safe_load_records(cache_dir, path_key, running_config, path_config, default_keys=None):
    """Helper to safely load and filter records by keywords."""
    real_path_key = running_config.get(path_key, None)
    if not real_path_key:
        return None
    
    jsonl_path = path_config.get(real_path_key)
    # print(jsonl_path)
    if not jsonl_path:
        return None
    full_path = jsonl_path if os.path.isabs(jsonl_path) else os.path.join(cache_dir, jsonl_path)
    records = U.load_records_from_jsonl(full_path)
    if not records:
        return None

    # pick keyword list from running_config or fallback
    key_map = {
        "A_item_path": "A_object_options",
        "B_item_path": "B_object_options",
        "GMAT_path": "ground_options",
        "fence_path": "fence_options",
        "anchor_path": "anchor_options",
        "outdoor_path": "outdoor_options",
        "sky_path": "sky_options",
        "surrounding_path": "surrounding_options",
        "C_item_path": "C_object_options",
    }
    option_key = key_map.get(path_key)
    keywords = running_config.get(option_key, default_keys or [])

    if not keywords or keywords == [""]:
        return records

    # filter
    filtered = [
        rec for rec in records
        if any(k in rec.get("new_file", "").lower() for k in keywords)
    ]
    return filtered or None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="generate scenes with grid placement")
    parser.add_argument('-n', '-S', '--name', '--scene-name', dest="name", type=str, required=True,
                        help='Scene config name to render.')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to scene_config.json. Defaults to scene_generation_src/config/scene_config.json.')
    parser.add_argument('--data-root', type=str, default=os.environ.get("BENCHMARKING_DATA_CACHE"),
                        help='Root containing kit_cache/, processed_asset/, and output scenes/.')
    parser.add_argument('--output-root', type=str, default="scenes",
                        help='Output folder under data-root, or an absolute output path.')
    parser.add_argument('--start-index', type=int, default=1,
                        help='First scene index to generate.')
    parser.add_argument('--samples', type=int, default=None,
                        help='Override Cycles samples from config.')
    parser.add_argument('--resolution', type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None,
                        help='Override render resolution from config.')
    args, unknown_args = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])

    cache_dir = args.data_root
    if cache_dir is None:
        raise RuntimeError("Set BENCHMARKING_DATA_CACHE or pass --data-root.")

    SCENE_NAME = args.name

    output_root = args.output_root if os.path.isabs(args.output_root) else os.path.join(cache_dir, args.output_root)
    scene_cache_dir = os.path.join(output_root, SCENE_NAME)
    # Make sure the directory exists
    os.makedirs(scene_cache_dir, exist_ok=True)


    # Load running config
    running_config, path_config = U.loading_config(scene_type=SCENE_NAME, path_return=True, config_path=args.config)
    # load A/B record items
     # A items
    A_record_items = safe_load_records(cache_dir, "A_item_path", running_config, path_config)
    B_record_items = safe_load_records(cache_dir, "B_item_path", running_config, path_config)
    ground_material_records = safe_load_records(cache_dir, "GMAT_path", running_config, path_config, ["floor", "ground"])
    fence_records = safe_load_records(cache_dir, "fence_path", running_config, path_config, ["fence", "wall"])
    anchor_records = safe_load_records(cache_dir, "anchor_path", running_config, path_config, [])
    outdoorart_records = safe_load_records(cache_dir, "outdoor_path", running_config, path_config, ["outdoorArt"])
    sky_records = safe_load_records(cache_dir, "sky_path", running_config, path_config, ["sky", "hdri"])
    surrounding_records = safe_load_records(cache_dir, "surrounding_path", running_config, path_config)
     # inject back to running_config
    C_record_items = safe_load_records(cache_dir, "C_item_path", running_config, path_config)

    running_config["A_record_items"] = A_record_items
    running_config["B_record_items"] = B_record_items
    running_config["ground_material_records"] = ground_material_records
    running_config["fence_records"] = fence_records
    running_config["anchor_records"] = anchor_records
    running_config["outdoorart_records"] = outdoorart_records
    running_config["sky_records"] = sky_records
    running_config["surrounding_records"] = surrounding_records
    running_config["C_record_items"] = C_record_items

     # number of scenes to generate
    NUMBER_OF_SCENES_TO_GENERATE = running_config.get("NUMBER_OF_SCENES_TO_GENERATE", 1)
    VIEWS_PER_SCENE = running_config.get("VIEWS_PER_SCENE", 8)
    RENDER_SAMPLES = args.samples or int(running_config.get("RENDER_SAMPLES", 256))
    RENDER_RESOLUTION = args.resolution or tuple(running_config.get("RENDER_RESOLUTION", [800, 600]))
    # Example 1) Fixed 1 views per scene
    try:
        generate_many_scenes(
            base_out_dir=scene_cache_dir,
            num_scenes=NUMBER_OF_SCENES_TO_GENERATE,
            views_per_scene=VIEWS_PER_SCENE,
            start_index=args.start_index,
            resolution=RENDER_RESOLUTION,
            samples=RENDER_SAMPLES,
            running_config=running_config,
        )
        bpy.ops.wm.quit_blender()
        sys.exit(0)
    except KeyboardInterrupt:
        bpy.ops.wm.quit_blender()
        sys.exit(0)
