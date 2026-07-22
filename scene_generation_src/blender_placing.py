
import bpy
import sys
import os
import json
import math
from mathutils import Vector
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import blender_utils as U


def raycast_down_from_world_xy(x, y, start_z=20.0, max_dist=50.0):
    """Cast a ray straight down in world space from (x, y, +start_z)."""
    deps = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    origin = Vector((x, y, start_z))
    direction = Vector((0, 0, -1))
    return scene.ray_cast(deps, origin, direction, distance=max_dist)

def random_generate_non_overlapping_points(n, radius=3.0, min_distance=0.8, positions=None):
    if positions is None:
        positions = []
    while len(positions) < n:
        x, y = random.uniform(-radius, radius), random.uniform(-radius, radius)
        if all((x - px)**2 + (y - py)**2 > min_distance**2 for px, py in positions):
            positions.append((x, y))
    return positions

# ------------------------------
# Variable-radius placement utils
# ------------------------------

def footprint_radius_basic(shape: str, size: float) -> float:
    """
    Estimate XY footprint radius for each primitive (match create_colored_object()).
    - cube: side = 2*size  -> circumscribed circle radius ≈ size * sqrt(2)
    - uv_sphere: radius = size
    - cylinder: base radius = size
    - cone: base radius1 = size  (we place on base)
    - torus: major=size, minor=0.3*size -> outer radius ≈ size + 0.3*size
    """
    shape = shape.lower()
    if shape == "cube":
        return size * math.sqrt(2.0)
    elif shape in ("uv_sphere", "cylinder", "cone"):
        return size
    elif shape == "torus":
        return size * (1.0 + 0.3)
    else:
        return size  # fallback
        

def sample_points_variable_radii(
    radii: list[float],
    arena_radius: float,
    border: float = 0.2,
    max_tries_per_point: int = 400,
    relax_iters: int = 8,
    relax_step: float = 0.45,
):
    """
    Poisson-like dart throwing for circles with different radii inside a disk.
    Returns list[(x,y)] aligned with 'radii' order.
    """
    pts: list[tuple[float, float, float]] = []  # (x,y,r)
    disk_r = max(0.01, arena_radius - border)

    for r in radii:
        ok = False
        tries = 0
        while not ok and tries < max_tries_per_point:
            # uniform in disk (sqrt trick)
            t = random.random() * 2 * math.pi
            rad = (disk_r - r) * math.sqrt(random.random())
            x = rad * math.cos(t)
            y = rad * math.sin(t)
            if all((x - px) ** 2 + (y - py) ** 2 >= (r + pr) ** 2 for (px, py, pr) in pts):
                pts.append((x, y, r))
                ok = True
            else:
                tries += 1
        if not ok:
            # greedy fallback: try to place anyway (may overlap)
            for _ in range(max_tries_per_point):
                t = random.random() * 2 * math.pi
                rad = (disk_r - r) * math.sqrt(random.random())
                x = rad * math.cos(t)
                y = rad * math.sin(t)
                # allow small overlaps
                if all((x - px) ** 2 + (y - py) ** 2 >= (r + pr - 1e-3 * disk_r) ** 2 for (px, py, pr) in pts):
                    pts.append((x, y, r))
                    ok = True
                    break
            if not ok:
                # last resort: place at center (will definitely overlap)
                pts.append((0.0, 0.0, max(0.5 * r, 1e-3)))

    # small Lloyd-like relaxation
    for _ in range(relax_iters):
        new_pts = []
        for i, (xi, yi, ri) in enumerate(pts):
            dx = dy = 0.0
            for j, (xj, yj, rj) in enumerate(pts):
                if i == j: continue
                vx = xi - xj; vy = yi - yj
                d = math.hypot(vx, vy) + 1e-9
                target = ri + rj
                overlap = target - d
                if overlap > 0:  # push away
                    s = (overlap / d) * 0.5 * relax_step
                    dx += vx * s
                    dy += vy * s
            # keep-inside constraint
            nx = xi + dx; ny = yi + dy
            rr = math.hypot(nx, ny)
            max_rr = max(1e-6, disk_r - ri)
            if rr > max_rr:
                nx *= max_rr / rr
                ny *= max_rr / rr
            new_pts.append((nx, ny, ri))
        pts = new_pts

    return [(x, y) for (x, y, r) in pts]

def sample_point_avoiding_obstacles(
    r_target: float,
    obstacles: list[tuple[float, float, float]],  # [(x,y,r), ...] existing objects with footprints
    arena_radius: float,
    border: float = 0.2,
    max_tries: int = 800,
) -> tuple[float, float]:
    """
    Dart-throw a single XY for a circle of radius r_target inside a disk arena,
    avoiding overlaps with 'obstacles' (each is (x,y,r)). Returns (x,y).
    """
    disk_r = max(0.01, arena_radius - border)
    for _ in range(max_tries):
        t = random.random() * 2 * math.pi
        rad = (disk_r - r_target) * math.sqrt(random.random())
        x = rad * math.cos(t)
        y = rad * math.sin(t)
        ok = True
        for (ox, oy, orad) in obstacles:
            if (x - ox) * (x - ox) + (y - oy) * (y - oy) < (r_target + orad) * (r_target + orad):
                ok = False
                break
        if ok:
            return x, y
    # last-resort: return center (may overlap; caller can retry/relax)
    return 0.0, 0.0

# =========================================================================
# Object placement utilities
# =========================================================================
def drop_to_ground(obj, ground_z: float, clearance: float = 0.0):
    """
    Snap the given object so that its lowest point sits exactly on the ground plane.

    Parameters
    ----------
    obj : bpy.types.Object or bpy.types.InstanceObject
        The object to be adjusted.
    ground_z : float
        The target ground height (e.g. from raycast hit location).
    clearance : float, optional
        Extra offset above the ground. Useful to leave a small gap (e.g. 0.001 for 1 mm).
        Default is 0.0 (object touches the ground).
    
    Notes
    -----
    - Works regardless of the object's origin location.
    - Evaluates modifiers and transformations before computing the world bounding box.
    - If the object is already below the ground, it will move upwards.
      If it floats above, it will move downwards.
    """

    deps = bpy.context.evaluated_depsgraph_get()

    # if obj.type == 'EMPTY' and getattr(obj, "instance_type", None) == 'COLLECTION':
    #     # For collection instances, compute bounds of all child objects
    #     center, size = U.get_visual_bounds([obj])
    #     min_z = center.z - 0.5 * size.z
    #     dz = (ground_z + clearance) - min_z
    #     obj.location.z += dz
    #     return dz
    
    obj_eval = obj.evaluated_get(deps)  # Apply modifiers for accurate geometry

    # Compute the object's world-space bounding box corners
    world_corners = [obj_eval.matrix_world @ Vector(corner) for corner in obj_eval.bound_box]

    # Find the minimum Z (lowest point of the object)
    min_z = min(v.z for v in world_corners)

    # How much to move: target ground height minus current lowest point
    dz = (ground_z + clearance) - min_z

    # Apply the adjustment
    obj.location.z += dz
    return dz  # return the applied shift for reference


# --------------------------------------------------------------------
# Calculations for camera placement to cover a scene
# --------------------------------------------------------------------

def camera_fov_xy(cam: bpy.types.Object, scene: bpy.types.Scene):
    """Return (fov_x, fov_y) in radians for current camera data & render aspect."""
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


def scene_center_and_radius(objs, exclude_names=frozenset()):
    """Compute (center, radius_xy) from a set of objects, ignoring excluded ones."""
    use = [o for o in objs if o.type == 'MESH' and o.name not in exclude_names]
    if not use:
        return Vector((0,0,0)), 1.0
    center, size = U.get_bounds(use)
    r = 0.5 * max(size.x, size.y)
    return center, max(r, 1e-4)  # guard against zero


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
