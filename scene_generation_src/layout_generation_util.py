import math, random
from mathutils import Vector
import blender_utils as U
from mathutils import Euler

def build_rect_layout(self, grid_n: int, seed: int | None = None,
                      disable_rows_random: bool = True, disable_minmax=(0, 2),
                      row_padding: float = 0.0):
    """
    Build a 5-row rectangular layout:
      Row 0: A-row with n cells (length n * A)
      Row 1: B-row with ~n * A total length (m cells of size B, centered)
      Row 2: Empty middle strip, but put two A at the ends: total length (n+2) * A
      Row 3: B-row (same as Row 1)
      Row 4: A-row with n cells (length n * A)

    - A cell size: self.cell_size
    - B cell size: self.B_cell_size
    - Centers of A go to self.cell_centers; of B go to self.b_cell_centers
    - Randomly disable 1~2 rows among {0,1,3,4} if disable_rows_random=True

    Side effects:
      - self.span_x = n * self.cell_size                 # horizontal target span for A rows
      - self.span_y = 4 * row_pitch                      # vertical extent across 5 rows
      - self.cell_centers, self.b_cell_centers populated
    """
    A = float(self.cell_size)
    B = float(self.B_cell_size)
    n = int(max(1, grid_n))
    # use english for annotation in code
    # Horizontal total length reference: A row is n*A
    span_x = (n+2) * A
    # B row uses m B (as close to span_x as possible), centered
    m = max(1, int(round(n*A / B)))

    # Row pitch: to avoid overlap, take max(A, B) for row height; can add a bit of padding
    # row_pitch = max(A, B) + float(row_padding)
    span_y = 0 

    # 5 rows' y-coordinates (centered around middle row y=0)
    y_rows = [
        # +(A+B),         # Row 0 (top)   : A
        # +0.5*(B+A),     # Row 1         : B
        # 0.0,            # Row 2 (middle): empty but two A at ends
        # -0.5*(B+A),     # Row 3         : B
        # -(A+B),         # Row 4 (bottom): A
        +A+1.5*B,       # Row 0 (top)   : A
        +B+0.5*A,       # Row 1         : B
        0.0,            # Row 2 (middle): empty but two A at ends
        -B-0.5*A,       # Row 3         : B
        -A-1.5*B        # Row 4 (bottom): A
    ]

    # Rows to disable (cannot disable Row 2)
    rows_candidates = [0, 1, 3, 4]
    disabled = set()
    if disable_rows_random:
        k = random.randint(disable_minmax[0], disable_minmax[1])
        k = max(disable_minmax[0], min(disable_minmax[1], k))
        disabled = set(random.sample(rows_candidates, k))

    # Generate A rows: Row 0 & Row 4
    a_rows = [0, 4]
    a_centers = []
    for ridx in a_rows:
        if ridx in disabled:
            continue
        y = y_rows[ridx]
        span_y += A

        # x ∈ [-(n-1)/2 * A, ..., +(n-1)/2 * A]
        for i in range(n):
            x = (i - (n - 1) * 0.5) * A
            if ridx == 0:
                a_centers.append((x, y, 0)) 
            else: a_centers.append((x, y, 180)) # originally facing -Y. this row is at +Y. so rotate 180

    # Generate B rows: Row 1 & Row 3
    b_rows = [1, 3]
    b_centers = []
    for ridx in b_rows:
        if ridx in disabled:
            continue
        y = y_rows[ridx]
        span_y += B
        # m B, overall length ~ n*A, horizontally centered
        # x ∈ [-(m-1)/2 * B, ..., +(m-1)/2 * B]
        for j in range(m):
            x = (j - (m - 1) * 0.5) * B
            if ridx == 1:
                b_centers.append((x, y, 0)) 
            else: b_centers.append((x, y, 180)) # originally facing -Y. this row is at +Y. so rotate 180

    # Middle row Row 2: empty row, but place 1 A at each end
    # "Ends" refer to the two sides of the central empty strip of length n*A, each offset outward by 0.5*A
    # Total length is (n+2)*A, corresponding to two centers at ±((n/2 + 0.5) * A)
    # mid_y = y_rows[2]
    # left_x  = -((n * 0.5) + 0.5) * A
    # right_x = +((n * 0.5) + 0.5) * A
    # a_centers.append((left_x,  mid_y, 90))
    # a_centers.append((right_x, mid_y, 270))
    # span_y += A
    # # ---New---
    # # span_y = A+B+(A+B)+B+A
    # # Center is A+B now
    # span_y += B

    # Write back attributes
    self.span_x = span_x
    self.span_y = span_y
    self.span = max(self.span_x, self.span_y, self.span)
    self.cell_centers = a_centers           # A class centers
    self.B_cell_centers = b_centers         # B class centers

    # self.disabled_rows = sorted(list(disabled))  # Optional: record which rows are disabled

def build_linear_layout(self, grid_n: int, seed: int | None = None, 
                        only_active_a: bool = False,
                        row_padding: float = 0.0):
    """
    Build a 5-row rectangular layout:
      Row 0: A-row with n cells (length n * A)
      Row 1: B-row with ~n * A total length (m cells of size B, centered)


    - A cell size: self.cell_size
    - B cell size: self.B_cell_size
    - Centers of A go to self.cell_centers; of B go to self.b_cell_centers
    - Randomly disable 1~2 rows among {0,1,3,4} if disable_rows_random=True

    Side effects:
      - self.span_x = n * self.cell_size                 # horizontal target span for A rows
      - self.span_y = 4 * row_pitch                      # vertical extent across 5 rows
      - self.cell_centers, self.b_cell_centers populated
    """
    A = float(self.cell_size)
    B = float(self.B_cell_size)
    n = int(max(1, grid_n))
    # use english for annotation in code
    # Horizontal total length reference: A row is n*A
    span_x = n * A
    # B row uses m B (as close to span_x as possible), centered
    m = max(1, int(round(span_x / B)))

    # Row pitch: to avoid overlap, take max(A, B) for row height; can add a bit of padding
    # row_pitch = max(A, B) + float(row_padding)
    span_y = 0 

    # 5 rows' y-coordinates (centered around middle row y=0)
    y_rows = [
        +(A+B),   # Row 0 (top)   : A
        +0.5*(B+A),     # Row 1         : B
    ]

    # Rows to disable (cannot disable Row 2)
    disabled = [3,4]
    if only_active_a:
        disabled.append(1)

    # Generate A rows: Row 0 & Row 4
    a_rows = [0, 4]
    a_centers = []
    for ridx in a_rows:
        if ridx in disabled:
            continue
        y = y_rows[ridx]
        span_y += A

        # x ∈ [-(n-1)/2 * A, ..., +(n-1)/2 * A]
        for i in range(n):
            x = (i - (n - 1) * 0.5) * A
            if ridx == 0:
                a_centers.append((x, y, 0)) 
            else: a_centers.append((x, y, 180)) # originally facing -Y. this row is at +Y. so rotate 180

    # Generate B rows: Row 1 & Row 3
    b_rows = [1, 3]
    b_centers = []
    for ridx in b_rows:
        if ridx in disabled:
            continue
        y = y_rows[ridx]
        span_y += B
        # m B, overall length ~ n*A, horizontally centered
        # x ∈ [-(m-1)/2 * B, ..., +(m-1)/2 * B]
        for j in range(m):
            x = (j - (m - 1) * 0.5) * B
            if ridx == 1:
                b_centers.append((x, y, 0)) 
            else: b_centers.append((x, y, 180)) # originally facing -Y. this row is at +Y. so rotate 180

    # Write back attributes
    self.span_x = span_x
    self.span_y = span_y 
    self.span = max(self.span_x, self.span_y, self.span)
    self.cell_centers = a_centers           # A class centers
    self.B_cell_centers = b_centers         # B class centers


def build_b_on_a_layout(self, grid_n: int, seed: int = None, row_padding: float = 0.0):
    """
    Build B-class candidate cell centers on top of every placed A object.

    Args:
        grid_n (int): number of cells per side for the B grid (real_b_grid_n).
        seed (int, optional): RNG seed (kept for consistency, not strictly needed here).
        row_padding (float, optional): shrink A's usable XY half-extent by this margin (in scene units).
                                       This helps keep B cells away from A's edges.
    Side effects:
        - Populates `self.B_cell_centers` with a flat list of (x, y, None) tuples.
    """

    if seed is not None:
        random.seed(seed)

    if not hasattr(self, "objects") or not isinstance(self.objects, dict):
        raise RuntimeError("self.objects must be a dict of {name: (root, children)}.")

    if not hasattr(self, "B_cell_size") or self.B_cell_size is None:
        raise RuntimeError("self.B_cell_size must be set before building B-on-A layout.")

    cell_size = float(self.B_cell_size)
    if cell_size <= 0:
        raise ValueError("self.B_cell_size must be > 0.")

    # Accumulate all valid B centers here
    B_centers = []

    # Helper: return (center_xy, half_extent_xy) for an A root
    def _a_xy_bounds(root):
        # Prefer cached size by asset_code
        asset_code = root.get("asset_code", None)
        size_vec = None
        if asset_code is not None:
            size_vec = self.object_bound_dict.get(asset_code, None)

        # # Fallback to visual bounds if cache miss (e.g., asset reused under a different code)
        # if size_vec is None:
        #     _, size_vec = U.get_visual_bounds([root])

        cx, cy = float(root.location.x), float(root.location.y)
        # World-aligned AABB half extents in XY from the measured size
        half_x = 0.5 * float(size_vec.x)
        half_y = 0.5 * float(size_vec.y)

        # Apply optional padding (shrink usable area)
        half_x = max(0.0, half_x - float(row_padding))
        half_y = max(0.0, half_y - float(row_padding))
        return (cx, cy), (half_x, half_y)

    # Identify all A objects by naming convention; we accept any name containing "_A"
    a_roots = []
    for name, (root, _children) in self.objects.items():
        if "_A" in name:
            a_roots.append(root)

    # Build grids over each A root
    for root in a_roots:
        (ax, ay), (half_x, half_y) = _a_xy_bounds(root)

        if abs(half_x - half_y) < 0.5:
            #  in case it is a round object.
            half_x = half_x / math.sqrt(2)
            half_y = half_y / math.sqrt(2)

        # Grid geometry centered at (ax, ay)
        grid_span = grid_n * cell_size
        x0 = ax - 0.5 * grid_span + 0.5 * cell_size
        y0 = ay - 0.5 * grid_span + 0.5 * cell_size

        # To ensure the *entire* B cell lies within A's footprint,
        # the B center must stay within (half_extent - 0.5*cell_size)
        safe_half_x = max(0.0, half_x - 0.5 * cell_size)
        safe_half_y = max(0.0, half_y - 0.5 * cell_size)

        # Generate candidate centers and keep only those fully inside A
        for r in range(grid_n):
            for c in range(grid_n):
                x = x0 + c * cell_size
                y = y0 + r * cell_size
                if (abs(x - ax) <= safe_half_x) and (abs(y - ay) <= safe_half_y):
                    B_centers.append((x, y, None))

    # Save out
    self.B_cell_centers = B_centers


def place_c_on_a_or_b(
    self,
    C_record_items: list[dict],
    place_count_C: int,
    row_padding: float = 0.0,
    max_tries_per_obj: int = 30,
    host_class: str = "A",
    name_prefix: str = "object_C",
):
    """
    Place C objects randomly inside the XY footprint of a randomly chosen A or B host object.

    - Host candidates: any object name containing "_A" or "_B" in self.objects
    - Placement: random (x,y) within host usable region computed by _host_xy_bounds (same idea as _a_xy_bounds)
    - Rotation: random deg (four_angles_deg if available else uniform)
    - Collision: avoid overlap among C objects using XY AABB overlap only (fast, simple)
    """
    print(F"[INFO] START placing C on {host_class} hosts...")
    if not C_record_items or place_count_C <= 0:
        return self

    if not hasattr(self, "objects") or not isinstance(self.objects, dict):
        raise RuntimeError("self.objects must be a dict of {name: (root, children)}.")

    # 1) collect host roots (A or B)
    host_roots = []
    if host_class == "A":
        for name, (root, _children) in self.objects.items():
            if "_A" in name and not "_C" in name:
                host_roots.append(root)
    elif host_class == "B":
        for name, (root, _children) in self.objects.items():
            if "_B" in name and not "_C" in name:
                host_roots.append(root)
    else:  # both A and B   
        for name, (root, _children) in self.objects.items():
            if (("_A" in name) or ("_B" in name)) and not "_C" in name:
                host_roots.append(root)

    if not host_roots:
        print("[place_c_on_a_or_b] No A/B hosts found. Skip placing C.")
        return self

    # 2) helper: host bounds in XY (same concept as layout_generation_util._a_xy_bounds)
    def _host_xy_bounds(root):
        asset_code = root.get("asset_code", None)
        size_vec = self.object_bound_dict.get(asset_code, None)

        if size_vec is None:
            # if cache miss, measure once (doesn't update focus)
            _, size_vec = U.get_visual_bounds([root])
            if asset_code is not None:
                self.object_bound_dict[asset_code] = size_vec

        cx, cy = float(root.location.x), float(root.location.y)
        half_x = 0.5 * float(size_vec.x)
        half_y = 0.5 * float(size_vec.y)

        # optional padding: shrink usable region
        half_x = max(0.0, half_x - float(row_padding))
        half_y = max(0.0, half_y - float(row_padding))

        # round-ish heuristic (kept consistent with your existing b_on_a_layout)
        if abs(half_x - half_y) < 0.5:
            half_x = half_x / math.sqrt(2)
            half_y = half_y / math.sqrt(2)

        return (cx, cy), (half_x, half_y)

    # 3) helper: pick rotation deg
    def _pick_deg():
        # print(self.four_angles_deg)
        if getattr(self, "four_angles_deg", None):
            # print("Has degree")
            return float(random.choice(self.four_angles_deg))
        # print("No degree")
        return float(random.choice(range(0, 360, 90)))

    # 4) helper: XY AABB overlap test
    # box = (x1, y1, x2, y2)
    def _overlap_xy(b1, b2):
        return (b1[0] < b2[2] and b1[2] > b2[0] and b1[1] < b2[3] and b1[3] > b2[1])

    placed_c_boxes = []  # keep XY AABBs of already placed C

    # 5) place each C
    for k in range(place_count_C):
        ok = False

        for attempt in range(max_tries_per_obj):
            host = random.choice(host_roots)
            (hx, hy), (h_half_x, h_half_y) = _host_xy_bounds(host)

            it = random.choice(C_record_items)
            blend_path = it.get("dst")
            if not blend_path:
                continue
            scale_size = float(it.get("scale", 1.0) or 1.0)

            obj_name = self._unique_name(f"{host.name}_under_{name_prefix}", mode=None)
            grp_list = self._import_asset_from_path(blend_path, name_hint=obj_name)
            if not grp_list:
                continue
            root = grp_list[0]
            root.scale = (scale_size, scale_size, scale_size)

            angle_deg = _pick_deg()
            # print(angle_deg)
            # print(f"Name: {obj_name}, Host: {host.name}, Try: {attempt+1}, Rot_deg: {angle_deg:.1f}")
            # set rotation early because size depends on rotation
            root.rotation_euler = Euler((0.0, 0.0, math.radians(angle_deg)), 'XYZ')
            root.location = (0.0, 0.0, -100.0)  # out of sight before final placement

            # measure C size (do NOT update focused_objects_AABB during measure)
            try:
                _center_tmp, c_size = self.get_bound_update_AABB(it, root, deg=None, focus_on_object=False)
            except Exception:
                _center_tmp, c_size = U.get_visual_bounds([root])

            c_w = float(c_size.x)
            c_h = float(c_size.y)

            # compute safe host half extents so that C's full AABB stays inside host
            safe_half_x = h_half_x - 0.5 * c_w
            safe_half_y = h_half_y - 0.5 * c_h
            if safe_half_x <= 1e-6 or safe_half_y <= 1e-6:
                # host too small for this C; cleanup and retry
                self.cleanup_objects(temp_handlers=[(root, [])])
                continue

            # sample random xy inside safe region
            x = random.uniform(hx - safe_half_x, hx + safe_half_x)
            y = random.uniform(hy - safe_half_y, hy + safe_half_y)

            new_box = (x - 0.5 * c_w, y - 0.5 * c_h, x + 0.5 * c_w, y + 0.5 * c_h)

            # collision check against previous Cs
            collide = any(_overlap_xy(new_box, b) for b in placed_c_boxes)
            if collide:
                self.cleanup_objects(temp_handlers=[(root, [])])
                continue

            # final place (raycast + drop)
            self.place_asset_by_xy(root, x, y, angle_deg)

            # now update AABB focus with deg (C should be part of focus by default)
            self.get_bound_update_AABB(it, root, deg=angle_deg, focus_on_object=True)

            # save metadata + track
            self.save_root_metadata(
                root, obj_name, blend_path, scale_size, angle_deg,
                asset_code=it.get("asset_code"),
                facing_value=it.get("facing", 0.0),
            )
            self.objects[obj_name] = (root, [])

            placed_c_boxes.append(new_box)
            ok = True
            break

        if not ok:
            print(f"[place_c_on_a_or_b] Failed placing C #{k+1} after {max_tries_per_obj} tries.")
        else:
            print(f"[place_c_on_a_or_b] Placed C #{k+1} successfully.")

    return self

