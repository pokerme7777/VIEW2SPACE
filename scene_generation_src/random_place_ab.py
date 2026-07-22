from random import choice, random, uniform
import blender_utils as U
from mathutils import Euler
import math


def place_asset_random_ab(
    self,
    A_record_items: list[dict],
    B_record_items: list[dict] | None = None,
    place_count_A: int = 1,
    place_count_B: int | None = None,
    max_tries_per_obj: int = 50,
    GRID_N: float = 3,
    keep_center: bool = False,
    keep_center_value: int = 1,
    radius_growth: float = 1.2,
    four_angles_deg: list[float] | None = None,
    name_prefix: str = "object",
):
    """
    Random placement for A and B (no grid, no cell).

    Rules:
    - Each object is placed at a random (x,y)
    - True object size comes from get_bound_update_bound_dict
    - Reject placement if XY AABB overlaps with any previous object
    - Sampling radius grows automatically if space becomes crowded
    - self.span is updated from final placed extents
    """

    if not A_record_items:
        raise RuntimeError("A_record_items is empty")

    if B_record_items is None:
        B_record_items = []

    if place_count_B is None:
        place_count_B = place_count_A

    if four_angles_deg is None:
        four_angles_deg = [0.0, 90.0, 180.0, 270.0]

    if keep_center:
        keep_center_value = float(keep_center_value)
    else:
        keep_center_value = 0.0

    placed_boxes: list[tuple[float, float, float, float]] = []

    # -----------------------------
    # helpers
    # -----------------------------

    def overlap_xy(b1, b2):
        return (
            b1[0] < b2[2] and b1[2] > b2[0] and
            b1[1] < b2[3] and b1[3] > b2[1]
        )

    def sample_xy(radius, keep_center_value):
        """
        Sample (x,y) uniformly in a square [-radius, radius]^2
        but reject points inside the central square
        [-keep_center_value, keep_center_value]^2.
        """
        if keep_center_value <= 0.0:
            return (
                uniform(-radius, radius),
                uniform(-radius, radius),
            )

        for _ in range(20):  # fast rejection
            x = uniform(-radius, radius)
            y = uniform(-radius, radius)

            # reject central forbidden region
            if abs(x) < keep_center_value and abs(y) < keep_center_value:
                continue

            return (x, y)

        # fallback: force point on the ring boundary
        sign_x = 1.0 if random() > 0.5 else -1.0
        sign_y = 1.0 if random() > 0.5 else -1.0
        if random() > 0.5:
            return (sign_x * keep_center_value, uniform(-radius, radius))
        else:
            return (uniform(-radius, radius), sign_y * keep_center_value)

    def pick_deg():
        return float(choice(four_angles_deg))

    # -----------------------------
    # core placement loop
    # -----------------------------
    self.cell_size = max(self._estimate_max_diameter_from_samples(A_record_items, max_samples=max(1, len(A_record_items))),
                               self._estimate_max_diameter_from_samples(B_record_items, max_samples=max(1, len(B_record_items))))
    
    init_radius = GRID_N * self.cell_size
    
    def place_many(record_items, count, suffix):
        nonlocal placed_boxes

        radius = float(init_radius)

        for idx in range(count):
            placed = False

            for attempt in range(max_tries_per_obj):
                record = choice(record_items)
                blend_path = record.get("dst")
                if not blend_path:
                    continue

                scale = float(record.get("scale", 1.0) or 1.0)
                deg = pick_deg()

                obj_name = self._unique_name(f"{name_prefix}_{suffix}", mode=None)
                grp = self._import_asset_from_path(blend_path, name_hint=obj_name)
                if not grp:
                    continue

                root = grp[0]
                root.scale = (scale, scale, scale)
                root.rotation_euler = Euler((0.0, 0.0, math.radians(deg)), "XYZ")
                root.location = (0.0, 0.0, -100.0)

                # --- true size ---
                _, size_vec = self.get_bound_update_bound_dict(record, root, deg=deg)
                w = float(size_vec.x)
                h = float(size_vec.y)

                x, y = sample_xy(radius, keep_center_value)

                new_box = (
                    x - 0.5 * w,
                    y - 0.5 * h,
                    x + 0.5 * w,
                    y + 0.5 * h,
                )

                if any(overlap_xy(new_box, b) for b in placed_boxes):
                    self.cleanup_objects(temp_handlers=[(root, [])])
                    continue

                # final placement
                self.place_asset_by_xy(root, x, y, deg)

                # update focus + bounds
                self.get_bound_update_AABB(record, root, deg=deg, focus_on_object=True)

                self.save_root_metadata(
                    root,
                    obj_name,
                    blend_path,
                    scale,
                    deg,
                    asset_code=record.get("asset_code"),
                    facing_value=record.get("facing", 0.0),
                )

                self.objects[obj_name] = (root, [])

                placed_boxes.append(new_box)
                placed = True
                break

            if not placed:
                # space too crowded → expand sampling radius
                # radius *= radius_growth
                print(
                    # f"[place_asset_random_ab] expand radius to {radius:.2f} "
                    f"(failed placing {suffix}_{idx})"
                )

    # -----------------------------
    # place A then B
    # -----------------------------

    place_many(A_record_items, place_count_A, "A")
    if B_record_items:
        place_many(B_record_items, place_count_B, "B")

    # -----------------------------
    # update span from true extents
    # -----------------------------

    self.span_x = 0.0
    self.span_y = 0.0

    for (x1, y1, x2, y2) in placed_boxes:
        self.span_x = max(self.span_x, abs(x1), abs(x2))
        self.span_y = max(self.span_y, abs(y1), abs(y2))

    self.span = 2 * init_radius
    self.side_len = self.span

    return self

