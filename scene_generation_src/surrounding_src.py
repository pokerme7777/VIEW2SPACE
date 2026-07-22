import bpy
import random
import math
from mathutils import Euler
import blender_utils as U

def place_surroundings(self,
                        running_config: dict | None = None,
                        name_prefix: str = "surrounding"):
    """
    Surrounding placement:
        - running_config["surrounding_records"] should be list[dict]
        each dict has keys like 'dst','scale','asset_code'.
        - Places items in a single ring around the current scene rectangle,
        filling each side sequentially (bottom -> right -> top -> left).
        - For each picked item: choose rotation (store it), measure its XY
        size via get_bound_update_AABB(..., focus_on_object=False),
        then compute center coords so the object lies just outside the scene
        boundary (half_side + half_size + clearance).
        - DOES NOT call update_focused_object_AABB (focus_on_object=False).
        - Minimal changes to existing data structures: items are saved with
        save_root_metadata and tracked in self.objects (remove if undesired).
    """
    if running_config is None:
        return self
    surrounding_items = running_config.get("surrounding_records", None)
    print(f"Placing surroundings, found {len(surrounding_items) if surrounding_items else 0} candidate items.")

    clearance = running_config.get("surrounding_clearance", 3)
    margin_between = running_config.get("surrounding_margin_between", 0.5)

    # Optional fence thickness if exists (push further outside)
    fence_off = getattr(self, "fence_thickness", 0.0) * 0.5
    outer_clearance = clearance + fence_off


    if not surrounding_items:
        return self

    # Determine half-lengths of scene on X and Y
    # Prefer explicit attributes if present: self.side_len or self.span_x/span_y
    if hasattr(self, "side_len") and getattr(self, "side_len"):
        half_x = half_y = 0.5 * float(self.side_len) + outer_clearance
    else:
        # try span_x/span_y set by layout util
        half_x = 0.5 * getattr(self, "span_x", getattr(self, "span", 0.0)) + outer_clearance
        half_y = 0.5 * getattr(self, "span_y", getattr(self, "span", 0.0)) + outer_clearance
        # fallback if zero
        if half_x == 0.0 and hasattr(self, "span"):
            half_x = half_y = 0.5 * float(self.span) + outer_clearance

    if half_x == 0.0 and half_y == 0.0:
        # nothing to surround
        return self

    # choose candidate rotation set
    four_angles = getattr(self, "four_angles_deg", None)
    # simple helper to choose angle and remember it in record copy
    def pick_angle(rec):
        if four_angles:
            return float(random.choice(four_angles))
        # else allow any multiple of 5 degrees for some variety
        return float(random.randrange(0, 360, 90))

    # Prepare cursors for four sides.
    # We'll fill each side sequentially. For each side we maintain a cursor
    # that is the center coordinate along the side's axis (in local side axes).
    # For bottom/top sides the axis is X; for left/right it's Y.
    side_list = ["bottom", "right", "top", "left"]

    # For each side we will compute a list of placements until that side is full.
    # We'll pick items randomly and place them. To avoid infinite loops when
    # very large objects exist, we will bail out if we try too many times for a side.
    max_attempts_per_side = max(50, len(surrounding_items) * 5)

    # Helper to import, scale, measure, and optionally cleanup (we keep placed object in scene)
    for side in side_list:
        # initialize cursor at the inner start of the side (from -half to +half)
        if side == "bottom":
            axis_start = -half_x
            axis_end = half_x
            axis_dir = +1  # move increasing X
        elif side == "top":
            axis_start = half_x
            axis_end = -half_x
            axis_dir = -1  # move decreasing X
        elif side == "left":
            axis_start = half_y
            axis_end = -half_y
            axis_dir = -1  # move decreasing Y (from +half to -half)
        else:  # right
            axis_start = -half_y
            axis_end = half_y
            axis_dir = +1  # move increasing Y

        # place first item at axis cursor = axis_start + first_half_size_along_axis
        cursor = axis_start
        placed_on_side = 0
        attempts = 0

        while attempts < max_attempts_per_side:
            attempts += 1
            it = random.choice(surrounding_items)
            path = it.get("dst")
            if not path:
                continue
            scale_size = float(it.get("scale", 1.0) or 1.0)
            angle_deg = pick_angle(it)

            # import asset
            obj_name = self._unique_name(name_prefix, mode=None)
            grp_list = self._import_asset_from_path(path, name_hint=obj_name)
            if not grp_list:
                continue
            root = grp_list[0]
            # apply scale and rotation BEFORE measuring, because measurement depends on world orientation
            root.scale = (scale_size, scale_size, scale_size)

            # measure bounds (do not update focused_objects_AABB)
            _, size_vec = self.get_bound_update_bound_dict(it, root)
            if angle_deg % 180 == 0:
                pass  # size_vec is correct
            else:
                # swap X and Y sizes
                size_vec1 = size_vec.copy()
                size_vec.x, size_vec.y = size_vec1.y, size_vec1.x
            obj_w = float(size_vec.x)   # extent along world X
            obj_h = float(size_vec.y)   # extent along world Y

            # rotate now (place_asset_by_xy will set the final rotation again, but measuring needs rot)
            root.rotation_euler = Euler((0.0, 0.0, math.radians(angle_deg)), 'XYZ')
            # put it high and out of sight so measurement works and raycast will be used for final placement
            root.location = (0.0, 0.0, -100.0)


            # compute center position depending on side
            if side == "bottom":
                y = -(half_y + obj_h * 0.5)
                # next cursor needs to be increased by half widths
                # the center x for this object is cursor + axis_dir * (obj_w * 0.5)
                x = cursor + axis_dir * (obj_w * 0.5 + margin_between)
            elif side == "top":
                y = +(half_y + obj_h * 0.5)
                x = cursor + axis_dir * (obj_w * 0.5 + margin_between)
            elif side == "left":
                x = -(half_x + obj_w * 0.5)
                y = cursor + axis_dir * (obj_h * 0.5 + margin_between)
            else:  # right
                x = +(half_x + obj_w * 0.5)
                y = cursor + axis_dir * (obj_h * 0.5 + margin_between)

            # Check if the center along axis would lie beyond the other end (overflow)
            # For bottom/top axis is X; for left/right axis is Y
            axis_coord = x if side in ("bottom", "top") else y
            axis_end_coord = axis_end
            # Because axis_dir encodes sign increment, check crossing
            if axis_dir > 0:
                beyond = axis_coord - axis_end_coord > obj_w 
            else:
                beyond = axis_end_coord - axis_coord > obj_w

            # finalize placement: use place_asset_by_xy for proper Z and drop
            self.place_asset_by_xy(root, x, y, angle_deg)

            # self.objects[obj_name] = (root, [])  # only for visibility tracking  demo VIS

            # save metadata (asset_code may be present in record)
            # asset_code = it.get("asset_code", f"sur_{placed_on_side}")
            # self.save_root_metadata(root, obj_name, path, scale_size, angle_deg, asset_code=asset_code, facing_value=it.get("facing", 0.0))

            # # record in objects tracking (background but still kept)
            # children = [c for c in bpy.data.objects if getattr(c, "parent", None) == root]
            # self.objects[obj_name] = (root, children)

            root["is_background"] = True

            # advance cursor by half(prev) + half(current)
            # compute advance: move cursor by axis_dir * (obj_size_along_axis)
            if side in ("bottom", "top"):
                # advance by full width (we already placed at cursor + half_width)
                cursor = cursor + axis_dir * obj_w
            else:
                cursor = cursor + axis_dir * obj_h

            placed_on_side += 1

            print(f"placing surrounding '{obj_name}' on {side} at ({x:.2f},{y:.2f}) size ({obj_w:.2f},{obj_h:.2f}) angle {angle_deg:.1f}°")

            if beyond:
                # side is full: cleanup the imported template we didn't use and break to next side
                # remove the last imported root (not placed)
                self.cleanup_objects(temp_handlers=[(root, [])])
                break
        # end while for this side; move to next side

    # done all sides
    return self
