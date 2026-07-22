import bpy
import os

def assign_instance_ids(
    self,
    start_id: int = 1,
    include_ground: bool = False,
    make_real: bool = True,
):
    """
    Assign a unique instance_id (Object Index / pass_index) to each placed asset root.

    One instance_id corresponds to one placed asset (root), and is shared by all
    renderable geometry belonging to that asset. This enables per-instance masks
    via the IndexOB render pass.

    This function correctly handles Collection Instances (Empty instancers):
    - Collection Instance children are not real objects in the scene hierarchy
    - Therefore, we optionally convert instances to real objects before assignment
      using bpy.ops.object.duplicates_make_real

    Parameters
    ----------
    start_id : int
        Starting value for instance IDs (must be > 0 for IndexOB).
    include_ground : bool
        Whether to include ground objects when assigning instance IDs.
    make_real : bool
        If True, Collection Instances are converted to real objects so that
        pass_index can be assigned per instance.
    """
    import bpy

    # Collect root objects (one root per placed asset)
    roots = [o[0] for o in self.objects.values() if o and o[0]]

    self.instance_id_map = {}
    cur = int(start_id)

    # Object types that actually render and therefore need pass_index
    render_types = {"MESH", "CURVE", "SURFACE", "META", "FONT"}

    # print("roots before filter:", [r.name for r in roots])

    def _copy_custom_props(src, dst):
        """
        Copy custom properties from src object to dst object.
        This is useful after making instances real, so that semantic metadata
        stored on the root is not lost.
        """
        for k in src.keys():
            if k == "_RNA_UI":
                continue
            try:
                dst[k] = src[k]
            except Exception:
                pass

    for root in roots:
        instance_id = int(cur)

        # Assign instance ID to root itself (harmless even if not renderable)
        root.pass_index = instance_id
        root["instance_id"] = instance_id

        # Record existing objects so we can detect newly created ones
        before_objects = set(bpy.data.objects)

        # If the root is a Collection Instance, its geometry does NOT exist
        # as real children yet. We must make instances real to assign pass_index.
        if (
            make_real
            and getattr(root, "instance_type", None) == "COLLECTION"
            and getattr(root, "instance_collection", None)
        ):
            bpy.ops.object.select_all(action="DESELECT")
            root.select_set(True)
            bpy.context.view_layer.objects.active = root
            try:
                bpy.ops.object.duplicates_make_real(
                    use_base_parent=True,
                    use_hierarchy=True,
                )
            except Exception:
                # Fail silently; downstream logic may still work for non-instanced roots
                pass

        after_objects = set(bpy.data.objects)
        created_objects = after_objects - before_objects

        # Assign instance_id to renderable descendants in the hierarchy
        # (covers normal imported assets and most make_real cases)
        for ch in root.children_recursive:
            if ch.type in render_types:
                ch.pass_index = instance_id
                ch["instance_id"] = instance_id

        # Assign instance_id to objects newly created by make_real
        # These may not always be direct children of the root
        for obj in created_objects:
            if obj.type in render_types:
                obj.pass_index = instance_id
                obj["instance_id"] = instance_id
                _copy_custom_props(root, obj)

        # Record mapping from root name to instance ID
        self.instance_id_map[root.name] = instance_id
        cur += 1

    return self.instance_id_map



def _setup_pass_outputs(self, out_basepath: str, enable_depth: bool = False, enable_seg: bool = False):
    """
    Setup compositor nodes to output:
      - Depth (Z) pass to <out_basepath>_depth.exr
      - Instance index (IndexOB) pass to <out_basepath>_seg.exr

    Notes:
      - Output as OpenEXR to preserve float/int-like values.
      - Requires objects have pass_index set (assign_instance_ids()).
    """
    if enable_depth is False or enable_seg is False:
        return  # Nothing to do
    scene = self.scene
    view_layer = bpy.context.view_layer

    # Enable passes
    if enable_depth:
        view_layer.use_pass_z = True
    if enable_seg:
        view_layer.use_pass_object_index = True

    # Enable compositor
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    # Nodes
    rlayers = tree.nodes.new(type="CompositorNodeRLayers")

    out_dir = os.path.dirname(out_basepath)
    stem = os.path.basename(out_basepath)

    # Depth output
    if enable_depth:
        depth_out = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_out.label = "DepthEXR"
        depth_out.base_path = out_dir  # we write absolute via file_slots path
        depth_out.format.file_format = "OPEN_EXR"
        depth_out.format.color_depth = "32"
        depth_out.format.exr_codec = "ZIP"
        depth_out.file_slots.clear()
        depth_out.file_slots.new("depth")
        depth_out.file_slots[-1].path = stem + "_depth"  # Blender will append frame; we render still => single file
        tree.links.new(rlayers.outputs["Depth"], depth_out.inputs[-1])

    # Segmentation (IndexOB) output
    if enable_seg:
        seg_socket = rlayers.outputs.get("IndexOB")
        if seg_socket is None:
            raise RuntimeError("No IndexOB output found. Check use_pass_object_index and pass_index assignment.")

        seg_out = tree.nodes.new(type="CompositorNodeOutputFile")
        seg_out.label = "SegEXR"
        seg_out.base_path = out_dir
        seg_out.format.file_format = "OPEN_EXR"
        seg_out.format.color_depth = "32"
        seg_out.format.exr_codec = "ZIP"

        seg_out.file_slots.clear()
        seg_out.file_slots.new("seg")
        seg_out.file_slots[-1].path = stem + "_seg"
        tree.links.new(seg_socket, seg_out.inputs[-1])


