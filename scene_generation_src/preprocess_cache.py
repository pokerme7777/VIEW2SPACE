import os
import json
import shutil
from pathlib import Path

def preprocess_cache(source_dir, target_dir, mapping_path, id_prefix=None, scale_value=1.0, facing_value=0.0):
    os.makedirs(target_dir, exist_ok=True)

    # load existing mapping
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            existing = [json.loads(line) for line in f if line.strip()]
    else:
        existing = []

    # build lookup for skip
    processed = {(x["old_folder"]) for x in existing}
    next_id = max([int(x.get("asset_code").split("_")[-1]) for x in existing], default=0) + 1

    for folder in os.listdir(source_dir):
        subdir = os.path.join(source_dir, folder)
        if not os.path.isdir(subdir):
            continue

        for file in os.listdir(subdir):
            is_blend = False
            is_exr = False
            is_blend = file.endswith(".blend")
            is_exr = file.endswith(".exr")
            if not (is_blend or is_exr) or "_" not in file:
                continue
            if (folder) in processed:
                continue

            src = os.path.join(subdir, file)
            base = file.split("_")[0]
            if is_blend:
                dst_name = base + ".blend"
            else:  # is_exr
                dst_name = base + ".exr"
            dst = os.path.join(target_dir, dst_name)

            # avoid overwrite
            i = 1
            while os.path.exists(dst):
                if is_blend:
                    dst_name = f"{base}_{i}.blend"
                else:  # is_exr
                    dst_name = f"{base}_{i}.exr"
                dst = os.path.join(target_dir, dst_name)
                i += 1

            shutil.copy2(src, dst)

            rec = {
                "asset_code": f"{id_prefix}_{next_id:02d}",
                "scale": scale_value,
                "new_file": dst_name,
                "dst": dst,
                "old_folder": folder,
                "facing": facing_value
            }
            existing.append(rec)
            next_id += 1

    # write back
    with open(mapping_path, "w", encoding="utf-8") as f:
        for x in existing:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"Processed {next_id - 1} assets, mapping saved to {mapping_path}")

def load_json(path):
    """Load a JSON file and return a Python object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Normalize downloaded BlenderKit assets and write mapping.jsonl.")
    parser.add_argument('-S', '--source-name', type=str, required=True,
                        help='Downloaded folder name under kit_cache, without the _cache suffix.')
    parser.add_argument('--target-name', type=str, default=None,
                        help='Processed asset folder/id prefix. Defaults to asset_scale_config folder_transit.')
    parser.add_argument('--data-root', type=str, default=os.environ.get("BENCHMARKING_DATA_CACHE"),
                        help='Root containing kit_cache/ and processed_asset/.')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to asset_scale_config.json.')
    parser.add_argument('--scale', type=float, default=None,
                        help='Override scale value written to mapping.jsonl.')
    parser.add_argument('--facing', type=float, default=None,
                        help='Override facing value written to mapping.jsonl.')

    args = parser.parse_args()

    BLENDER_FOLDER = args.data_root
    if BLENDER_FOLDER is None:
        raise RuntimeError("Set BENCHMARKING_DATA_CACHE or pass --data-root.")
    package_root = Path(__file__).resolve().parent
    CONFIG_PATH = args.config or os.path.join(package_root, "config/asset_scale_config.json")
    config = load_json(CONFIG_PATH)
    current_process_dir = f"{args.source_name}_cache"  # Change this to process different directories
    transit_name = args.target_name or config.get("folder_transit", {}).get(current_process_dir)
    if not transit_name:
        raise RuntimeError(f"No target name configured for {current_process_dir}. Pass --target-name.")
    scale_value = args.scale if args.scale is not None else config.get("scales", {}).get(transit_name, 1.0)
    facing_value = args.facing if args.facing is not None else config.get("facing", {}).get(transit_name, 0.0)

    preprocess_cache(
        source_dir=os.path.join(BLENDER_FOLDER, f"kit_cache/{current_process_dir}"),
        target_dir=os.path.join(BLENDER_FOLDER, f"processed_asset/{transit_name}"),
        mapping_path=os.path.join(BLENDER_FOLDER, f"processed_asset/{transit_name}/mapping.jsonl"),
        id_prefix=transit_name, scale_value=scale_value, facing_value=facing_value
    )
