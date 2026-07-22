import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def preview_all_pngs(folder, not_title=False):
    """
    Read all PNG files in the folder, display them in a grid with their filenames as labels,
    and save the combined image as preview_all.png in the same folder.
    The grid layout is chosen to be as square as possible (e.g., 3x3, 3x4, 4x4).
    """
    png_files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
    preview_all_png_num = sum([1 for fname in png_files if "preview_all" in fname])
    
    display_files = [fname for fname in png_files if "preview_all" not in fname]
    n = len(display_files)
    if n == 0:
        print("No PNG files found.")
        return

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    if cols * (rows - 1) >= n:
        rows -= 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for idx, fname in enumerate(display_files):
        img = mpimg.imread(os.path.join(folder, fname))
        axes[idx].imshow(img)
        # Set title closer to image
        if not not_title:
            axes[idx].set_title(fname, fontsize=10, pad=-20)  # pad=-5 makes the label closer
        axes[idx].axis('off')

    for ax in axes[n:]:
        ax.axis('off')
        
    plt.subplots_adjust(hspace=0.15)  # Reduce vertical space between images and titles
    if not_title:
        out_path = os.path.join(folder, "preview_all.png")
    else:
        out_path = os.path.join(folder, "preview_all_withtitle.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved preview image to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combine rendered asset previews into overview sheets.")
    parser.add_argument('-S', '--source-name', type=str, default=None,
                        help='Asset theme name under processed_asset/<name>_preview.')
    parser.add_argument('--folder', type=str, default=None,
                        help='Preview folder path. Overrides --source-name.')
    parser.add_argument('--data-root', type=str, default=os.environ.get("BENCHMARKING_DATA_CACHE"),
                        help='Root containing processed_asset/.')

    args = parser.parse_args()
    if args.folder:
        preview_folder = args.folder
    else:
        if not args.source_name:
            raise RuntimeError("Pass --folder or --source-name.")
        if args.data_root is None:
            raise RuntimeError("Set BENCHMARKING_DATA_CACHE or pass --data-root.")
        preview_folder = os.path.join(args.data_root, f"processed_asset/{args.source_name}_preview")
    preview_all_pngs(preview_folder)
    preview_all_pngs(preview_folder, not_title=True)
