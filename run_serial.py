import os
import json
import numpy as np
import argparse
from collections import defaultdict
import pandas as pd
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description="All-in-one hierarchical Fe region analysis across multiple snapshots")
parser.add_argument("--cell_size", type=float, default=68.0)
parser.add_argument("--num_bins", type=int, default=8)
parser.add_argument("--sub_bins", type=int, default=2)
parser.add_argument("--partition_threshold", type=int, default=15)
parser.add_argument("--fe_rich_threshold", type=int, default=7)
parser.add_argument("--fe_poor_threshold", type=int, default=4)
parser.add_argument("--fe_file", type=str, required=True)
parser.add_argument("--mg_file", type=str, required=True)
parser.add_argument("--si_file", type=str, required=True)
parser.add_argument("--o_file", type=str, required=True)
parser.add_argument("--n_file", type=str, required=True)
parser.add_argument("--skip", type=int, default=1, help="Process every Nth snapshot")
parser.add_argument("--save_intermediates", action="store_true")
parser.add_argument("--cleanup", action="store_true")
args = parser.parse_args()

atom_types = ['fe', 'mg', 'si', 'o', 'n']
atomic_weights = {'fe': 55.845, 'mg': 24.305, 'si': 28.085, 'o': 15.999, 'n': 14.007}

bin_size = args.cell_size / args.num_bins
shape = (args.num_bins,) * 3
sub_bin_size = bin_size / args.sub_bins
grid_shape = (args.num_bins * args.sub_bins,) * 3

def load_all_snapshots(filepath):
    with open(filepath) as f:
        return json.load(f)

snapshots_per_element = {el: load_all_snapshots(getattr(args, f"{el}_file")) for el in atom_types}
total_snapshots = len(snapshots_per_element['fe'])

def extract_coordinates(snapshot):
    return np.array([atom["atom_coordinate"] for atom in snapshot])

def get_bin_indices(coords):
    return np.array([np.clip((coords[:, i] / bin_size).astype(int), 0, args.num_bins - 1) for i in range(3)]).T

def count_atoms(bin_indices, shape):
    count = np.zeros(shape, dtype=int)
    for x, y, z in bin_indices:
        count[x, y, z] += 1
    return count

def get_sub_bin(coord, xmin, ymin, zmin):
    return tuple(((coord - np.array([xmin, ymin, zmin])) // sub_bin_size).astype(int))

def compute_weight_percent(counts):
    total_weight = sum(counts[el] * atomic_weights[el] for el in atom_types)
    return {el: (counts[el] * atomic_weights[el] / total_weight * 100) if total_weight > 0 else 0 for el in atom_types}

def create_publication_ready_slice_plot(label_grid, fe_counts, z, output_path=None):
    color_map = {0: "#009E73", 1: "#D55E00", 2: "#0072B2"}
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 16)
    ax.set_xticks(np.arange(0, 17, 2))
    ax.set_yticks(np.arange(0, 17, 2))
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_title(f"Fe-rich / Fe-poor / Intermediate Sub-bins (Z = {z})", fontsize=14)
    ax.grid(False)
    for x in range(16):
        for y in range(16):
            label = label_grid[x, y, z]
            color = color_map[label]
            rect = patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="black", linewidth=0.3)
            ax.add_patch(rect)
            fe = fe_counts[x, y, z]
            text_color = "white" if label in [1, 2] else "black"
            ax.text(x + 0.5, y + 0.5, str(fe), ha="center", va="center", fontsize=6.5, color=text_color)
    for i in range(0, 16, 2):
        for j in range(0, 16, 2):
            bin_rect = patches.Rectangle((i, j), 2, 2, linewidth=1.5, edgecolor='black', facecolor='none')
            ax.add_patch(bin_rect)
    legend_elements = [
        Line2D([0], [0], color=color_map[1], lw=4, label='Fe-rich'),
        Line2D([0], [0], color=color_map[2], lw=4, label='Fe-poor'),
        Line2D([0], [0], color=color_map[0], lw=4, label='Intermediate')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=10)
    ax.set_aspect("equal")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

# Track total element mass and total region mass across all snapshots
cumulative_mass = {region: defaultdict(float) for region in ["Fe-rich", "Fe-poor"]}
total_mass_per_region = {region: 0.0 for region in ["Fe-rich", "Fe-poor"]}

for snapshot_idx in range(0, total_snapshots, args.skip):
    print(f"\nProcessing snapshot {snapshot_idx} ...")
    output_dir = f"snapshot_{snapshot_idx}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    raw_snapshots = {el: snapshots_per_element[el][snapshot_idx] for el in atom_types}
    atom_coords = {el: extract_coordinates(raw_snapshots[el]) for el in atom_types}
    bin_indices = {k: get_bin_indices(v) for k, v in atom_coords.items()}
    counts = {k: count_atoms(bin_indices[k], shape) for k in atom_coords}
    rich_bins, poor_bins = [], []
    for i in range(args.num_bins):
        for j in range(args.num_bins):
            for k in range(args.num_bins):
                fe_cnt = counts['fe'][i, j, k]
                (rich_bins if fe_cnt > args.partition_threshold else poor_bins).append((i, j, k))
    sub_bin_lookup = {}
    def process_bins(filtered_bins):
        for i, j, k in filtered_bins:
            xmin, ymin, zmin = i * bin_size, j * bin_size, k * bin_size
            for atom_type, coords in atom_coords.items():
                for coord in coords:
                    if xmin <= coord[0] < xmin + bin_size and ymin <= coord[1] < ymin + bin_size and zmin <= coord[2] < zmin + bin_size:
                        sub_index = get_sub_bin(coord, xmin, ymin, zmin)
                        gx, gy, gz = i * args.sub_bins + sub_index[0], j * args.sub_bins + sub_index[1], k * args.sub_bins + sub_index[2]
                        key = f"{gx}_{gy}_{gz}"
                        if key not in sub_bin_lookup:
                            sub_bin_lookup[key] = {el: 0 for el in atom_types}
                        sub_bin_lookup[key][atom_type] += 1
    process_bins(rich_bins)
    process_bins(poor_bins)
    label_grid = np.zeros(grid_shape, dtype=int)
    fe_counts = np.zeros(grid_shape, dtype=int)
    atom_lookup = {}
    rich_mask = np.zeros(grid_shape, dtype=bool)
    high_fe_mask = np.zeros(grid_shape, dtype=bool)
    for bin_set, label in [(rich_bins, 1), (poor_bins, 2)]:
        for i, j, k in bin_set:
            for sx in range(args.sub_bins):
                for sy in range(args.sub_bins):
                    for sz in range(args.sub_bins):
                        gx, gy, gz = i * args.sub_bins + sx, j * args.sub_bins + sy, k * args.sub_bins + sz
                        key = f"{gx}_{gy}_{gz}"
                        sub = sub_bin_lookup.get(key)
                        if not sub:
                            continue
                        atom_lookup[(gx, gy, gz)] = sub
                        fe_counts[gx, gy, gz] = sub.get("fe", 0)
                        if label == 1:
                            rich_mask[gx, gy, gz] = True
                            if sub["fe"] > args.fe_rich_threshold:
                                high_fe_mask[gx, gy, gz] = True
    interior_mask = binary_erosion(rich_mask) | high_fe_mask
    for (x, y, z), sub in atom_lookup.items():
        if interior_mask[x, y, z]:
            label_grid[x, y, z] = 1
    for (x, y, z), sub in atom_lookup.items():
        if label_grid[x, y, z] == 0 and sub["fe"] <= args.fe_poor_threshold:
            label_grid[x, y, z] = 2
    mid_start = grid_shape[2] // 2 - 1
    for z in range(mid_start, mid_start + 3):
        create_publication_ready_slice_plot(label_grid, fe_counts, z, f"{output_dir}/plots/zslice_{z}.png")
    mapper = {0: "boundary", 1: "Fe-rich", 2: "Fe-poor"}
    region_counts = {region: defaultdict(int) for region in mapper.values()}
    labeled_atom_counts = {}
    for key, atom_counts in sub_bin_lookup.items():
        x, y, z = map(int, key.split("_"))
        label = mapper[int(label_grid[x, y, z])]
        labeled_atom_counts[key] = atom_counts.copy()
        labeled_atom_counts[key]["label"] = label
        for el in atom_types:
            region_counts[label][el] += atom_counts[el]
    wt_percent_rich = compute_weight_percent(region_counts["Fe-rich"])
    wt_percent_poor = compute_weight_percent(region_counts["Fe-poor"])
    df = pd.DataFrame({
        "Fe-rich Count": region_counts["Fe-rich"],
        "Fe-poor Count": region_counts["Fe-poor"],
        "Boundary Count": region_counts["boundary"],
        "Fe-rich wt%": wt_percent_rich,
        "Fe-poor wt%": wt_percent_poor
    })
    df.to_csv(f"{output_dir}/region_summary.csv")
    print(df)

    for region in ["Fe-rich", "Fe-poor"]:
        region_mass = 0.0
        for el in atom_types:
            mass = region_counts[region][el] * atomic_weights[el]
            cumulative_mass[region][el] += mass
            region_mass += mass
        total_mass_per_region[region] += region_mass

# ----------------------------
# Final Average wt% Summary
# ----------------------------

print("\n=== Average wt% across all snapshots ===")
avg_wt_data = {}
for region in ["Fe-rich", "Fe-poor"]:
    print(f"\nRegion: {region}")
    avg_wt_data[region] = {}
    for el in atom_types:
        avg_wt = (cumulative_mass[region][el] / total_mass_per_region[region] * 100) if total_mass_per_region[region] > 0 else 0
        avg_wt_data[region][el] = avg_wt
        print(f"{el.upper()}: {avg_wt:.2f}%")

# Save average wt% to CSV
avg_df = pd.DataFrame(avg_wt_data).T[atom_types]
avg_df.to_csv("average_weight_percent.csv")
