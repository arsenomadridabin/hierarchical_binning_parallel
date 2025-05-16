import os
import json
import numpy as np
import argparse
from collections import defaultdict
from multiprocessing import Pool
import pandas as pd
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description="Parallel Fe region segmentation for multiple snapshots")
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
parser.add_argument("--skip", type=int, default=10)
parser.add_argument("--nproc", type=int, default=4)
parser.add_argument("--out_dir", type=str, default="outputs")
parser.add_argument("--plot_k", type=int, default=1, help="Plot middle z +/- k slices")
args = parser.parse_args()

atom_types = ['fe', 'mg', 'si', 'o', 'n']
atomic_weights = {'fe': 55.845, 'mg': 24.305, 'si': 28.085, 'o': 15.999, 'n': 14.007}
os.makedirs(args.out_dir, exist_ok=True)

def load_all_snapshots(filepath):
    with open(filepath) as f:
        return json.load(f)

full_data = {el: load_all_snapshots(getattr(args, f"{el}_file")) for el in atom_types}

def plot_z_slices(label_grid, fe_counts, snapshot_idx):
    plot_dir = os.path.join(args.out_dir, f"plots_{snapshot_idx:05d}")
    os.makedirs(plot_dir, exist_ok=True)
    color_map = {0: "#009E73", 1: "#D55E00", 2: "#0072B2"}
    mid_z = label_grid.shape[2] // 2

    for z in range(mid_z - args.plot_k, mid_z + args.plot_k + 1):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.set_facecolor('white')

        for x in range(label_grid.shape[0]):
            for y in range(label_grid.shape[1]):
                label = label_grid[x, y, z]
                color = color_map[label]
                rect = patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='black', linewidth=0.3)
                ax.add_patch(rect)
                fe = fe_counts[x, y, z]
                if label in [1, 2]:
                    ax.text(x + 0.5, y + 0.5, str(fe), ha='center', va='center', fontsize=6.5, color='white')

        for i in range(0, label_grid.shape[0], 2):
            for j in range(0, label_grid.shape[1], 2):
                rect = patches.Rectangle((i, j), 2, 2, linewidth=1.0, edgecolor='black', facecolor='none')
                ax.add_patch(rect)

        ax.set_xlim(0, label_grid.shape[0])
        ax.set_ylim(0, label_grid.shape[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        plt.gca().invert_yaxis()

        legend_elements = [
            Line2D([0], [0], color=color_map[1], lw=4, label='Fe-rich'),
            Line2D([0], [0], color=color_map[2], lw=4, label='Fe-poor'),
            Line2D([0], [0], color=color_map[0], lw=4, label='Intermediate')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8, frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"zslice_{z}.png"))
        plt.close()

def process_snapshot(index):
    print(f"Processing snapshot {index}...", flush=True)
    snapshot = {el: full_data[el][index] for el in atom_types}
    coords = {el: np.array([atom['atom_coordinate'] for atom in snapshot[el]]) for el in atom_types}

    bin_size = args.cell_size / args.num_bins
    grid_shape = (args.num_bins * args.sub_bins,) * 3

    def count_atoms(indices):
        count = np.zeros((args.num_bins,) * 3, dtype=int)
        for i in indices:
            count[tuple(i)] += 1
        return count

    bin_indices = {k: np.clip((coords[k] / bin_size).astype(int), 0, args.num_bins - 1) for k in coords}
    counts = {k: count_atoms(bin_indices[k]) for k in coords}

    rich_bins, poor_bins = [], []
    for i in range(args.num_bins):
        for j in range(args.num_bins):
            for k in range(args.num_bins):
                if counts['fe'][i, j, k] > args.partition_threshold:
                    rich_bins.append((i, j, k))
                else:
                    poor_bins.append((i, j, k))

    sub_bin_size = bin_size / args.sub_bins
    sub_bin_lookup = {}

    def process_bins(filtered_bins):
        for i, j, k in filtered_bins:
            origin = np.array([i, j, k]) * bin_size
            for atom_type, positions in coords.items():
                for coord in positions:
                    if np.all((coord >= origin) & (coord < origin + bin_size)):
                        sub = ((coord - origin) // sub_bin_size).astype(int)
                        gx, gy, gz = i * args.sub_bins + sub[0], j * args.sub_bins + sub[1], k * args.sub_bins + sub[2]
                        key = f"{gx}_{gy}_{gz}"
                        if key not in sub_bin_lookup:
                            sub_bin_lookup[key] = {el: 0 for el in atom_types}
                        sub_bin_lookup[key][atom_type] += 1

    process_bins(rich_bins)
    process_bins(poor_bins)

    label_grid = np.zeros(grid_shape, dtype=int)
    fe_counts = np.zeros(grid_shape, dtype=int)
    rich_mask = np.zeros(grid_shape, dtype=bool)
    high_fe_mask = np.zeros(grid_shape, dtype=bool)
    atom_lookup = {}

    for key, val in sub_bin_lookup.items():
        x, y, z = map(int, key.split("_"))
        atom_lookup[(x, y, z)] = val
        fe_counts[x, y, z] = val["fe"]
        if val["fe"] > 0:
            rich_mask[x, y, z] = True
        if val["fe"] > args.fe_rich_threshold:
            high_fe_mask[x, y, z] = True

    interior_mask = binary_erosion(rich_mask) | high_fe_mask
    for (x, y, z), val in atom_lookup.items():
        if interior_mask[x, y, z]:
            label_grid[x, y, z] = 1

    for (x, y, z), val in atom_lookup.items():
        if label_grid[x, y, z] == 0 and val["fe"] <= args.fe_poor_threshold:
            label_grid[x, y, z] = 2

    plot_z_slices(label_grid, fe_counts, index)

    region_counts = {region: defaultdict(int) for region in ['Fe-rich', 'Fe-poor', 'boundary']}
    mapper = {0: 'boundary', 1: 'Fe-rich', 2: 'Fe-poor'}

    for key, val in sub_bin_lookup.items():
        x, y, z = map(int, key.split("_"))
        label = label_grid[x, y, z]
        for el in atom_types:
            region_counts[mapper[label]][el] += val[el]

    def compute_weight_percent(counts):
        total = sum(counts[el] * atomic_weights[el] for el in atom_types)
        return {el: (counts[el] * atomic_weights[el] / total * 100) if total > 0 else 0 for el in atom_types}

    df = pd.DataFrame({
        "Fe-rich Count": region_counts["Fe-rich"],
        "Fe-poor Count": region_counts["Fe-poor"],
        "Boundary Count": region_counts["boundary"],
        "Fe-rich wt%": compute_weight_percent(region_counts["Fe-rich"]),
        "Fe-poor wt%": compute_weight_percent(region_counts["Fe-poor"]),
    })

    df.to_csv(os.path.join(args.out_dir, f"summary_{index:05d}.csv"))
    with open(os.path.join(args.out_dir, f"sub_bin_atom_counts_{index:05d}.json"), "w") as f:
        json.dump(sub_bin_lookup, f, indent=2)

    return {
        "index": index,
        "region_mass": {
            region: sum(region_counts[region][el] * atomic_weights[el] for el in atom_types)
            for region in ["Fe-rich", "Fe-poor"]
        },
        "region_element_mass": {
            region: {
                el: region_counts[region][el] * atomic_weights[el] for el in atom_types
            } for region in ["Fe-rich", "Fe-poor"]
        }
    }

if __name__ == '__main__':
    total_snapshots = len(full_data['fe'])
    indices = list(range(0, total_snapshots, args.skip))

    with Pool(processes=args.nproc) as pool:
        results = pool.map(process_snapshot, indices)

    final = {region: defaultdict(float) for region in ["Fe-rich", "Fe-poor"]}
    total_mass = {region: 0 for region in final}

    for res in results:
        for region in final:
            m = res["region_mass"][region]
            total_mass[region] += m
            for el in atom_types:
                final[region][el] += res["region_element_mass"][region][el]

    avg_wt_percent = {
        region: {
            el: (final[region][el] / total_mass[region]) * 100 if total_mass[region] > 0 else 0
            for el in atom_types
        } for region in final
    }

    avg_df = pd.DataFrame(avg_wt_percent)
    avg_df.to_csv(os.path.join(args.out_dir, "average_weight_percent.csv"))
    print("Average wt% across snapshots saved to average_weight_percent.csv")
