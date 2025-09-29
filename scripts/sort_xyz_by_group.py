from datetime import timedelta
from pathlib import Path
from tqdm import tqdm

import os
import argparse
import ase.io
import time

def group_xyz_by_dataset_name(xyz_path, output_dir, dataset_type, group_name, nconfigs = ''):
    # Read all atoms from the original XYZ file
    atoms_list = ase.io.read(xyz_path, ':')  # Load all configurations
    grouped_atoms = {}

    # Group atoms by 'dataset_name' (assuming it's in the 'info' dictionary of each atom)
    for atoms in tqdm(atoms_list, desc="Grouping by dataset_name"):
        dataset_name = atoms.info.get(group_name, 'unknown')
        if dataset_name not in grouped_atoms:
            grouped_atoms[dataset_name] = []
        grouped_atoms[dataset_name].append(atoms)

    # Write separate XYZ files for each group
    for dataset_name, group_atoms in grouped_atoms.items():
        dataset_dir = os.path.join(output_dir, dataset_name, dataset_type)
        os.makedirs(dataset_dir, exist_ok=True)
        output_file = os.path.join(dataset_dir, f'data_{nconfigs}.xyz')
        
        # Write grouped atoms to a new XYZ file
        ase.io.write(output_file, group_atoms)
        print(f"Saved {len(group_atoms)} configurations to {output_file}")

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        help = 'Path to input dataset',
                        type = str,
                        default = './labels')
    parser.add_argument('--output_path',
                        help = 'Path to output dataset',
                        type = str,
                        default = './Datasets')
    parser.add_argument('--dataset_type',
                        help = 'Dataset type (from train, test and val)',
                        type = str,
                        default = 'train')
    parser.add_argument('--group_name',
                        help = 'Name of the Grouping property',
                        type = str,
                        default = 'Unknown')
    parser.add_argument('--nconfigs',
                        help = 'No. of Configs to be used',
                        default = '')
    return parser

if __name__ == "__main__":
    start = time.monotonic()
    parser = create_arg_parser()
    args = parser.parse_args()

    xyz_path = args.input_path
    output_dir = args.output_path
    dataset_type = args.dataset_type
    group_name = args.group_name
    nconfigs = ''

    # run the grouping function
    group_xyz_by_dataset_name(xyz_path = xyz_path,
                              output_dir = output_dir,
                              dataset_type=dataset_type,
                              group_name = group_name,)
    end = time.monotonic()
    print(f'\n\nTotal execution time: {timedelta(seconds = end - start)}')
