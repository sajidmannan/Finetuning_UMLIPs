"""
Convert xyz to lmdb format.
"""

from ase.io import read
from datetime import timedelta
from pathlib import Path
from torch_geometric.data import Data
from tqdm import tqdm

import argparse
import json
import lmdb
import numpy as np
import os
import pickle
import time
import torch

def write_lmdb(files, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    print(f"Writing to {str(dir / 'data.lmdb')}")
    db = lmdb.open(
        str(dir / "data.lmdb"),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    collect_meta = {
        "force": [],
        "total_energy": [],
    }
    i = 0
    for file_path in tqdm(files):
        diff_idx= 0
        for atoms in read(file_path, index=":"):
            new_sid = str(diff_idx)
            data = xyz_to_pyg_data(atoms, sid = new_sid )
            txn = db.begin(write=True)
            txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()
            diff_idx += 1
            i += 1
            for key in collect_meta.keys():
                if key == "force":
                    collect_meta[key].append(atoms.get_forces())
                elif key == "total_energy":
                    collect_meta[key].append(atoms.get_potential_energy())
                else:
                    collect_meta[key].append(atoms.info[key])

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(i, protocol=-1))
    txn.commit()

    db.sync()
    db.close()
    return collect_meta


def process_meta(results, dirs):
    all_results = {}
    
    for key in results.keys():
        all_results[key] = []
        for i in range(len(results[key])):
            if results[key][i] is not None:
                all_results[key].append(torch.tensor(results[key][i]))

        if all_results[key][0].numel() > 1:
            all_results[key] = torch.cat(all_results[key])
        else:
            all_results[key] = torch.tensor(all_results[key])

    meta_data = {}
    for key in all_results.keys():
        meta_data[f"{key}_mean"] = float(all_results[key].mean())
        meta_data[f"{key}_std"] = float(all_results[key].std())
        meta_data[f"{key}_min"] = float(all_results[key].min())
        meta_data[f"{key}_max"] = float(all_results[key].max())

    with open(dirs / "meta_data.json", "w") as f:
        json.dump(meta_data, f)

def xyz_to_pyg_data(atoms, sid):

    # Input
    atomic_numbers = torch.tensor(
       atoms.get_atomic_numbers()
    )  # natoms
    cell = torch.tensor(atoms.cell, dtype=torch.float).unsqueeze(
        dim=0
    )  # 1 3 3

    pos = torch.tensor(
                       atoms.get_positions()
                      ).float()  # natoms 3

    natoms = torch.tensor(len(atomic_numbers)).unsqueeze(0)
    fixed = torch.zeros(len(atomic_numbers))

    # Output
    total_energy = torch.tensor(atoms.get_potential_energy(), dtype=torch.float)
    force = torch.tensor(atoms.get_forces(), dtype=torch.float)  # natoms 3
    temperature = atoms.info['temperature']
    group_name = atoms.info['chemical_system']
    data = Data(
                atomic_numbers=atomic_numbers,
                pos=pos,
                cell=cell,
                natoms=natoms,
                fixed=fixed,
                total_energy=total_energy,
                force=force,
                temperature=temperature,
                group_name=group_name,
                sid=sid,
                )
    return data

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xyz_path',
                        help = 'Path to the xyz dataset',
                        type = str,
                        default = './data/train')
    parser.add_argument('--output_dir',
                        type = str,
                        help = 'Path to the LMDB dataset',
                        default = './data/train')
    return parser

def main(preprocessed_datapath, lmdb_output_dir):
    files = [os.path.join(preprocessed_datapath, f)
             for f in os.listdir(preprocessed_datapath)
             # if f.endswith('.xyz')]
             if f.endswith( '_01.xyz')]
    lmdb_output_dir = Path(lmdb_output_dir)

    meta_info = write_lmdb(files,  lmdb_output_dir)
    process_meta(meta_info, lmdb_output_dir)

    print("Done!")

if __name__ == "__main__":
    start = time.monotonic()
    parser = create_arg_parser()
    args = parser.parse_args()

    preprocessed_datapath = args.xyz_path
    lmdb_output_dir = args.output_dir
    
    main(preprocessed_datapath, lmdb_output_dir)
    end = time.monotonic()
    print(f'Total Execution Time: {timedelta(seconds = end - start)}')
