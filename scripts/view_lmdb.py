# from ase.io import read
# from ase import Atoms
from src.distill_datasets import SimpleDataset
from fairchem.core.common.registry import registry

import time
# import os
# import lmdb
# import numpy as np
# import torch

'''
start = time.monotonic()
dataset_path = '/home/civil/staff/anair.cstaff/mlff_distill_stuff/datasets/labels/SPICE/spice_separated/Solvated_Amino_Acids/train'
# dataset_path = '/scratch/civil/staff/anair.cstaff/mlff-distill_stuff/Data/reduced_MPMorph/MPMorph_seperated/Li/train/lmdb'
dataset = registry.get_dataset_class('lmdb')({'src':dataset_path})
data = dataset[0]
end = time.monotonic()
print(f'Dataset Keys (len(dataset)): {data} ({end - start}s)')
'''

start = time.monotonic()
dataset_path = '/scratch/civil/staff/anair.cstaff/mlff-distill_stuff/Data/reduced_MPMorph/MPMorph_seperated/Li/train/lmdb'
# dataset_path = '/home/civil/staff/anair.cstaff/mlff_distill_stuff/datasets/labels/SPICE/spice_separated/Solvated_Amino_Acids/train'
dataset = registry.get_dataset_class('lmdb')({'src':dataset_path})
forces = dataset[0].get('force')
end = time.monotonic()
print(f'Dataset Forces ({len(dataset)}): {forces.shape} ({end - start}s)')


start = time.monotonic()
saved_forces_path = '/scratch/civil/staff/anair.cstaff/mlff-distill_stuff/labels/mace_mpa_0/Li_100/train_forces'
# saved_forces_path = '/home/civil/staff/anair.cstaff/mlff_distill_stuff/datasets/labels/labels/SPICE_labels/mace_off_large_SpiceAminos/train_forces'
saved_forces = SimpleDataset(saved_forces_path)
label_forces = saved_forces[0]
end = time.monotonic()
print(f'Label Forces ({len(saved_forces)}): {label_forces.shape} ({end - start}s)')

start = time.monotonic()
hessian_labels_path = '/scratch/civil/staff/anair.cstaff/mlff-distill_stuff/labels/mace_mpa_0/Li_100/force_jacobians'
# hessian_labels_path = '/home/civil/staff/anair.cstaff/mlff_distill_stuff/datasets/labels/labels/SPICE_labels/mace_off_large_SpiceAminos/force_jacobians'
hessian_labels = SimpleDataset(hessian_labels_path)
hessian_label = hessian_labels[0]
end = time.monotonic()
print(f'Hessian Labels ({len(hessian_labels)}): {hessian_label.shape} ({end - start}s)')
