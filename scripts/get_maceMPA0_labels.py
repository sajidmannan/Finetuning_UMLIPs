from ase import Atoms
from ase.io import read
from fairchem.core.common.registry import registry
from mace.calculators import MACECalculator
from tqdm import tqdm
from torch.utils.data import Subset

import argparse
import lmdb
import numpy as np
import os
import torch

def read_data(file_name):
    env = lmdb.open(file_name)
    txn = env.begin()
    cursor = txn.cursor()

    with env.begin() as txn:
        for key, value in cursor:
            print(f'{key}: {value.shape}')

def record_and_save(dataset, file_path, fn):
    # Assuming train_loader is your DataLoader
    print('-'*80)
    # print(f'dataset while recording: {len(dataset)}')
    avg_num_atoms = dataset[0].natoms.item()
    map_size = 1099511627776 * 2

    env = lmdb.open(file_path, map_size = map_size)
    env_info = env.info()
    for sample in tqdm(dataset):
        with env.begin(write = True) as txn:
            sample_id = str(int(sample.id))
            sample_output = fn(sample)  # this function needs to output an array where each element correponds to the label for an entire molecule
            # Convert tensor to bytes and write to LMDB
            sample_output = sample_output.astype(np.float32)
            txn.put(sample_id.encode(), sample_output.tobytes())
    print(f"\nAll tensors saved to LMDB:{file_path}")

def record_labels(labels_folder, dataset_path, model_path, device):
    os.makedirs(labels_folder, exist_ok=True)
    # Load the dataset
    train_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(dataset_path, 'train', 'lmdb_01')})
    val_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(dataset_path, 'val', 'lmdb_01')})

    print('-' * 80)
    print(f'train_dataset: {len(train_dataset)}\nval_dataset: {len(val_dataset)}')
    print('-' * 80)

    # Load the model
    print(f'loading model from {model_path}')
    calc = MACECalculator(model_path= model_path,
                          dispersion=False,
                          default_dtype='float64',
                          device = device)
 
    def get_forces(sample):
        atomic_numbers = sample.atomic_numbers.numpy()
        atoms = Atoms(numbers=atomic_numbers, positions=sample.pos.numpy(),
                       cell=sample.cell.numpy()[0], pbc=True)
        atoms.calc = calc
        forces = atoms.get_forces()
        print(f'forces: {forces.shape}')
        return forces

    def get_hessians(sample, device = 'cuda'):
        atomic_numbers = sample.atomic_numbers.numpy()
        natoms = sample.natoms
        atoms = Atoms(numbers=atomic_numbers, positions=sample.pos.numpy(),
                      cell=sample.cell.numpy()[0], pbc=True)
        atoms.calc = calc

        hessian = calc.get_hessian(atoms = atoms)
        print(f'hessian: {hessian.shape}')
        
        # this is SUPER IMPORTANT!!! multiply by -1
        reshaped_hessian = - 1 * hessian.reshape(natoms, 3, natoms, 3)
        print(f'reshaped hessian: {reshaped_hessian.shape}')
        
        return reshaped_hessian 

    os.makedirs(os.path.join(labels_folder, 'force_jacobians'),
                exist_ok = True)
    record_and_save(train_dataset,
                    os.path.join(labels_folder,
                                 'force_jacobians',
                                 'force_jacobians.lmdb'),
                    get_hessians)

    os.makedirs(os.path.join(labels_folder, 'train_forces'), 
                exist_ok = True)
    record_and_save(train_dataset,
                    os.path.join(labels_folder,
                                 'train_forces',
                                 'train_forces.lmdb'),
                    get_forces)

    os.makedirs(os.path.join(labels_folder, 'val_forces'), 
                exist_ok = True)
    record_and_save(val_dataset,
                    os.path.join(labels_folder,
                                 'val_forces',
                                 'val_forces.lmdb'),
                    get_forces)

def create_arg_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--labels_folder',
                        help = 'path for saving the hessian labels',
                        type = str,
                        default = './labels')
    parser.add_argument('--dataset_path',
                        type = str,
                        help = 'path to the dataset',
                        default = './dataset/')
    parser.add_argument('--device', 
                        help = 'Device to run the model',
                        type = str,
                        default = 'cuda')
    parser.add_argument('--model_path',
                        help = 'Path to teacher model checkpoint',
                        default = './teacher_model')
    return parser 

if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    
    labels_folder = args.labels_folder
    dataset_path = args.dataset_path
    device = args.device
    model_path = args.model_path

    record_labels(labels_folder, dataset_path, model_path, device)
