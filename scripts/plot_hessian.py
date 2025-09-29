from src.distill_datasets import SimpleDataset
from fairchem.core.common.registry import registry

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import time

def plot_hessian(hessian = None, dataset = None, idx = 0):
    natoms = dataset[idx].natoms
    print(hessian[idx].shape)
    hessian_label = hessian[idx].reshape(natoms * 3, natoms * 3)
    fig, axes = plt.subplots()
    sns.heatmap(hessian_label, cmap = 'YlGnBu')
    fig.savefig(f'../bash_scripts/hessian_map_YlGnBu_{idx}.png', dpi = 300)
    plt.close(fig)

def plot_eigenvalues(hessian = None, dataset = None, idx = 0):
    natoms = dataset[idx].natoms
    print(hessian[idx].shape)
    hessian_label = hessian[idx].reshape(3 * natoms, 3 * natoms)
    fig, axes = plt.subplots()
    eigenvalues, eigenvectors = np.linalg.eig(hessian_label)
    sns.barplot(eigenvalues)
    fig.savefig(f'../bash_scripts/eigenvalue_{idx}.svg', dpi = 300)
    plt.close(fig)

def create_arg_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--hessian_path',
                        help = 'path to the hessian labels',
                        type = str,
                        default = './labels')
    parser.add_argument('--dataset_path',
                        type = str,
                        help = 'path to the dataset',
                        default = './dataset/')
    parser.add_argument('--idxs', 
                        help = 'Index',
                        type = int,
                        nargs = '+',
                        default = 0)
    return parser

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()

    hessian_path = args.hessian_path
    dataset_path = args.dataset_path
    idxs = args.idxs

    hessian = SimpleDataset(hessian_path)
    dataset = registry.get_dataset_class('lmdb')({'src':dataset_path})
    for idx in idxs:
        plot_hessian(hessian, dataset, idx)
        # plot_eigenvalues(hessian, dataset, idx)
