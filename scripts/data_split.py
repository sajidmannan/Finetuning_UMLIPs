from ase.io import read, write
import math
import os
import random

def my_write_xyz(fileobj, images, comment='', fmt='%22.15f'):
    comment = comment.rstrip()
    if '\n' in comment:
        raise ValueError('Comment line should not have line breaks.')
    for i, atoms in enumerate(images):
        lattice = ''
        for (x, y, z) in images[0].cell:
            lattice += f'{x} {y} {z} '
        temperature = None
        if 'temperature' in atoms.info.keys():
            temperature = atoms.info['temperature']
        chemical_system = None
        if 'chemical_system' in atoms.info.keys():
            chemical_system = atoms.info['chemical_system']
        chemical_system = 'chemical_system=Li'
        lattice = lattice.rstrip()
        pbc_val = ' '.join([str(p) for p in atoms.pbc.tolist()]).replace('True', 'T').replace('False', 'F')
        comment = f'Lattice=\"{lattice}\" Properties=species:S:1:pos:R:{len(atoms.positions[0])}:forces:R:{len(atoms.get_forces()[0])} energy={atoms.get_total_energy()}'
        if temperature:
            comment += f' temperature={temperature} '
        if chemical_system:
            comment += f' chemical_system={chemical_system} '
        comment += f' pbc=\"{pbc_val}\"'
        natoms = len(atoms)
        fileobj.write(f'{natoms}\n{comment}\n')
        for s, (x, y, z), (fx, fy, fz) in zip(atoms.symbols, atoms.positions, atoms.get_forces()):
            fileobj.write('%-2s %s %s %s %s %s %s\n' % (s, fmt % x, fmt % y, fmt % z, fmt % fx, fmt % fy, fmt % fz))

def create_data_split(dataset_path):
    atoms_list = read(dataset_path, format = 'extxyz', index = ':')
    nconfigs = len(atoms_list)
    idxs = list(range(nconfigs))
    train_size = math.floor(nconfigs * 0.6)
    val_size = test_size = math.floor((nconfigs - train_size) * 0.5)
    train_size, test_size, val_size = (1, 1, 1)

    random_seed = 123
    random.seed(random_seed)
    random.shuffle(idxs)

    train_idxs = idxs[:train_size]
    val_idxs = idxs[train_size:train_size + val_size]
    test_idxs = idxs[train_size + val_size:train_size + val_size + test_size]

    train_set = [atoms_list[i] for i in train_idxs]
    val_set = [atoms_list[i] for i in val_idxs]
    test_set = [atoms_list[i] for i in test_idxs]
    print(f'train_set: {len(train_set)}\ntest_set: {len(test_set)}\nval_set: {len(val_set)}')

    for name, dataset in {'train': train_set, 'val': val_set, 'test': test_set}.items():
        with open(f'{name}.xyz', 'a') as f:
            my_write_xyz(f, dataset)
        f.close()

dataset_path = '/scratch/civil/staff/anair.cstaff/test_stuff/test_mace_stuff/sample_data/xyz_data/compiled_Li_data.xyz'
create_data_split(dataset_path = dataset_path)
