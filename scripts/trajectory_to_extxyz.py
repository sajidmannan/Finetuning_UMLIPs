from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory as AseTrajectory
from ase.io.trajectory import TrajectoryReader as AseTrajReader
from monty.io import zopen
from monty.json import MontyDecoder
from pathlib import Path
from pymatgen.core.trajectory import Trajectory as PmgTrajectory
from tempfile import NamedTemporaryFile

import json

def read_file(traj_path : str | Path) -> list[dict]:
    decoder = MontyDecoder()
    with zopen(str(traj_path),'rt',encoding='utf-8') as f:
        traj = [decoder.decode(line) for line in f]
    return traj

def pmg_to_ase_trajectory(
    file_name: str | Path | PmgTrajectory,
    ase_traj_file: str | Path | None = None,
) -> AseTrajectory:
    if AseTrajReader is None:
        raise ImportError(
            'You must install ASE to use the ASE trajectory' +\
            ' functionality of this class.'
        )

    if ase_traj_file is None:
        temp_file = NamedTemporaryFile()
        ase_traj_file = temp_file.name
        
    if isinstance(file_name, str | Path):
        pmg_traj = PmgTrajectory.from_file(file_name)
    else:
        pmg_traj = file_name
        
    mapping = {
        'energy': 'total',
        'forces': 'forces',
        'stress': 'stress',
    }

    for idx, structure in enumerate(pmg_traj):
        atoms = structure.to_ase_atoms()
        atoms.calc = SinglePointCalculator(
            atoms=atoms,
            **{
                k: pmg_traj.frame_properties[idx][v] \
                for k, v in mapping.items() \
                if v in pmg_traj.frame_properties[idx]
            },
        )
        with AseTrajectory(
                           ase_traj_file,
                           'a' if idx > 0 else 'w', \
                           atoms=atoms
                          ) as _traj_file:
            _traj_file.write()

    ase_traj = AseTrajectory(ase_traj_file, 'r')
    if temp_file is not None:
        temp_file.close()

    return ase_traj

def write_to_xyz(fileobj, images, temperature = None, comment='', fmt='%22.15f'):
    comment = comment.rstrip()
    if '\n' in comment:
        raise ValueError('Comment line should not have line breaks.')
    if temperature:
        extra = f'temperature={temperature}'
    # print('a')
    for i, atoms in enumerate(images):
        lattice = ''
        for (x, y, z) in images[0].cell:
            lattice += f'{x} {y} {z} '
        lattice = lattice.rstrip()
        pbc_val = ' '.join([str(p) for p in atoms.pbc.tolist()])\
                     .replace('True', 'T')\
                     .replace('False', 'F')
        comment = ''.join([f'Lattice=\'{lattice}\' ', \
                           'Properties=species:S:1:', \
                           f'pos:R:{len(atoms.positions[0])}:', \
                           f'forces:R:{len(atoms.get_forces()[0])} ', \
                           f'{extra} energy={atoms.get_total_energy()}', \
                           f' pbc=\'{pbc_val}\''])
        natoms = len(atoms)
        fileobj.write(f'{natoms}\n{comment}\n')
        for s, (x, y, z), (fx, fy, fz) in zip(atoms.symbols, \
                                              atoms.positions, \
                                              atoms.get_forces()):
            fileobj.write('%-2s %s %s %s %s %s %s\n' \
                           % (s, fmt % x, fmt % y, fmt % z, \
                              fmt % fx, fmt % fy, fmt % fz))

def main():
    file_path = ''.join(['/Users/aravind/Downloads/temperature=1000K/', \
                         'chemical_system=Li/dt=2025-01-10-19-16-12-836124.jsonl'])
    dest_path = '/Users/aravind/Downloads/temperature=1000K/'

    trajs = read_file(traj_path = file_path)
    metadata = trajs[0]['metadata']
    with open('Li_metadata.json', 'w') as file:
        json.dump(metadata, file)
    ase_traj = []
    for traj in trajs:
        ase_traj += pmg_to_ase_trajectory(traj['trajectory'])
    # ase_traj = pmg_to_ase_trajectory(trajs[0]['trajectory'])
    print(len(ase_traj))
    with open('Li_data.xyz', 'w') as f:
        write_to_xyz(fileobj = f, \
                     images = ase_traj, \
                     temperature = '1000K',
                       comment = 'chemical_system=Li')

    # dataset_path = '/mnt/e/Work/Datasets/amorphous_diffusivity_dataset/trajectories/'
    # temp_list = ['temperature=1000K', 'temperature=1500K', 'temperature=2000K', 'temperature=2500K', 'temperature=5000K']
    # dest_dir = '/mnt/e/Work/Datasets/amorphous_diffusivity_dataset/trajectories-compiled/'

    # dataset_list = []
    # dataset_list = {os.path.join(
    #         dataset_path, temperature):
    #         {
    #             x:
    #                 os.listdir(os.path.join(dataset_path, temperature, x))
    #             for x in os.listdir(os.path.join(dataset_path, temperature))
    #         }
    #         for temperature in temp_list
    #     }
    # for parent_dir in list(dataset_list.keys())[:1]:
    #     dataset_to_xyz(parent_dir = parent_dir, dataset_list = dataset_list, dest_dir = dest_dir)

if __name__ == '__main__':
    main()

# pmg_trajs = read_trajectories(traj_path = traj_path)
# ase_traj = {
#             entry['metadata'] : pmg_to_ase_trajectory(entry['trajectory']) for entry in pmg_traj[:1]
#            }
