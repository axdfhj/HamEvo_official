import os
import os.path as osp
from pathlib import Path
from typing import Literal

import numpy as np
import pyscf
import torch
from ase.io import read
from torch_geometric.data import Data, InMemoryDataset

from src.utils.physical_cal import cut_matrix

from .base import build_orbital_mask, log


class CustomInferData(InMemoryDataset):
    def __init__(
        self,
        root: str | Path,
        name: str = "inference",
        xyz_file: str = "xtb.trj",
        full_orbitals: int = 18,
        transform: callable | None = None,
        pre_filter: callable | None = None,
        data_type: Literal["float32", "float64"] = "float32",
    ) -> None:
        self.folder = osp.join(root, name)
        self.name = name
        self.dtype = getattr(torch, data_type)
        self.full_orbitals = full_orbitals
        self.orbital_mask = build_orbital_mask(full_orbitals)

        super().__init__(self.folder, transform, pre_filter)

        self.trj_file = osp.join(self.folder, xyz_file)
        if not osp.exists(self.trj_file):
            raise FileNotFoundError(f"Trajectory file not found at {self.trj_file}")

        self.configurations = self._read_xtb_trj()
        self.n_conformations = len(self.configurations)
        log.info(f"Total number of configurations: {self.n_conformations}")

    def _read_xtb_trj(self):
        configurations = []
        all_conformers = read(self.trj_file, index=":", format="xyz")

        if os.environ.get("LIMIT_SAMPLES") is not None:
            limit_samples = int(os.environ["LIMIT_SAMPLES"])
            total_configs = len(all_conformers)
            log.info(f"Total configs: {total_configs}. Limiting to {limit_samples}")
            if 0 < limit_samples < total_configs:
                indices = np.unique(np.linspace(0, total_configs - 1, num=limit_samples, dtype=int))
                all_conformers = [all_conformers[i] for i in indices.tolist()]

        for atoms in all_conformers:
            configurations.append(
                {
                    "atoms": atoms.get_atomic_numbers(),
                    "positions": atoms.get_positions(),
                }
            )
        return configurations

    def len(self):
        return self.n_conformations

    @property
    def raw_file_names(self):
        return ["xtb.trj"]

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    def get(self, idx):
        config = self.configurations[idx]
        atoms = torch.tensor(config["atoms"], dtype=torch.long)
        pos = torch.tensor(config["positions"], dtype=self.dtype)

        mol = pyscf.gto.Mole()
        atom_list = [[int(atoms[i]), pos[i].tolist()] for i in range(len(atoms))]
        mol.build(verbose=0, atom=atom_list, basis="def2svp", unit="ang")
        ovlp = torch.tensor(mol.intor("int1e_ovlp"), dtype=self.dtype)
        orbital_mask = torch.cat(
            [self.orbital_mask[atom.item()] + i * self.full_orbitals for i, atom in enumerate(atoms)]
        ).numpy()

        _, _, diag_ham_mask, non_diag_ham_mask, edge_index_full = cut_matrix(
            torch.zeros_like(ovlp), atoms, self.full_orbitals, self.orbital_mask
        )

        return Data(
            mol_id=idx,
            pos=pos,
            atoms=atoms,
            ovlp=ovlp,
            orbital_mask=orbital_mask,
            diag_ham_mask=diag_ham_mask,
            non_diag_ham_mask=non_diag_ham_mask,
            edge_index_full=edge_index_full,
        )
