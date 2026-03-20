"""Shared utilities and base class for Hamiltonian dataset implementations."""

import pickle
import sys

import lmdb
import numpy as np
import pyscf
import torch
from torch_geometric.data import InMemoryDataset

from src.utils import RankedLogger

# numpy._core compat hack for older pickled data
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric

log = RankedLogger(__name__, rank_zero_only=True)

BOHR2ANG = 1.8897259886


def build_orbital_mask(full_orbitals: int) -> dict:
    """Build the orbital mask dictionary for a given basis set size."""
    orbital_mask = {}
    if full_orbitals == 14:
        idx_line1 = torch.tensor([0, 1, 3, 4, 5])
        idx_line2 = torch.arange(full_orbitals)
        for i in range(1, 11):
            orbital_mask[i] = idx_line1 if i <= 2 else idx_line2
    elif full_orbitals == 18:
        idx_line1 = torch.tensor([0, 1, 4, 5, 6])
        idx_line2 = torch.tensor([0, 1, 2, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17])
        idx_line3 = torch.arange(full_orbitals)
        for i in range(1, 18):
            if i <= 2:
                orbital_mask[i] = idx_line1
            elif i <= 10:
                orbital_mask[i] = idx_line2
            else:
                orbital_mask[i] = idx_line3
    return orbital_mask


def compute_num_orbitals(atoms, orbital_mask: dict) -> int:
    """Compute total number of orbitals for a molecule."""
    return sum(len(orbital_mask[atom]) for atom in atoms)


def compute_overlap_pyscf(atoms, pos, basis="def2svp", unit="ang"):
    """Compute overlap matrix using PySCF."""
    mol = pyscf.gto.Mole()
    t = [[atoms[i], pos[i]] for i in range(len(atoms))]
    mol.build(verbose=0, atom=t, basis=basis, unit=unit)
    return mol.intor("int1e_ovlp")


def read_lmdb(path: str, idx: int):
    """Read a single entry from an LMDB database."""
    db_env = lmdb.open(path, readonly=True, lock=False)
    with db_env.begin() as txn:
        data = txn.get(int(idx).to_bytes(length=4, byteorder="big"))
        data = pickle.loads(data)
    return data


def count_lmdb_entries(path: str) -> int:
    """Count the number of entries in an LMDB database."""
    db_env = lmdb.open(path, readonly=True, lock=False)
    with db_env.begin() as txn:
        cursor = txn.cursor()
        return sum(1 for _ in cursor.iternext(keys=True, values=False))


class BaseHamiltonianDataset(InMemoryDataset):
    """Base class for LMDB-backed Hamiltonian datasets.

    Subclasses must implement ``_get_raw_data(idx)`` returning:
        (num_orbitals, mol_id, atoms, pos, Ham_list, ovlp)
    """

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return self.n_conformations

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def _get_raw_data(self, idx):
        raise NotImplementedError

    def get(self, idx):
        return self._get_raw_data(idx)
