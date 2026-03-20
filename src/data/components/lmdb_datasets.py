"""LMDB-backed Hamiltonian dataset implementations.

All five LMDB datasets share the same backend, base class, and return contract:
    (num_orbitals, mol_id, atoms, pos, Ham_list, ovlp)
"""

import os
import os.path as osp
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .base import (
    BOHR2ANG,
    BaseHamiltonianDataset,
    build_orbital_mask,
    compute_num_orbitals,
    compute_overlap_pyscf,
    count_lmdb_entries,
    log,
    read_lmdb,
)

# ---------------------------------------------------------------------------
# MD17
# ---------------------------------------------------------------------------


class MD17Loop(BaseHamiltonianDataset):
    def __init__(
        self,
        root: str | Path = "scf_dataset/md17-v2",
        name="ethanol",
        full_orbitals: int = 14,
        split_path: str | None = None,
        transform: callable | None = None,
        pre_filter: callable | None = None,
        get_ovlp: bool = False,
        data_type: Literal["float32", "float64"] = "float32",
    ) -> None:
        self.folder = osp.join(root, name)
        self.name = name
        self.dtype = getattr(torch, data_type)
        self.full_orbitals = full_orbitals
        self.orbital_mask = build_orbital_mask(full_orbitals)
        self.get_ovlp = get_ovlp

        super().__init__(self.folder, transform, pre_filter)

        lmdb_path = osp.join(self.folder, "processed", f"{name}.lmdb")
        self.n_conformations = count_lmdb_entries(lmdb_path)
        log.info(f"total conformation number is {self.n_conformations}.")

        if split_path is not None:
            self.molecular_splits = torch.load(split_path)
        else:
            self.molecular_splits = torch.load(os.path.join(root, name, "splits.pt"))

    @property
    def raw_file_names(self):
        return sorted(os.listdir(self.raw_dir))

    @property
    def processed_file_names(self):
        return [f"{self.name}.lmdb/data.mdb"]

    def _get_raw_data(self, idx):
        path = os.path.join(self.processed_dir, f"{self.name}.lmdb")
        data_dict = read_lmdb(path, idx)

        mol_id, atoms, pos, Ham_list = (
            data_dict["mol_id"],
            data_dict["atoms"],
            data_dict["pos"],
            data_dict["Ham_list"],
        )
        num_orbitals = compute_num_orbitals(atoms, self.orbital_mask)
        ovlp = compute_overlap_pyscf(atoms, pos) if self.get_ovlp else None
        pos = pos.reshape(-1, 3)
        return num_orbitals, mol_id, atoms, pos, Ham_list, ovlp


# ---------------------------------------------------------------------------
# QH9 Stable / Dynamic (LMDB)
# ---------------------------------------------------------------------------


class QH9Loop(BaseHamiltonianDataset):
    def __init__(
        self,
        root: str | Path = "scf_dataset/qh9stable",
        full_orbitals: int = 14,
        transform: callable | None = None,
        pre_transform: callable | None = None,
        pre_filter: callable | None = None,
        get_ovlp: bool = False,
        version: Literal["id", "ood", "100k", "300k"] = "id",
        data_type: Literal["float32", "float64"] = "float32",
    ) -> None:
        self.folder = root
        self.version = version
        self.dtype = getattr(torch, data_type)
        self.full_orbitals = full_orbitals
        self.orbital_mask = build_orbital_mask(full_orbitals)
        self.get_ovlp = get_ovlp

        self.mode = "dynamic" if version in ["100k", "300k"] else "stable"
        self.lmdb_path = os.path.join(self.folder, "processed", f"qh9{self.mode}.lmdb")
        super().__init__(root, transform, pre_transform, pre_filter)

        self.n_conformations = count_lmdb_entries(self.lmdb_path)
        log.info(f"total conformation number is {self.n_conformations}.")
        self.molecular_splits = torch.load(os.path.join(root, f"{version}_splits.pt"))

    @property
    def raw_file_names(self):
        return sorted(os.listdir(self.raw_dir))

    @property
    def processed_file_names(self):
        return [f"qh9{self.mode}.lmdb/data.mdb"]

    def _get_raw_data(self, idx):
        data_dict = read_lmdb(self.lmdb_path, idx)

        mol_id, atoms, pos, Ham_list = (
            data_dict["mol_id"],
            data_dict["atoms"],
            data_dict["pos"],
            data_dict["Ham_list"],
        )
        num_orbitals = compute_num_orbitals(atoms, self.orbital_mask)
        Ham_list = [Ham.reshape(num_orbitals, num_orbitals) for Ham in Ham_list]
        ovlp = compute_overlap_pyscf(atoms, pos) if self.get_ovlp else None
        pos = pos.reshape(-1, 3)
        return num_orbitals, mol_id, atoms, pos, Ham_list, ovlp


class QH9Dynamic(BaseHamiltonianDataset):
    def __init__(
        self,
        root="scf_dataset/",
        task="",
        split="geometry",
        version="300k",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        data_type="float32",
        test=True,
    ):
        assert task in [""], f"Unsupported task: {task}"
        self.dtype = getattr(torch, data_type)
        self.version = version
        self.test = test
        if self.version == "300k":
            self.folder = os.path.join(root, "QH9Dynamic_300k")
        elif self.version == "100k":
            self.folder = os.path.join(root, "QH9Dynamic_100k")
        else:
            log.info(f"Version {version} not in [100k, 300k], using 300k.")
            self.folder = os.path.join(root, "QH9Dynamic_300k")
            self.version = "300k"

        self.split = split
        self.full_orbitals = 14
        self.orbital_mask = build_orbital_mask(14)

        super().__init__(self.folder, transform, pre_transform, pre_filter)
        self.train_mask, self.val_mask, self.test_mask = torch.load(self.processed_paths[0])
        self.slices = {
            "id": torch.arange(self.train_mask.shape[0] + self.val_mask.shape[0] + self.test_mask.shape[0] + 1)
        }
        self.molecular_splits = (self.train_mask, self.val_mask, self.test_mask)

    @property
    def raw_file_names(self):
        return [f"QH9Dynamic_{self.version}.db"]

    @property
    def processed_file_names(self):
        if self.split == "geometry":
            return ["processed_QH9Dynamic_geometry.pt", "QH9Dynamic.lmdb/data.mdb"]
        elif self.split == "mol":
            return ["processed_QH9Dynamic_mol.pt", "QH9Dynamic.lmdb/data.mdb"]

    def get(self, idx):
        lmdb_path = os.path.join(self.processed_dir, "QH9Dynamic.lmdb")
        data_dict = read_lmdb(lmdb_path, idx)

        mol_id, num_nodes = data_dict["id"], data_dict["num_nodes"]
        atoms = np.frombuffer(data_dict["atoms"], np.int32)
        pos = np.frombuffer(data_dict["pos"], np.float64)
        Ham = np.frombuffer(data_dict["Ham"], np.float64)
        pos = pos.reshape(num_nodes, 3) / BOHR2ANG
        num_orbitals = sum(5 if atom <= 2 else 14 for atom in atoms)
        Ham = Ham.reshape(num_orbitals, num_orbitals)
        pos = pos.reshape(-1, 3)
        return num_orbitals, mol_id, atoms, pos, [Ham], None


# ---------------------------------------------------------------------------
# GDB-17
# ---------------------------------------------------------------------------


class GDB17LoopV2(BaseHamiltonianDataset):
    def __init__(
        self,
        root: str | Path = "scf_dataset/gdb-17v2",
        transform: callable | None = None,
        pre_filter: callable | None = None,
        data_type: Literal["float32", "float64"] = "float32",
        version: Literal["ood", "ood_50"] = "ood",
        get_ovlp: bool = True,
        processed_link: str | Path | None = None,
    ) -> None:
        self.folder = root
        self.version = version
        assert version in ["ood", "ood_50"], f"Unsupported version: {version}"
        self.dtype = getattr(torch, data_type)
        self.full_orbitals = 18
        self.processed_link = processed_link
        self.orbital_mask = build_orbital_mask(18)
        self.get_ovlp = get_ovlp

        super().__init__(self.folder, transform, pre_filter)

        lmdb_path = osp.join(self.folder, "processed", "gdb-17.lmdb")
        self.n_conformations = count_lmdb_entries(lmdb_path)
        log.info(f"total conformation number is {self.n_conformations}.")
        self.cycle_training = False

        self.weights = torch.load(os.path.join(root, "split/n_cycle.pt"))
        self.cycle_splits = torch.load(os.path.join(root, "split/gdb-17_random_12.pt"))
        self.molecular_splits = torch.load(os.path.join(root, "split/gdb-17_random_12.pt"))

    @property
    def raw_file_names(self) -> list[str]:
        return sorted(os.listdir(self.raw_dir))

    @property
    def processed_file_names(self) -> list[str]:
        return ["gdb-17.lmdb/data.mdb"]

    def _get_raw_data(self, idx):
        local_dir = os.path.join(self.root, "processed") if self.processed_link is None else self.processed_link
        data_dict = read_lmdb(os.path.join(local_dir, "gdb-17.lmdb"), idx)

        mol_id, atoms, pos, Ham_list = (
            data_dict["mol_id"],
            data_dict["atoms"],
            data_dict["pos"],
            data_dict["Ham_list"],
        )
        num_orbitals = compute_num_orbitals(atoms, self.orbital_mask)
        pos = pos.reshape(-1, 3)
        ovlp = compute_overlap_pyscf(atoms, pos) if self.get_ovlp else None
        return num_orbitals, mol_id, atoms, pos, Ham_list, ovlp


# ---------------------------------------------------------------------------
# Custom (flexible LMDB / DB)
# ---------------------------------------------------------------------------


class CustomData(BaseHamiltonianDataset):
    def __init__(
        self,
        root: str | Path = "scf_dataset/md-dataset",
        name="Naphthalene",
        full_orbitals: int = 18,
        transform: callable | None = None,
        pre_filter: callable | None = None,
        train_samples: int = None,
        split_path: str = None,
        get_ovlp: bool = False,
        unit_transfer: bool = False,
        data_type: Literal["float32", "float64"] = "float32",
    ) -> None:
        self.folder = osp.join(root, name)
        self.name = name
        self.unit_transfer = unit_transfer
        self.dtype = getattr(torch, data_type)
        self.full_orbitals = full_orbitals
        self.orbital_mask = build_orbital_mask(full_orbitals)
        self.get_ovlp = get_ovlp

        super().__init__(self.folder, transform, pre_filter)

        if "QMUGS_NATOMS" in os.environ:
            self.qmugs_natoms = int(os.environ["QMUGS_NATOMS"])
            self.lmdb_path = osp.join(self.folder, "processed", f"conf_{self.qmugs_natoms}atoms.db")
        else:
            db_path = osp.join(self.folder, "processed", f"{name}.db")
            lmdb_path_alt = osp.join(self.folder, "processed", f"{name}.lmdb")
            self.lmdb_path = db_path if osp.exists(db_path) else lmdb_path_alt

        self.n_conformations = count_lmdb_entries(self.lmdb_path)
        log.info(f"total conformation number is {self.n_conformations}.")

        # Load splits
        try:
            if split_path is None:
                split_path = os.path.join(root, name, "splits.pt")
            else:
                split_path = osp.join(root, name, split_path)
            self.molecular_splits = torch.load(split_path)
            log.info(os.path.basename(split_path))
        except FileNotFoundError:
            indices = np.arange(self.n_conformations)
            n = train_samples if train_samples is not None else self.n_conformations
            train_indices = np.random.choice(indices, n, replace=False)
            self.molecular_splits = (train_indices, indices, indices)
            log.info("no splits file found, using random splits.")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"{self.name}.db/data.mdb"]

    def get(self, idx):
        data_dict = read_lmdb(self.lmdb_path, idx)

        try:
            mol_id, atoms, pos, Ham_list = (
                data_dict["mol_id"],
                data_dict["atoms"],
                data_dict["pos"],
                data_dict["Ham_list"],
            )
        except (KeyError, TypeError):
            mol_id = data_dict["mol_id"]
            atoms = np.frombuffer(data_dict["atoms"], np.int32)
            pos = np.frombuffer(data_dict["pos"], np.float64)
            Ham_list = [np.frombuffer(data_dict["Ham"], np.float64)]

        num_orbitals = compute_num_orbitals(atoms, self.orbital_mask)
        Ham_list = [ham.reshape(num_orbitals, num_orbitals) for ham in Ham_list]
        ovlp = compute_overlap_pyscf(atoms, pos) if self.get_ovlp else None

        if self.unit_transfer:
            pos = pos / BOHR2ANG
        pos = pos.reshape(-1, 3)
        return num_orbitals, mol_id, atoms, pos, Ham_list, ovlp
