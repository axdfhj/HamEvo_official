import os
import os.path as osp
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset

from .base import (
    BOHR2ANG,
    build_orbital_mask,
    compute_num_orbitals,
    log,
)
from .sqlite_backend import HamiltonianDatabase


class NablaDFTData(InMemoryDataset):
    def __init__(
        self,
        root: str | Path = "scf_dataset/nablaDFT_dataset",
        db_path: str = None,
        root_train: str = None,
        root_val: str = None,
        root_test: str = None,
        split_train: str = None,
        split_val: str = None,
        split_test: str = None,
        full_orbitals: int = 18,
        transform: callable | None = None,
        pre_filter: callable | None = None,
        train_samples: int = None,
        split_path: str = None,
        get_ovlp: bool = False,
        unit_transfer: bool = False,
        data_type: Literal["float32", "float64"] = "float32",
    ) -> None:
        self.db_path = db_path
        self.name = "nablaDFT"
        self.folder = root
        self.unit_transfer = unit_transfer
        self.dtype = getattr(torch, data_type)
        self.full_orbitals = full_orbitals
        self.orbital_mask = build_orbital_mask(full_orbitals)
        self.get_ovlp = get_ovlp

        self._split_train_idx = torch.load(split_train) if split_train else None
        self._split_val_idx = torch.load(split_val) if split_val else None
        self._split_test_idx = torch.load(split_test) if split_test else None

        if self._split_train_idx is not None:
            log.info(f"Loaded train split filter with {len(self._split_train_idx)} indices")
        if self._split_val_idx is not None:
            log.info(f"Loaded val split filter with {len(self._split_val_idx)} indices")
        if self._split_test_idx is not None:
            log.info(f"Loaded test split filter with {len(self._split_test_idx)} indices")

        super().__init__(self.folder, transform, pre_filter)

        if root_train is not None and root_val is not None and root_test is not None:
            self._databases = []
            self._database_ranges = []
            current_start = 0
            for path in [root_train, root_val, root_test]:
                if path is not None and os.path.exists(path):
                    db = HamiltonianDatabase(path)
                    db_size = len(db)
                    self._databases.append(db)
                    self._database_ranges.append((current_start, current_start + db_size))
                    current_start += db_size
                    log.info(f"Loaded database {path} with {db_size} samples")
            self.n_conformations = current_start
            self._create_virtual_splits()
        else:
            if self.db_path is None:
                raise ValueError("Either db_path or (root_train, root_val, root_test) must be provided")
            self._database = HamiltonianDatabase(self.db_path)
            self.n_conformations = len(self._database)
            self._create_single_database_splits(split_path, train_samples)

        log.info(f"Total conformation number is {self.n_conformations}.")

    def _create_virtual_splits(self):
        train_start, train_end = self._database_ranges[0]
        val_start, val_end = self._database_ranges[1]
        test_start, test_end = self._database_ranges[2]

        train_indices = (
            (train_start + self._split_train_idx)
            if self._split_train_idx is not None
            else np.arange(train_start, train_end)
        )
        val_indices = (
            (val_start + self._split_val_idx) if self._split_val_idx is not None else np.arange(val_start, val_end)
        )
        test_indices = (
            (test_start + self._split_test_idx) if self._split_test_idx is not None else np.arange(test_start, test_end)
        )

        self.molecular_splits = (train_indices, val_indices, test_indices)
        self.n_conformations = test_end
        log.info(f"Virtual splits: train({len(train_indices)}), val({len(val_indices)}), test({len(test_indices)})")

    def _create_single_database_splits(self, split_path, train_samples):
        try:
            if split_path is None:
                split_path = os.path.join(self.folder, "splits.pt")
            else:
                split_path = osp.join(self.folder, split_path)
            self.molecular_splits = torch.load(split_path)
            log.info(os.path.basename(split_path))
        except FileNotFoundError:
            indices = np.arange(self.n_conformations)
            n = train_samples if train_samples is not None else self.n_conformations
            train_indices = np.random.choice(indices, n, replace=False)
            self.molecular_splits = (train_indices, indices, indices)
            log.info("no splits file found, using random splits.")

        if self._split_train_idx is not None or self._split_val_idx is not None or self._split_test_idx is not None:
            train_indices, val_indices, test_indices = self.molecular_splits
            if self._split_train_idx is not None:
                train_indices = train_indices[self._split_train_idx]
            if self._split_val_idx is not None:
                val_indices = val_indices[self._split_val_idx]
            if self._split_test_idx is not None:
                test_indices = test_indices[self._split_test_idx]
            self.molecular_splits = (train_indices, val_indices, test_indices)
            self.n_conformations = len(train_indices) + len(val_indices) + len(test_indices)

    def _get_database_and_local_idx(self, global_idx):
        if hasattr(self, "_databases"):
            if global_idx < self._database_ranges[0][1]:
                return self._databases[0], global_idx - self._database_ranges[0][0]
            elif global_idx < self._database_ranges[1][1]:
                return self._databases[1], global_idx - self._database_ranges[1][0]
            else:
                return self._databases[2], global_idx - self._database_ranges[2][0]
        else:
            return self._database, global_idx

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
        return [f"{self.name}.db"]

    def get(self, idx):
        database, local_idx = self._get_database_and_local_idx(idx)
        Z, R, E, F, H, S, C, moses_id, conformer_id = database[local_idx]

        atoms = Z
        pos = R
        if not self.unit_transfer:
            pos = pos / BOHR2ANG

        num_orbitals = compute_num_orbitals(atoms, self.orbital_mask)
        mol_id = f"{moses_id}_{conformer_id}"
        ovlp = S
        Ham_list = [H]
        pos = pos.reshape(-1, 3)
        return num_orbitals, mol_id, atoms, pos, Ham_list, ovlp
