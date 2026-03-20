import os
import random
from typing import Literal

import numpy as np
import torch
from torch import dtype
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset

from src.utils import cut_matrix, matrix_transform


def sample_n_elements(Ham_list, n):
    mul = len(Ham_list) // n
    if len(Ham_list) >= n:
        return random.sample(Ham_list, n)
    else:
        return Ham_list * mul + random.choices(Ham_list, k=n - mul * len(Ham_list))


class CustomSubset(InMemoryDataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: list[int] | np.ndarray,
        mode: Literal["pt", "ft", "predict"] = "pt",
        dtype: dtype = torch.float32,
        more_info: bool = False,
        package: Literal["pyscf", "psi4"] = "pyscf",
    ) -> None:
        super().__init__()
        self.split_indices = indices
        self.ori_dataset = dataset
        self.mode = mode
        self.dtype = dtype
        self.full_orbitals = self.ori_dataset.full_orbitals
        self.orbital_mask = self.ori_dataset.orbital_mask
        self.more_info = more_info
        self.package = package
        if self.more_info:
            self.dm_ovlp_path = os.path.join(self.ori_dataset.root, "dm_ovlp")

    def len(self):
        return len(self.split_indices)

    def get(self, idx: int) -> Data:
        h_dim, mol_id, atoms, pos, Ham_list, ovlp = self.ori_dataset[self.split_indices[idx]]
        if self.mode == "pt":
            pre_idx = np.random.randint(1, len(Ham_list) - 2)
            selected_dict = {"pre": Ham_list[pre_idx], "after": Ham_list[pre_idx + 1]}
        elif self.mode in ["ft", "predict"]:
            selected_dict = {"stable": Ham_list[-1], "pre": Ham_list[0]}
        else:
            raise KeyError(f"Invalid mode: {self.mode}")
        cut_matrix_dict: dict[str, torch.Tensor] = self.matrix_preprocess(selected_dict, atoms, h_dim)
        data = Data(
            mol_id=mol_id,
            pos=torch.tensor(pos, dtype=self.dtype),
            atoms=torch.tensor(atoms, dtype=torch.long).view(-1, 1),
            **cut_matrix_dict,
        )
        if ovlp is not None:
            data.ovlp = ovlp
        data.orbital_mask = torch.cat(
            [self.orbital_mask[atom] + idx * self.full_orbitals for idx, atom in enumerate(atoms)]
        ).numpy()
        return data

    def matrix_preprocess(
        self,
        selected_dict: dict,
        atoms: np.ndarray,
        h_dim: int,
    ) -> dict[str, torch.Tensor]:
        matrix_dict = {}
        convention = "pyscf_def2svp" if self.package == "pyscf" else "psi4_def2svp"
        trans = lambda x: torch.tensor(matrix_transform(x, atoms, convention=convention), dtype=self.dtype)
        for prefix, ham in selected_dict.items():
            ham = trans(ham)
            (
                matrix_dict[f"{prefix}_diag_ham"],
                matrix_dict[f"{prefix}_non_diag_ham"],
                matrix_dict["diag_ham_mask"],
                matrix_dict["non_diag_ham_mask"],
                matrix_dict["edge_index_full"],
            ) = cut_matrix(ham, atoms, self.full_orbitals, self.orbital_mask)
        result = {}
        for key, value in matrix_dict.items():
            result[key] = value.type(self.dtype)
        return result
