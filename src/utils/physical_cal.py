from argparse import Namespace
from typing import Literal

import cupy
import numpy as np
import pyscf
import torch
import torch.nn as nn
from gpu4pyscf.dft.rks import RKS
from torch_geometric.data import Data

from .utils import type_as

convention_dict = {
    "pyscf_631G": Namespace(
        atom_to_orbitals_map={1: "ss", 6: "ssspp", 7: "ssspp", 8: "ssspp", 9: "ssspp"},
        orbital_idx_map={"s": [0], "p": [1, 2, 0], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={1: [0, 1], 6: [0, 1, 2, 3, 4], 7: [0, 1, 2, 3, 4], 8: [0, 1, 2, 3, 4], 9: [0, 1, 2, 3, 4]},
    ),
    "psi4_def2svp": Namespace(
        atom_to_orbitals_map={
            1: "ssp",
            6: "sssppd",
            7: "sssppd",
            8: "sssppd",
            9: "sssppd",
            15: "sssspppd",
            16: "sssspppd",
            17: "sssspppd",
        },
        orbital_idx_map={"s": [0], "p": [0, 1, 2], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2],
            6: [0, 1, 2, 3, 4, 5],
            7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
            15: [0, 1, 2, 3, 4, 5, 6, 7],
            16: [0, 1, 2, 3, 4, 5, 6, 7],
            17: [0, 1, 2, 3, 4, 5, 6, 7],
        },
    ),
    "back2psi4": Namespace(
        atom_to_orbitals_map={
            1: "ssp",
            6: "sssppd",
            7: "sssppd",
            8: "sssppd",
            9: "sssppd",
            15: "sssspppd",
            16: "sssspppd",
            17: "sssspppd",
        },
        orbital_idx_map={"s": [0], "p": [0, 1, 2], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2],
            6: [0, 1, 2, 3, 4, 5],
            7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
            15: [0, 1, 2, 3, 4, 5, 6, 7],
            16: [0, 1, 2, 3, 4, 5, 6, 7],
            17: [0, 1, 2, 3, 4, 5, 6, 7],
        },
    ),
    "pyscf_def2svp": Namespace(
        atom_to_orbitals_map={
            1: "ssp",
            6: "sssppd",
            7: "sssppd",
            8: "sssppd",
            9: "sssppd",
            15: "sssspppd",
            16: "sssspppd",
            17: "sssspppd",
        },
        orbital_idx_map={"s": [0], "p": [1, 2, 0], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2],
            6: [0, 1, 2, 3, 4, 5],
            7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
            15: [0, 1, 2, 3, 4, 5, 6, 7],
            16: [0, 1, 2, 3, 4, 5, 6, 7],
            17: [0, 1, 2, 3, 4, 5, 6, 7],
        },
    ),
    "back2pyscf": Namespace(
        atom_to_orbitals_map={
            1: "ssp",
            6: "sssppd",
            7: "sssppd",
            8: "sssppd",
            9: "sssppd",
            15: "sssspppd",
            16: "sssspppd",
            17: "sssspppd",
        },
        orbital_idx_map={"s": [0], "p": [2, 0, 1], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2],
            6: [0, 1, 2, 3, 4, 5],
            7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
            15: [0, 1, 2, 3, 4, 5, 6, 7],
            16: [0, 1, 2, 3, 4, 5, 6, 7],
            17: [0, 1, 2, 3, 4, 5, 6, 7],
        },
    ),
}


def matrix_transform(
    matrices: np.ndarray | torch.Tensor,
    atoms: list[int],
    convention: Literal["pyscf_631G"] = "pyscf_631G",
    tc: bool = False,
) -> np.ndarray | torch.Tensor:
    conv = convention_dict[convention]
    orbitals = ""
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int32)
    transform_signs = np.concatenate(transform_signs)

    if tc:
        transform_indices = torch.from_numpy(transform_indices).to(matrices.device).type(torch.long)
        transform_signs = torch.from_numpy(transform_signs).to(matrices.device)

    matrices_new = matrices[..., transform_indices, :]
    matrices_new = matrices_new[..., :, transform_indices]
    matrices_new = matrices_new * transform_signs[:, None]
    matrices_new = matrices_new * transform_signs[None, :]
    return matrices_new


def cut_matrix(
    matrix: torch.Tensor, atoms: list[int] | torch.Tensor, full_orbitals: int, orbital_mask: dict[int, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    all_diagonal_matrix_blocks = []
    all_non_diagonal_matrix_blocks = []
    all_diagonal_matrix_block_masks = []
    all_non_diagonal_matrix_block_masks = []
    col_idx = 0
    edge_index_full = []
    for idx_i, atom_i in enumerate(atoms):  # (src)
        row_idx = 0
        atom_i = atom_i.item()
        mask_i = orbital_mask[atom_i]
        for idx_j, atom_j in enumerate(atoms):  # (dst)
            edge_index_full.append([idx_j, idx_i])
            atom_j = atom_j.item()
            mask_j = orbital_mask[atom_j]
            matrix_block = torch.zeros(full_orbitals, full_orbitals).type_as(matrix)
            matrix_block_mask = torch.zeros(full_orbitals, full_orbitals).type_as(matrix)
            extracted_matrix = matrix[row_idx : row_idx + len(mask_j), col_idx : col_idx + len(mask_i)]
            # for matrix_block
            tmp = matrix_block[mask_j]
            tmp[:, mask_i] = extracted_matrix
            matrix_block[mask_j] = tmp
            tmp = matrix_block_mask[mask_j]
            tmp[:, mask_i] = 1
            matrix_block_mask[mask_j] = tmp
            if idx_i == idx_j:
                all_diagonal_matrix_blocks.append(matrix_block)
                all_diagonal_matrix_block_masks.append(matrix_block_mask)
            else:
                all_non_diagonal_matrix_blocks.append(matrix_block)
                all_non_diagonal_matrix_block_masks.append(matrix_block_mask)
            row_idx = row_idx + len(mask_j)
        col_idx = col_idx + len(mask_i)
    return (
        torch.stack(all_diagonal_matrix_blocks, dim=0),
        torch.stack(all_non_diagonal_matrix_blocks, dim=0),
        torch.stack(all_diagonal_matrix_block_masks, dim=0),
        torch.stack(all_non_diagonal_matrix_block_masks, dim=0),
        torch.tensor(edge_index_full).transpose(-1, -2),
    )


def batch_shape_matrix(
    net: nn.Module,
    batch: Data,
    ham_diag: torch.Tensor,
    ham_non_diag: torch.Tensor,
    package: Literal["pyscf", "psi4"] = "pyscf",
) -> list[torch.Tensor]:
    hamiltonian = net.build_final_matrix(batch, ham_diag, ham_non_diag)
    ham_list = []
    convention = "back2pyscf" if package == "pyscf" else "back2psi4"
    for mol_idx, matrix in enumerate(hamiltonian):
        atoms = batch.atoms[batch.ptr[mol_idx] : batch.ptr[mol_idx + 1]]
        ham_idx = matrix_transform(matrix, atoms.cpu().squeeze().numpy(), convention=convention, tc=True)
        ham_list.append(ham_idx)
    return ham_list


def build_rks(
    atoms: torch.Tensor | np.ndarray | list[int],
    pos: torch.Tensor | np.ndarray | list[list[float]],
):
    if isinstance(atoms, torch.Tensor):
        atoms = atoms.squeeze().cpu().tolist()
    elif isinstance(atoms, np.ndarray):
        atoms = atoms.squeeze()
    if isinstance(pos, torch.Tensor):
        pos = pos.squeeze().cpu().tolist()
    elif isinstance(pos, np.ndarray):
        pos = pos.squeeze()

    t = [[atoms[i], pos[i]] for i in range(len(atoms))]
    mol = pyscf.gto.Mole()
    mol.build(verbose=0, atom=t, basis="def2svp", unit="ang")
    mf = RKS(mol)
    mf.xc = "b3lyp"
    mf.grids.build()
    mf.grids.weights = cupy.asarray(mf.grids.weights)
    mf.grids.coords = cupy.asarray(mf.grids.coords)
    mf._numint.build(mf.mol, mf.grids.coords)
    return mf


def cal_orbital_and_energies(
    overlap_matrix: torch.Tensor,
    full_hamiltonian: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
    eps = 1e-8 * torch.ones_like(eigvals)
    eigvals = torch.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)

    Fs = torch.bmm(torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian), frac_overlap)
    orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
    orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)
    return orbital_energies, orbital_coefficients


def get_mo_occ(
    mo_energy: torch.Tensor,
    atoms: torch.Tensor = None,
    mol: pyscf.gto.Mole = None,
) -> torch.Tensor:
    if atoms is not None:
        nocc = atoms.sum() // 2
    else:
        nocc = mol.nelectron // 2
    e_idx = torch.range(0, len(mo_energy) - 1).long()
    nmo = mo_energy.size(0)
    mo_occ = torch.zeros(nmo).type_as(mo_energy)
    mo_occ[e_idx[:nocc]] = 2
    return mo_occ


def make_rdm1(mo_coeff: torch.Tensor, mo_occ: torch.Tensor) -> torch.Tensor:
    mocc = mo_coeff[:, mo_occ > 0]
    dm = (mocc * mo_occ[mo_occ > 0]).mm(mocc.T.conj())
    return dm


def get_dipole(density_matrix: torch.Tensor, mol: pyscf.gto.Mole) -> torch.Tensor:
    with mol.with_common_origin([0, 0, 0]):
        dip_ints = torch.tensor(mol.intor_symmetric("int1e_r", comp=3)).type_as(density_matrix)
        nuc_dip = torch.tensor(mol.atom_charges().dot(mol.atom_coords())).type_as(density_matrix)
    elec_dip = torch.einsum("xij,ji->x", dip_ints, density_matrix)
    dipole = nuc_dip - elec_dip
    return dipole


def batch_make_rdm1(
    batch: Data,
    ham_list: list[torch.Tensor],
    flatten: bool = False,
) -> tuple[list[torch.Tensor] | torch.Tensor, list[torch.Tensor]]:
    dm_list = []
    mo_coeff_list = []
    batch_ovlp = (
        batch.ovlp.reshape(-1, batch.ovlp.shape[-1], batch.ovlp.shape[-1])
        if isinstance(batch.ovlp, torch.Tensor)
        else batch.ovlp
    )
    for mol_idx, fock in enumerate(ham_list):
        start, end = batch.ptr[mol_idx], batch.ptr[mol_idx + 1]
        atoms = batch.atoms[start:end]
        ovlp = type_as(batch_ovlp[mol_idx], batch.pos)
        orb_ener, orb_coeff = cal_orbital_and_energies(ovlp.unsqueeze(0), fock.unsqueeze(0))
        mo_occ = get_mo_occ(orb_ener[0], atoms=atoms)
        dm = make_rdm1(orb_coeff[0], mo_occ)
        dm_list.append(dm)
        mo_coeff_list.append(orb_coeff[0])
    if flatten:
        dm_list = torch.cat([dm.reshape(-1) for dm in dm_list])
    return dm_list, mo_coeff_list


def get_properties_error(
    net: nn.Module,
    batch: Data,
    z_pred: torch.Tensor,
    z_gt: torch.Tensor,
    ovlps: list[np.ndarray] = None,
    package: Literal["pyscf", "psi4"] = "pyscf",
):
    n_diag = len(batch.diag_ham_mask)
    hamiltonian_pred = net.build_final_matrix(batch, z_pred[:n_diag], z_pred[n_diag:])
    hamiltonian_gt = net.build_final_matrix(batch, z_gt[:n_diag], z_gt[n_diag:])

    batch_properties = []
    for mol_idx, (ham_pred, ham_gt) in enumerate(zip(hamiltonian_pred, hamiltonian_gt)):
        properties = {}
        atoms = batch.atoms[batch.ptr[mol_idx] : batch.ptr[mol_idx + 1]]
        pos = batch.pos[batch.ptr[mol_idx] : batch.ptr[mol_idx + 1]]

        atoms_np = atoms.squeeze().cpu().numpy()
        pos_np = pos.squeeze().cpu().numpy()
        mol = pyscf.gto.Mole()
        t = [[atoms_np[i], pos_np[i]] for i in range(len(atoms))]
        mol.build(verbose=0, atom=t, basis="def2svp", unit="ang")

        convention = "back2pyscf" if package == "pyscf" else "back2psi4"
        ham_pred = matrix_transform(ham_pred, atoms.cpu().squeeze().numpy(), convention=convention, tc=True)

        ham_gt = matrix_transform(ham_gt, atoms.cpu().squeeze().numpy(), convention=convention, tc=True)

        if ovlps is not None:
            ovlp = ovlps[mol_idx]
        elif hasattr(batch, "ovlp"):
            batch_ovlp = (
                batch.ovlp.reshape(-1, batch.ovlp.shape[-1], batch.ovlp.shape[-1])
                if isinstance(batch.ovlp, torch.Tensor)
                else batch.ovlp
            )
            ovlp = batch_ovlp[mol_idx]
        else:
            ovlp = mol.intor("int1e_ovlp")
        overlap = type_as(ovlp, batch.pos)
        orb_ener_pred, orb_coeff_pred = cal_orbital_and_energies(overlap.unsqueeze(0), ham_pred.unsqueeze(0))
        orb_ener_gt, orb_coeff_gt = cal_orbital_and_energies(overlap.unsqueeze(0), ham_gt.unsqueeze(0))
        num_orb = int(atoms.sum() / 2)
        properties["ham_pred"], properties["ham_gt"] = ham_pred, ham_gt
        properties["HOMO_pred"], properties["HOMO_gt"] = orb_ener_pred[:, num_orb - 1], orb_ener_gt[:, num_orb - 1]
        properties["LUMO_pred"], properties["LUMO_gt"] = orb_ener_pred[:, num_orb], orb_ener_gt[:, num_orb]
        properties["ener_pred"], properties["coeff_pred"], properties["ener_gt"], properties["coeff_gt"] = (
            orb_ener_pred[:, :num_orb],
            orb_coeff_pred[:, :, :num_orb],
            orb_ener_gt[:, :num_orb],
            orb_coeff_gt[:, :, :num_orb],
        )

        properties["dm_pred"], properties["dm_gt"] = (
            make_rdm1(orb_coeff_pred[0], get_mo_occ(orb_ener_pred[0], atoms=atoms)),
            make_rdm1(orb_coeff_gt[0], get_mo_occ(orb_ener_gt[0], atoms=atoms)),
        )

        properties["dipole_pred"], properties["dipole_gt"] = (
            get_dipole(properties["dm_pred"], mol),
            get_dipole(properties["dm_gt"], mol),
        )

        batch_properties.append(properties)
    return batch_properties
