import time
from contextlib import nullcontext

import hydra
import rootutils
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch_geometric.loader import DataLoader
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import pyscf

from src.data.components.inference import CustomInferData
from src.models.deqham_module import DEQHamLitModule
from src.utils import RankedLogger
from src.utils.physical_cal import cal_orbital_and_energies, get_dipole, get_mo_occ, make_rdm1, matrix_transform

HARTREE_TO_EV = 27.211386

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig):
    assert cfg.ckpt_path, "ckpt_path is required for inference"
    assert cfg.inference.name, "inference.name is required (dataset directory name)"

    log.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
    model: LightningModule = DEQHamLitModule.load_from_checkpoint(cfg.ckpt_path, load_ckpt=False)
    model.eval()

    log.info(f"Loading inference dataset from {cfg.inference.root}/{cfg.inference.name}")
    infer_dataset = CustomInferData(
        root=cfg.inference.root,
        name=cfg.inference.name,
        xyz_file=cfg.inference.xyz_file,
        full_orbitals=cfg.inference.full_orbitals,
        data_type=cfg.data_type,
    )
    log.info(f"Dataset created with {len(infer_dataset)} configurations")

    dataloader = DataLoader(
        infer_dataset,
        batch_size=cfg.inference.batch_size,
        num_workers=cfg.inference.num_workers,
        shuffle=False,
    )

    homo_list, lumo_list, dipole_list = [], [], []
    last_z_pred = None

    with torch.no_grad():
        use_ema = callable(model.ema.average_parameters)
        ema_env = model.ema.average_parameters() if use_ema else nullcontext()
        with ema_env:
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                batch = batch.to(model.device)
                batch = model.preprocess(batch, mode="infer")

                # Warm-start from previous prediction if available
                if last_z_pred is not None:
                    batch.z["diag_ham"] = last_z_pred[: len(batch.diag_ham_mask)]
                    batch.z["non_diag_ham"] = last_z_pred[len(batch.diag_ham_mask) :]

                if not hasattr(batch, "pos") or batch.pos is None:
                    raise ValueError("batch.pos is required but not found")
                if not hasattr(batch, "ovlp") or batch.ovlp is None:
                    raise ValueError("batch.ovlp is required but not found")

                log.info(f"### Batch: {batch_idx:03d} ###")
                start_time = time.time()
                result = model.f_solver(
                    lambda z: model.func(z, batch),
                    model.concat_z(batch["z"]),
                    threshold=cfg.inference.f_thres,
                    eps=cfg.inference.eps,
                )
                end_time = time.time()
                log.info(
                    f"f_solver time: {end_time - start_time:.4f}s, "
                    f"gx: {result['gx'].abs().mean():.6f}, nstep: {result['nstep']}"
                )

                z_pred = result["result"]
                last_z_pred = z_pred.clone()

                n_diag = len(batch.diag_ham_mask)
                hamiltonian_pred = model.func.build_final_matrix(batch, z_pred[:n_diag], z_pred[n_diag:])

                for mol_idx, ham_pred in enumerate(hamiltonian_pred):
                    start, end = batch.ptr[mol_idx].item(), batch.ptr[mol_idx + 1].item()
                    atoms = batch.atoms[start:end]
                    pos = batch.pos[start:end]

                    atoms_np = atoms.squeeze().cpu().numpy()
                    pos_np = pos.squeeze().cpu().numpy()

                    ham_pred = matrix_transform(ham_pred, atoms_np, convention="back2pyscf", tc=True)

                    mol = pyscf.gto.Mole()
                    atom_spec = [[atoms_np[i], pos_np[i]] for i in range(len(atoms_np))]
                    mol.build(verbose=0, atom=atom_spec, basis="def2svp", unit="ang")

                    if isinstance(batch.ovlp, torch.Tensor):
                        batch_ovlp = batch.ovlp.reshape(-1, batch.ovlp.shape[-1], batch.ovlp.shape[-1])
                        overlap = batch_ovlp[mol_idx]
                    else:
                        overlap = torch.tensor(batch.ovlp[mol_idx])
                    overlap = overlap.to(batch.pos.device, batch.pos.dtype)

                    orb_ener_pred, orb_coeff_pred = cal_orbital_and_energies(
                        overlap.unsqueeze(0), ham_pred.unsqueeze(0)
                    )

                    mo_occ = get_mo_occ(orb_ener_pred[0], atoms=atoms)
                    dm = make_rdm1(orb_coeff_pred[0], mo_occ)

                    dipole = get_dipole(dm, mol)
                    eBohr_to_Debye = 2.541746
                    dipole_norm = dipole.norm(dim=-1) * eBohr_to_Debye

                    num_orb = int(atoms.sum() / 2)
                    homo_list.append(orb_ener_pred[:, num_orb - 1].item() * HARTREE_TO_EV)
                    lumo_list.append(orb_ener_pred[:, num_orb].item() * HARTREE_TO_EV)
                    dipole_list.append(dipole_norm.item())

                log.info(
                    f"HOMO: {homo_list[-1]:.4f} eV, LUMO: {lumo_list[-1]:.4f} eV, Dipole: {dipole_list[-1]:.4f} Debye"
                )

    homo_t = torch.tensor(homo_list)
    lumo_t = torch.tensor(lumo_list)
    dipole_t = torch.tensor(dipole_list)
    log.info(f"HOMO: {homo_t.mean().item():.4f} +/- {homo_t.std().item():.4f} eV")
    log.info(f"LUMO: {lumo_t.mean().item():.4f} +/- {lumo_t.std().item():.4f} eV")
    log.info(f"Dipole: {dipole_t.mean().item():.4f} +/- {dipole_t.std().item():.4f} Debye")


if __name__ == "__main__":
    main()
