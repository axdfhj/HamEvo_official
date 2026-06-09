import random
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Literal

import numpy as np
import torch
import torch_ema
from e3nn import o3
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torchmetrics import MeanMetric, MinMetric

from src.utils import (
    ClipGradNorm,
    MetricRegistry,
    RankedLogger,
    batch_make_rdm1,
    batch_shape_matrix,
    get_properties_error,
    is_rank_zero,
    jac_loss_estimate,
)

BOHR2ANG = 1.8897259886
HARTREE_BOHR_TO_KCAL_ANG = 1185.82
log = RankedLogger(__name__, rank_zero_only=True)


class DEQHamLitModule(LightningModule):
    """Deep Equilibrium Hamiltonian prediction module."""

    def __init__(
        self,
        lit_cfg: DictConfig,
        net: nn.Module,
        distance_expansion: nn.Module,
        node_emb: nn.Module,
        criterion: nn.Module,
        f_solver: Callable | None,
        b_solver: Callable | None,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        ema: Callable | None = None,
        clip_grad_norm: ClipGradNorm | None = None,
        compile: bool = False,
        ckpt_cfg: DictConfig = None,
        load_ckpt: bool = True,
    ) -> None:
        """Initialize DEQHamLitModule."""
        super().__init__()

        self.lit_cfg = lit_cfg
        self.save_hyperparameters(logger=False)

        # deq module
        self.func = net
        self.distance_expansion = distance_expansion
        self.node_emb = node_emb

        # solver
        self.f_solver = f_solver
        self.b_solver = b_solver

        # loss function
        self.criterion = criterion

        # init ema
        if self.hparams.ema is not None:
            self.ema: torch_ema.ExponentialMovingAverage = self.hparams.ema(
                filter(lambda p: p.requires_grad, self.parameters())
            )

        if self.hparams.ckpt_cfg and self.hparams.load_ckpt:
            checkpoint = torch.load(self.hparams.ckpt_cfg.ckpt)
            if self.hparams.ckpt_cfg.ema_param and "ema_state_dict" in checkpoint.keys():
                self.ema.load_state_dict(checkpoint["ema_state_dict"])
                self.ema.copy_to()
            else:
                self.load_state_dict(checkpoint["state_dict"], strict=False)
            # reinit ema if loading weight from checkpoint.
            self.ema: torch_ema.ExponentialMovingAverage = self.hparams.ema(
                filter(lambda p: p.requires_grad, self.parameters())
            )
            log.info(f"Loading ckpt from {self.hparams.ckpt_cfg.ckpt}.")

        if self.hparams.lit_cfg.get("masked_finetune", False):
            n_atoms = self.hparams.lit_cfg.masked_finetune.n_atoms
            full_orbitals = self.hparams.lit_cfg.masked_finetune.full_orbitals
            self.func.set_masked_finetune(n_atoms, full_orbitals)

        self.automatic_optimization = False

    def _init_metrics(self):
        """Initialize metrics using MetricRegistry."""
        self.val_loss = MeanMetric()
        self.val_mae_best = MinMetric()

        self.val_registry = MetricRegistry()
        self.test_registry = MetricRegistry()

        test_metrics = [
            "test_loss",
            "sample_Ham",
            "orb_coeff",
            "orb_ener",
            "HOMO_err",
            "LUMO_err",
            "HOMO_pred",
            "HOMO_gt",
            "tot_ener_err_diag",
            "tot_ener_err_non_diag",
            "hf_force_err",
        ]
        for metric in test_metrics:
            self.test_registry.register(metric, "mean")

        self.val_registry.to(self.device)
        self.test_registry.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (not used directly — inference goes through model_step)."""
        raise NotImplementedError

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.trainer.save_checkpoint("model_after_sanity_check.ckpt")
        self.val_loss.reset()

    def _solve_fixed_point(self, batch: Data, f_thres: float, use_ema: bool = False) -> dict:
        """Run the forward DEQ solver to find the fixed point."""
        with torch.no_grad():
            ema_env = self.ema.average_parameters() if use_ema else nullcontext()
            start_time = time.time()
            with ema_env:
                result = self.f_solver(
                    lambda z: self.func(z, batch),
                    self.concat_z(batch["z"]),
                    threshold=f_thres,
                )
            end_time = time.time()
            if is_rank_zero():
                log.debug(f"f_solver time: {end_time - start_time} seconds")
        return result

    def _pretrain_step(self, batch: Data) -> dict:
        """Single-step pretrain: predict next SCF iteration from current."""
        bs = len(batch.ptr) - 1
        mask = torch.cat([batch["diag_ham_mask"], batch["non_diag_ham_mask"]])

        z_input = self.concat_z(batch["z_pre"])
        new_z = self.func(z_input, batch)

        gt_z = self.concat_z(batch["z_after"])
        loss_dict = self.criterion(new_z, gt_z, prefix="step", mask=mask)
        loss = loss_dict["step_loss"]

        self.log_dict(loss_dict, prog_bar=True, rank_zero_only=True, batch_size=bs, sync_dist=True)
        return {"loss": loss}

    def _finetune_step(self, batch: Data) -> dict:
        """Fine-tune via implicit differentiation through the fixed point."""
        bs = len(batch.ptr) - 1
        mask = torch.cat([batch["diag_ham_mask"], batch["non_diag_ham_mask"]])
        diag_dim = len(batch["diag_ham_mask"])
        loss_dict = {}

        rand_f_thres = random.randint(-self.lit_cfg.rand_f_thres_delta, 0)
        f_thres = self.lit_cfg.f_thres + (random.randint(-rand_f_thres, 0) if rand_f_thres > 0 else 0)

        with torch.no_grad():
            ema_env = nullcontext()
            start_time = time.time()
            with ema_env:
                result = self.f_solver(
                    lambda z: self.func(z, batch),
                    self.concat_z(batch["z"]),
                    threshold=f_thres,
                )
            end_time = time.time()
            if is_rank_zero():
                log.debug(f"f_solver time: {end_time - start_time} seconds")

        z_pred = result["result"].detach().requires_grad_()
        loss_dict.update({"nstep": result["nstep"], "gx": result["gx"].abs().mean()})
        if is_rank_zero():
            log.debug(f"gx:{result['gx'].abs().mean()}, nstep:{result['nstep']}")

        new_z1s = self.func(z_pred, batch)
        if np.random.uniform(0, 1) < self.lit_cfg.jac_loss_freq:
            jac_loss = jac_loss_estimate(new_z1s, z_pred, vecs=1)
            loss_dict["l_jac"] = jac_loss

        def backward_hook(grad):
            if self.hook is not None:
                self.hook.remove()
                torch.cuda.synchronize()
            with torch.no_grad():
                new_grad = self.b_solver(
                    lambda y: torch.autograd.grad(new_z1s, z_pred, y, retain_graph=True)[0] + grad,
                    torch.zeros_like(grad),
                )
            if is_rank_zero():
                log.debug(f"b_nstep:{new_grad['nstep']}, b_gx:{new_grad['gx'].abs().mean()}")
            new_grad = new_grad["result"]
            return new_grad

        self.hook = new_z1s.register_hook(backward_hook)
        star_loss = self.criterion(
            new_z1s,
            self.concat_z(batch["z_star"]),
            prefix="star",
            mask=mask,
        )
        loss_dict.update(star_loss)

        if self.lit_cfg.get("dm_loss_weight", 0) > 0:
            assert "ovlp" in batch.keys()
            if "dm" in batch.keys():
                gt_dm = torch.cat([torch.tensor(dm).reshape(-1) for dm in batch.dm]).type_as(batch.pos)
            else:
                gt_dm, gt_orb_coeff = batch_make_rdm1(
                    batch,
                    batch_shape_matrix(
                        self.func,
                        batch,
                        batch["z_star"]["diag_ham"],
                        batch["z_star"]["non_diag_ham"],
                        package=self.hparams.lit_cfg.get("package", "pyscf"),
                    ),
                    flatten=True,
                )
            pred_dm, pred_orb_coeff = batch_make_rdm1(
                batch,
                batch_shape_matrix(
                    self.func,
                    batch,
                    new_z1s[:diag_dim],
                    new_z1s[diag_dim:],
                    package=self.hparams.lit_cfg.get("package", "pyscf"),
                ),
                flatten=True,
            )
            dm_loss = self.criterion(pred_dm, gt_dm, prefix="dm")
            loss_dict.update(dm_loss)

        loss = (
            loss_dict["star_loss"]
            + loss_dict.get("l_jac", 0.0) * self.lit_cfg.jac_loss_weight
            + loss_dict.get("dm_loss", 0.0) * self.lit_cfg.dm_loss_weight
        )

        self.log_dict(loss_dict, prog_bar=True, rank_zero_only=True, batch_size=bs, sync_dist=True)
        return {"loss": loss}

    def _predict_step(self, batch: Data) -> dict:
        """Prediction step: solve to fixed point and evaluate properties."""
        bs = len(batch.ptr) - 1
        mask = torch.cat([batch["diag_ham_mask"], batch["non_diag_ham_mask"]])
        diag_dim = len(batch["diag_ham_mask"])
        loss_dict = {}

        result = self._solve_fixed_point(batch, self.lit_cfg.f_thres, use_ema=False)
        z_pred = result["result"].detach().requires_grad_()
        loss_dict.update({"nstep": result["nstep"], "gx": result["gx"].abs().mean()})
        if is_rank_zero():
            log.debug(f"gx:{result['gx'].abs().mean()}, nstep:{result['nstep']}")

        pred_loss = self.criterion(
            z_pred,
            self.concat_z(batch["z_star"]),
            prefix="pred",
            mask=mask,
        )
        loss_dict.update(pred_loss)
        loss = loss_dict["pred_mae"]

        diag_mae = self.criterion(
            z_pred[:diag_dim],
            self.concat_z(batch["z_star"])[:diag_dim],
            prefix="diag",
            mask=mask[:diag_dim],
        )["diag_mae"]
        non_diag_mae = self.criterion(
            z_pred[diag_dim:],
            self.concat_z(batch["z_star"])[diag_dim:],
            prefix="non_diag",
            mask=mask[diag_dim:],
        )["non_diag_mae"]

        if is_rank_zero():
            log.info(f"diag_mae:{diag_mae.item():.8f}, non_diag_mae:{non_diag_mae.item():.8f}")
        if self.trainer.testing:
            self.test_registry.update("tot_ener_err_diag", diag_mae)
            self.test_registry.update("tot_ener_err_non_diag", non_diag_mae)

        batch_properties = get_properties_error(
            self.func,
            batch,
            z_pred,
            self.concat_z(batch["z_star"]),
            batch.get("ovlp", None),
            package=self.hparams.lit_cfg.get("package", "pyscf"),
        )

        for mol_idx, properties in enumerate(batch_properties):
            ham_err = self.criterion(
                properties["ham_pred"],
                properties["ham_gt"],
            )["_mae"]

            self.test_registry.update("sample_Ham", ham_err)

            if "hf_force_pred" in properties.keys() and "hf_force_gt" in properties.keys():
                hf_force_mae = (properties["hf_force_pred"] - properties["hf_force_gt"]).abs().mean()
                self.test_registry.update("hf_force_err", hf_force_mae)

            coeff_cos = torch.cosine_similarity(properties["coeff_pred"], properties["coeff_gt"], dim=1).abs().mean()
            self.test_registry.update("orb_coeff", coeff_cos)

            orb_ener_err = self.criterion(
                properties["ener_pred"],
                properties["ener_gt"],
            )["_mae"]
            self.test_registry.update("orb_ener", orb_ener_err)

            HOMO_err = (properties["HOMO_pred"] - properties["HOMO_gt"]).abs()
            LUMO_err = (properties["LUMO_pred"] - properties["LUMO_gt"]).abs()
            if is_rank_zero():
                log.info(
                    f"natoms:{batch.ptr[mol_idx + 1] - batch.ptr[mol_idx]}, HOMO_err:{HOMO_err.mean().item():.8f}, LUMO_err:{LUMO_err.mean().item():.8f}"
                )
                log.info(f"occupied orbital energy mae:{orb_ener_err:.8f}")
                log.info(
                    f"HOMO_pred:{properties['HOMO_pred'].mean().item():.8f}, HOMO_gt:{properties['HOMO_gt'].mean().item():.8f}"
                )
                log.info(
                    f"LUMO_pred:{properties['LUMO_pred'].mean().item():.8f}, LUMO_gt:{properties['LUMO_gt'].mean().item():.8f}"
                )
                log.info(f"orbital cos-similarity:{coeff_cos:.8f}")
                if "hf_force_pred" in properties.keys() and "hf_force_gt" in properties.keys():
                    log.info(f"HF force error: {hf_force_mae * HARTREE_BOHR_TO_KCAL_ANG:.8f} kcal/mol/ang")

            self.test_registry.update("HOMO_err", HOMO_err)
            self.test_registry.update("HOMO_pred", properties["HOMO_pred"])
            self.test_registry.update("HOMO_gt", properties["HOMO_gt"])
            self.test_registry.update("LUMO_err", LUMO_err)

            loss_dict.update(
                {
                    "coeff_cos": coeff_cos,
                    "orb_ener_err": orb_ener_err,
                    "HOMO_err": HOMO_err,
                    "LUMO_err": LUMO_err,
                }
            )

        self.log_dict(loss_dict, prog_bar=True, rank_zero_only=True, batch_size=bs, sync_dist=True)
        return {"loss": loss}

    def model_step(
        self,
        batch: Data,
        mode: Literal["pt", "ft", "predict"],
    ) -> dict:
        """Dispatch to the appropriate step method based on mode."""
        if mode == "pt":
            return self._pretrain_step(batch)
        elif mode == "ft":
            return self._finetune_step(batch)
        elif mode == "predict":
            return self._predict_step(batch)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def preprocess(
        self,
        batch: Data,
        mode: Literal["pt", "ft", "predict", "infer"] = "pt",
    ) -> Data:

        # Create molecular graph (fully connected within max_num_neighbors)
        atoms = batch.atoms.squeeze()
        edge_index = radius_graph(batch.pos, 100, batch.batch, max_num_neighbors=self.lit_cfg.max_num_neighbors)
        edge_vec = (batch.pos[edge_index[0].long()] - batch.pos[edge_index[1].long()]).requires_grad_()
        batch.edge_index = batch.edge_index_all = edge_index

        # Compute edge features
        vec_sh = o3.Irreps.spherical_harmonics(lmax=self.hparams.lit_cfg.get("order", 4))
        if self.distance_expansion is not None:
            dis_feat = self.distance_expansion(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type_as(batch.pos)
            batch.dis_feat = batch.dis_feat_all = dis_feat
        vec_feat_sh = o3.spherical_harmonics(
            vec_sh, edge_vec[:, [1, 2, 0]], normalize=True, normalization="component"
        ).type_as(batch.pos)
        if self.node_emb is not None:
            node_emb = self.node_emb(atoms.long(), dis_feat, vec_feat_sh, edge_index, pos=batch.pos)
            batch.node_emb = node_emb

        batch.vec_feat_sh = batch.vec_feat_sh_all = vec_feat_sh

        # Compute edge transpose index
        start_edge_index = 0
        all_transpose_index = []
        for graph_idx in range(batch.ptr.shape[0] - 1):
            num_nodes = batch.ptr[graph_idx + 1] - batch.ptr[graph_idx]
            graph_edge_index = edge_index[:, start_edge_index : start_edge_index + num_nodes * (num_nodes - 1)]
            sub_graph_edge_index = graph_edge_index - batch.ptr[graph_idx]
            bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
            transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
            transpose_index = transpose_index + start_edge_index
            all_transpose_index.append(transpose_index)
            start_edge_index = start_edge_index + num_nodes * (num_nodes - 1)
        batch.all_transpose_index = torch.cat(all_transpose_index, dim=-1)

        # Collate DEQ fixed-point variables
        if mode == "pt":
            batch["z_pre"] = Data(
                diag_ham_mask=batch["diag_ham_mask"],
                non_diag_ham_mask=batch["non_diag_ham_mask"],
                diag_ham=batch["pre_diag_ham"],
                non_diag_ham=batch["pre_non_diag_ham"],
            )
            batch["z_after"] = Data(
                diag_ham_mask=batch["diag_ham_mask"],
                non_diag_ham_mask=batch["non_diag_ham_mask"],
                diag_ham=batch["after_diag_ham"],
                non_diag_ham=batch["after_non_diag_ham"],
            )
        elif mode in ["ft", "predict"]:
            batch["z_star"] = Data(
                diag_ham_mask=batch["diag_ham_mask"],
                non_diag_ham_mask=batch["non_diag_ham_mask"],
                diag_ham=batch["stable_diag_ham"],
                non_diag_ham=batch["stable_non_diag_ham"],
            )
            batch["z"] = batch["z_star"]
        elif mode == "infer":
            batch["z"] = Data(
                diag_ham_mask=batch["diag_ham_mask"],
                non_diag_ham_mask=batch["non_diag_ham_mask"],
                diag_ham=torch.zeros_like(batch["diag_ham_mask"]),
                non_diag_ham=torch.zeros_like(batch["non_diag_ham_mask"]),
            )
        else:
            raise KeyError(f"Invalid mode: {mode}")
        return batch

    def optimize_step(self, loss):
        self.optimizers().zero_grad()
        self.manual_backward(loss)
        if self.hparams.clip_grad_norm is not None:
            tot_norm = self.clip_grad_norm(self.parameters())
            if tot_norm > self.clip_grad_norm.max_norm:
                log.info(
                    f"Step-{self.global_step}: Clip gradient norm {tot_norm:.3f} to {self.clip_grad_norm.max_norm:.3f}."
                )
        self.optimizers().step()
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"], prog_bar=True, rank_zero_only=True)
        if not isinstance(self.lr_schedulers(), torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_schedulers().step()
        if self.hparams.ema is not None:
            self.ema.update()

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        batch = self.preprocess(batch, mode=self.lit_cfg.train_mode)
        output = self.model_step(batch, mode=self.lit_cfg.train_mode)

        self.optimize_step(output["loss"])
        return output["loss"]

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        """Perform a single validation step."""
        batch = self.preprocess(batch, mode=self.lit_cfg.val_mode)
        output = self.model_step(batch, mode=self.lit_cfg.val_mode)

        self.val_loss(output["loss"])

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        mae = self.val_loss.compute()
        self.val_mae_best(mae)
        self.log("val/mae", mae, sync_dist=True, prog_bar=True)
        self.log("val/mae_best", self.val_mae_best.compute(), sync_dist=True, prog_bar=True)

        if (
            isinstance(self.lr_schedulers(), torch.optim.lr_scheduler.ReduceLROnPlateau)
            and not self.trainer.sanity_checking
        ):
            self.lr_schedulers().step(mae)
        self.val_loss.reset()
        self.val_registry.reset_all()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step."""
        batch = self.preprocess(batch, mode=self.hparams.lit_cfg.get("test_mode", "predict"))
        output = self.model_step(batch, mode=self.hparams.lit_cfg.get("test_mode", "predict"))

        self.test_registry.update("test_loss", output["loss"])

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.test_registry.log_all(self, prefix="test/", sync_dist=True, prog_bar=True)

        HOMO_pred = self.test_registry.get("HOMO_pred").compute()
        HOMO_gt = self.test_registry.get("HOMO_gt").compute()
        self.log("test/HOMO_ave_err", HOMO_pred - HOMO_gt, sync_dist=True, prog_bar=True)

        self.test_registry.reset_all()

    def setup(self, stage: str) -> None:
        """Lightning setup hook — initializes EMA, grad clipping, metrics."""
        if self.hparams.ema:
            self.ema.to(self.device)

        if self.hparams.clip_grad_norm is not None and stage == "fit":
            self.clip_grad_norm = self.hparams.clip_grad_norm

        if self.hparams.compile and stage == "fit":
            self.func = torch.compile(self.func)

        self._init_metrics()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and LR scheduler."""
        optimizer = self.hparams.optimizer(params=filter(lambda p: p.requires_grad, self.parameters()))
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, "ema"):
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        if hasattr(self, "clip_grad_norm"):
            checkpoint["clip_grad_state_dict"] = self.clip_grad_norm.state_dict()

    def on_load_checkpoint(self, checkpoint):
        log.info(f"Loading checkpoint from global step {checkpoint['global_step']}.")
        if hasattr(self, "ema") and "ema_state_dict" in checkpoint.keys():
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
            self.ema.to(self.device)
        if hasattr(self, "clip_grad_norm") and "clip_grad_norm" in checkpoint.keys():
            self.clip_grad_norm.load_state_dict(checkpoint["clip_grad_state_dict"])

    def concat_z(self, z: torch.Tensor | None) -> torch.Tensor:
        if z is None:
            return None
        z = torch.concat([z["diag_ham"], z["non_diag_ham"]], dim=0) * torch.concat(
            [z["diag_ham_mask"], z["non_diag_ham_mask"]], dim=0
        )
        return z
