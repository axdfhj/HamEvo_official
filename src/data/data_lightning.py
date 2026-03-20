from typing import Any, Literal

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import Sampler
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from src.utils import RankedLogger

from .components.dataloader import CustomPyGDataLoader
from .components.subset import CustomSubset

log = RankedLogger(__name__, rank_zero_only=True)


class DataModule(LightningDataModule):
    """`LightningDataModule` for Hamiltonian prediction datasets.

    Wraps dataset objects and provides train/val/test dataloaders with
    configurable batch sizes, workers, and split strategies.
    """

    def __init__(
        self,
        train_cfg: DictConfig,
        val_cfg: DictConfig,
        test_cfg: DictConfig = None,
        train_sampler: Sampler | None = None,
        dataset: InMemoryDataset = None,
        train_dataset: InMemoryDataset | None = None,
        val_dataset: InMemoryDataset | None = None,
        package: Literal["pyscf", "psi4"] = "pyscf",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["dataset", "train_dataset", "val_dataset"])
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.dtype = dataset.dtype if dataset is not None else train_dataset.dtype
        self.package = package

    def setup(self, stage: str | None = None) -> None:
        train_split = (
            self.dataset.molecular_splits[0] if self.dataset is not None else self.train_dataset.molecular_splits[0]
        )
        self.train_data = CustomSubset(
            self.dataset if self.train_dataset is None else self.train_dataset,
            train_split,
            self.hparams.train_cfg.mode,
            dtype=self.dtype,
            package=self.package,
        )

        val_split = (
            self.dataset.molecular_splits[1] if self.dataset is not None else self.val_dataset.molecular_splits[1]
        )
        self.val_data = CustomSubset(
            self.dataset if self.val_dataset is None else self.val_dataset,
            val_split,
            self.hparams.val_cfg.mode,
            dtype=self.dtype,
            package=self.package,
        )

        if stage in ("test", "predict") and self.dataset is not None:
            test_split = self.dataset.molecular_splits[2]
            test_split = val_split if len(test_split) == 0 else test_split
            self.test_data = CustomSubset(
                self.dataset,
                test_split,
                self.hparams.test_cfg.mode,
                more_info=self.hparams.test_cfg.get("more_info", False),
                dtype=self.dtype,
                package=self.package,
            )
        if self.hparams.train_sampler is not None and stage == "fit":
            weights = self.dataset.weights
            num_samples = len(self.train_data)
            self.train_sampler = self.hparams.train_sampler(weights=weights[train_split], num_samples=num_samples)

    def train_dataloader(self) -> CustomPyGDataLoader:
        if hasattr(self, "train_sampler"):
            log.info(f"Training sampler: {type(self.train_sampler)}")
            sampler = self.train_sampler
        else:
            sampler = None
        return CustomPyGDataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.train_cfg.batch_size,
            num_workers=self.hparams.train_cfg.num_workers,
            pin_memory=self.hparams.train_cfg.pin_memory,
            shuffle=self.hparams.train_cfg.shuffle,
            sampler=sampler,
            collate_fn=self.hparams.train_cfg.get("collate_fn", None),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.val_cfg.batch_size,
            num_workers=self.hparams.val_cfg.num_workers,
            pin_memory=self.hparams.val_cfg.pin_memory,
            shuffle=self.hparams.val_cfg.shuffle,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.test_cfg.batch_size,
            num_workers=self.hparams.test_cfg.num_workers,
            pin_memory=self.hparams.test_cfg.pin_memory,
            shuffle=self.hparams.test_cfg.shuffle,
        )

    def teardown(self, stage: str | None = None) -> None:
        pass

    def state_dict(self) -> dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass
