from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pyscf
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame


# original pyg Collater
class Collater:
    def __init__(
        self,
        dataset: Dataset | Sequence[BaseData] | DatasetAdapter,
        follow_batch: list[str] | None = None,
        exclude_keys: list[str] | None = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: list[Any]) -> Any:
        elem = batch[0]

        if isinstance(elem, BaseData):
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, dim=0)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


# customed pyg Collater for batch-level operation
class CustomCollater:
    def __init__(
        self,
        dataset: Dataset | Sequence[BaseData] | DatasetAdapter,
        follow_batch: list[str] | None = None,
        exclude_keys: list[str] | None = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: list[Any]) -> Any:
        elem = batch[0]

        if isinstance(elem, BaseData):
            batch = Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
            bs = len(batch.ptr) - 1
            idx = np.random.randint(0, bs)
            start, end = batch.ptr[idx], batch.ptr[idx + 1]
            # make conformations same with sampled one
            smp_atoms, smp_pos = batch.atoms[start:end], batch.pos[start:end]
            batch.atoms = smp_atoms.repeat(bs, *[1] * (smp_atoms.dim() - 1))
            batch.pos = smp_pos.repeat(bs, *[1] * (smp_pos.dim() - 1))
            np_atoms = smp_atoms.squeeze().cpu().numpy()
            np_pos = smp_pos.squeeze().cpu().numpy()
            t = [[np_atoms[i], np_pos[i]] for i in range(len(np_atoms))]
            mol = pyscf.gto.Mole()
            mol.build(verbose=0, atom=t, basis="def2svp", unit="ang")
            batch.eri = mol.intor("int2e", aosym="s1")
            return batch
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, dim=0)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


class CustomPyGDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Dataset | Sequence[BaseData] | DatasetAdapter,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: list[str] | None = None,
        exclude_keys: list[str] | None = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        # kwargs.pop('collate_fn', None)
        collate_fn = kwargs.pop("collate_fn", None)
        assert collate_fn in ["same_conf", None], f"Unsupported collate_fn: {collate_fn}"

        mapping_dict = {
            "same_conf": CustomCollater,
            None: Collater,
        }

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=mapping_dict[collate_fn](dataset, follow_batch, exclude_keys),
            **kwargs,
        )
