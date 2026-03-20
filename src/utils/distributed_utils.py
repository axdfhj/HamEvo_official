"""Utilities for distributed training and data collection."""

from typing import Any

import torch
import torch.distributed as dist

from . import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=False)


def is_distributed() -> bool:
    """Check if we are in a distributed environment."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get the rank of the current process."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_rank_zero() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def gather_object(obj: Any, dst: int = 0) -> list[Any] | None:
    """
    Gather objects from all processes to the destination process.

    Args:
        obj: Object to gather from each process
        dst: Destination rank (default: 0)

    Returns:
        List of objects from all processes if on destination rank, None otherwise
    """
    if not is_distributed():
        return [obj]

    world_size = get_world_size()

    if get_rank() == dst:
        output = [None for _ in range(world_size)]
    else:
        output = None

    dist.gather_object(obj, output, dst=dst)
    return output


def all_gather_object(obj: Any) -> list[Any]:
    """
    Gather objects from all processes to all processes.

    Args:
        obj: Object to gather from each process

    Returns:
        List of objects from all processes
    """
    if not is_distributed():
        return [obj]

    world_size = get_world_size()
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, obj)
    return output


def gather_tensors(tensor: torch.Tensor, dst: int = 0) -> torch.Tensor | None:
    """
    Gather tensors from all processes to the destination process.

    Args:
        tensor: Tensor to gather from each process
        dst: Destination rank (default: 0)

    Returns:
        Concatenated tensor from all processes if on destination rank, None otherwise
    """
    if not is_distributed():
        return tensor

    world_size = get_world_size()

    # Get tensor shapes from all processes
    shape = torch.tensor(tensor.shape, device=tensor.device)
    shape_list = [torch.zeros_like(shape) for _ in range(world_size)]
    dist.all_gather(shape_list, shape)

    # Prepare gather list
    if get_rank() == dst:
        gather_list = []
        for i in range(world_size):
            gather_shape = tuple(shape_list[i].cpu().numpy())
            gather_list.append(torch.zeros(gather_shape, dtype=tensor.dtype, device=tensor.device))
    else:
        gather_list = None

    # Gather tensors
    dist.gather(tensor, gather_list, dst=dst)

    # Concatenate on destination rank
    if get_rank() == dst:
        return torch.cat(gather_list, dim=0)
    return None


def all_gather_tensors(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes to all processes.

    Args:
        tensor: Tensor to gather from each process

    Returns:
        Concatenated tensor from all processes
    """
    if not is_distributed():
        return tensor

    world_size = get_world_size()

    # Get tensor shapes from all processes
    shape = torch.tensor(tensor.shape, device=tensor.device)
    shape_list = [torch.zeros_like(shape) for _ in range(world_size)]
    dist.all_gather(shape_list, shape)

    # Prepare gather list
    gather_list = []
    for i in range(world_size):
        gather_shape = tuple(shape_list[i].cpu().numpy())
        gather_list.append(torch.zeros(gather_shape, dtype=tensor.dtype, device=tensor.device))

    # Gather tensors
    dist.all_gather(gather_list, tensor)

    # Concatenate
    return torch.cat(gather_list, dim=0)


def gather_dict(data_dict: dict[str, Any], dst: int = 0) -> dict[str, Any] | None:
    """
    Gather dictionaries from all processes to the destination process.

    Args:
        data_dict: Dictionary to gather from each process
        dst: Destination rank (default: 0)

    Returns:
        Combined dictionary if on destination rank, None otherwise
    """
    # Gather all dictionaries
    all_dicts = gather_object(data_dict, dst=dst)

    if all_dicts is None:
        return None

    # Combine dictionaries
    combined = {}
    for d in all_dicts:
        for key, value in d.items():
            if key not in combined:
                combined[key] = []

            if isinstance(value, list):
                combined[key].extend(value)
            else:
                combined[key].append(value)

    return combined


def synchronize():
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()
