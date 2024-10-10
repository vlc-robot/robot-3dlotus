"""
Distributed tools
"""
from typing import Tuple, Union, Dict, Any

import os
import pickle

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .logger import LOGGER


def get_local_rank() -> int:
    if os.environ.get("LOCAL_RANK", "") != "":
        local_rank = int(os.environ["LOCAL_RANK"])
    elif os.environ.get("SLURM_LOCALID", "") != "":
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        local_rank = -1
    return local_rank

def load_init_param(opts):
    """
    Load parameters for the rendezvous distributed procedure
    """
    # num of gpus per node
    # WARNING: this assumes that each node has the same number of GPUs
    if os.environ.get("SLURM_NTASKS_PER_NODE", "") != "":   # JZ
        num_gpus = int(os.environ['SLURM_NTASKS_PER_NODE'])
    elif os.environ.get("SLURM_TASKS_PER_NODE", "") != "":  # CLEPS
        num_gpus = int(os.environ['SLURM_TASKS_PER_NODE'])
    else:
        num_gpus = torch.cuda.device_count()

    # world size
    if os.environ.get("WORLD_SIZE", "") != "":
        world_size = int(os.environ["WORLD_SIZE"])
    elif os.environ.get("SLURM_JOB_NUM_NODES", ""): # JZ
        num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        world_size = num_nodes * num_gpus
    elif os.environ.get("SLURM_NNODES", "") != "": # CLEPS
        num_nodes = int(os.environ["SLURM_NNODES"])
        world_size = num_nodes * num_gpus
    else:
        raise RuntimeError("Can't find any world size")
    opts.world_size = world_size

    # rank
    if os.environ.get("RANK", "") != "":
        # pytorch.distributed.launch provide this variable no matter what
        rank = int(os.environ["RANK"])
    elif os.environ.get("SLURM_PROCID", "") != "":
        rank = int(os.environ["SLURM_PROCID"])
    else:
        if os.environ.get("NODE_RANK", "") != "":
            opts.node_rank = int(os.environ["NODE_RANK"])
        elif os.environ.get("SLURM_NODEID", "") != "":
            opts.node_rank = int(os.environ["SLURM_NODEID"])
        else:
            raise RuntimeError("Can't find any rank or node rank")
        rank = opts.local_rank + opts.node_rank * num_gpus
    opts.rank = rank

    init_method = "env://" # need to specify MASTER_ADDR and MASTER_PORT
    
    return {
        "backend": "nccl",
        "init_method": init_method,
        "rank": opts.rank,
        "world_size": opts.world_size,
    }

def init_distributed(opts):
    init_param = load_init_param(opts)
    rank = init_param["rank"]
    print(f"Init distributed {init_param['rank']} - {init_param['world_size']}")

    dist.init_process_group(**init_param)

def is_default_gpu(opts) -> bool:
    return opts.local_rank == -1 or dist.get_rank() == 0

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def set_cuda(opts) -> Tuple[bool, int, torch.device]:
    """
    Initialize CUDA for distributed computing
    """
    local_rank = get_local_rank()
    opts.local_rank = local_rank

    if not torch.cuda.is_available():
        assert local_rank == -1, local_rank
        return True, 0, torch.device("cpu")

    # get device settings
    if opts.local_rank != -1:
        init_distributed(opts)
        torch.cuda.set_device(opts.local_rank)
        device = torch.device("cuda", opts.local_rank)
        n_gpu = 1
        default_gpu = dist.get_rank() == 0
        if default_gpu:
            LOGGER.info(f"Found {dist.get_world_size()} GPUs")
    else:
        default_gpu = True
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()

    return default_gpu, n_gpu, device


def wrap_model(
    model: torch.nn.Module, device: torch.device, local_rank: int,
    find_unused_parameters: bool = False
) -> torch.nn.Module:
    model.to(device)

    if local_rank != -1:
        model = DDP(
            model, device_ids=[local_rank], find_unused_parameters=find_unused_parameters
        )
        # At the time of DDP wrapping, parameters and buffers (i.e., model.state_dict()) 
        # on rank0 are broadcasted to all other ranks.
    elif torch.cuda.device_count() > 1:
        LOGGER.info("Using data parallel")
        model = torch.nn.DataParallel(model)

    return model
