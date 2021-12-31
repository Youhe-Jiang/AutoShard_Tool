import torch
import torch.nn as nn
import copy
from typing import Optional, Any, Union, Callable

import torch
import functools
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from torch.optim import Adam
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    get_full_params,
)
from torch.testing._internal.common_utils import TEST_WITH_DEV_DBG_ASAN, run_tests
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper, disable_checkpointing
import time
import numpy as np
from AutoShard import AutoShard
from fairscale.nn import auto_wrap, default_auto_wrap_policy, enable_wrap, wrap


class AutoShard:
    def __init__(self, module, sharding_ratio, in_features, out_features):
        self.module = module
        self.sharding_ratio = sharding_ratio
        self.in_features = in_features
        self.out_features = out_features
        self.sharded_parameters = 0
        self.full_parameters = sum(m.numel() for m in module.parameters())

    def recursive_search(self, module):
        for name, child in module.named_children():
            if sum(m.numel() for m in child.parameters()) > sum(m.numel() for m in nn.Linear(self.in_features, self.out_features, bias=True).parameters()):
                self.recursive_search(child)
            else:
                self.sharded_parameters += sum(m.numel() for m in child.parameters())
                if self.sharded_parameters < self.full_parameters * self.sharding_ratio and sum(m.numel() for m in child.parameters()) > sum(m.numel() for m in nn.LayerNorm(self.in_features).parameters()):
                    wrapped_child = wrap(child)
                    setattr(module, name, wrapped_child)
        return module

    def gen_model(self):
        module = self.recursive_search(self.module)
        return module

class TestUnevenParamShard(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_one_iteration(self):
        module = nn.Transformer(nhead=16, num_encoder_layers=12)
        with enable_wrap(wrapper_cls=FSDP):
            module = AutoShard(module=module,
                               sharding_ratio=0.35,
                               in_features=512,
                               out_features=2048).gen_model()
        print(module)
if __name__ == "__main__":
    run_tests()
