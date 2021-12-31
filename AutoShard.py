import torch
import torch.nn as nn
import numpy as np


class AutoShard:
    def __init__(self, module, in_features, out_features, sharding_ratio):
        self.module = module
        self.layers = []
        self.in_features = in_features
        self.out_features = out_features
        self.sharding_ratio = sharding_ratio

    def recursive_search(self, module):
        for child in module.children():
            if sum(m.numel() for m in child.parameters()) > sum(m.numel() for m in nn.Linear(self.in_features, self.out_features, bias=True).parameters()):
                self.recursive_search(child)
            else:
                self.layers.append(child)
        return self.layers

    def gen_model(self):
        layers = self.recursive_search(self.module)
        model = nn.Sequential(*list(layers)[:])
        return model

    def get_sharding_index(self):
        model = self.gen_model()
        model_parameters = sum(m.numel() for m in model.parameters())
        layer = []
        for i in range(len(self.layers)):
            per_layer_parameters = sum(m.numel() for m in model[i].parameters())
            if per_layer_parameters > 0:
                layer_parameters = per_layer_parameters
                layer_index = i
                layer.append((layer_parameters, layer_index))
        np.random.shuffle(layer)
        i = 0
        j = 0
        sharding_index = []
        while i < model_parameters*self.sharding_ratio:
            i += layer[j][0]
            sharding_index.append(layer[j][1])
            j += 1
        return sharding_index
