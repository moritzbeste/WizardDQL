import torch
import torch.nn as nn

class TrumpModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        sizes = [input_size, input_size, output_size]
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.extend([
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),
                nn.ReLU(),
            ])
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
