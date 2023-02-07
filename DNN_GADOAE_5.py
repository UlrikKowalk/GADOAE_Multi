
from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt


class DNN_GADOAE_5(nn.Module):

    def __init__(self, num_channels, num_dimensions, num_output_classes):
        super().__init__()

        self.num_input = int(num_channels * (num_channels - 1) / 2 + num_dimensions * num_channels)
        self.num_classes = num_output_classes

        self.flatten0 = nn.Flatten()

        # Concatenation: 16+45=61

        # numLayers * 105 + numDims * 15, 1layer: 135 -> 1024  3layers: 360 -> 1024
        self.linear0 = nn.Sequential(
            nn.Linear(in_features=self.num_input, out_features=1024),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        # 1024 -> 1024
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        # 1024 -> 1024
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        # 1024 -> 1024
        self.linear3 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        # 1024 -> 1024
        self.linear4 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        # 1024 -> 72
        self.linear5 = nn.Linear(
            in_features=1024, out_features=self.num_classes
        )

        # 512 -> 512
        self.act = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):

        # input shape: [X, 360]
        x = self.linear0(input_data)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)

        # x = self.softmax(x)

        return x

