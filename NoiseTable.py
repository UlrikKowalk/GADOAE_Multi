import math

import numpy as np
import scipy.io as sio
import torch
from matplotlib import pyplot as plt, gridspec


class NoiseTable:

    def __init__(self, noise_table, digits):

        print('Loading noise table...', end = ' ')
        contents_matfile = sio.loadmat(noise_table)
        self.coordinates = contents_matfile['coordinates']
        self.noise_table = contents_matfile['noise_table']
        print('finished.')
        self.digits = digits
        self.max_length = len(self.noise_table)

        # Round / discretize coordinates to certain number of digits
        for idx, node in enumerate(self.coordinates):
            self.coordinates[idx] = [round(i, self.digits) for i in node]

    def lookup(self, coordinates, length):

        output_noise = torch.zeros(size=(len(coordinates), self.max_length))

        # Round / discretize coordinates to certain number of digits
        for idx, node in enumerate(coordinates):
            coordinates[idx] = [round(i, self.digits) for i in node]

        # Number of samples by which noise is rolled for variability
        shift = torch.randint(high=self.max_length, size=(1,))

        for idx_node, node in enumerate(coordinates):
            for idx, coord_set in enumerate(self.coordinates):
                if(all(coord_set == node)):

                    output_noise[idx_node, :] = torch.roll(torch.tensor(self.noise_table[:, idx]), shifts=(shift,), dims=0)

        return self.cut_to_length(output_noise, length)

    def cut_to_length(self, signal, length):

        if signal.dim() == 1:
            desired_len = int(length)
            if signal.shape[0] < desired_len:
                factor = int(np.ceil(desired_len / signal.shape[0]))
                signal = signal.repeat(factor)
                signal = signal[:desired_len]
            elif signal.shape[0] > desired_len:
                signal = signal[:desired_len]
        else:
            desired_len = int(length)
            if signal.shape[1] < desired_len:
                factor = int(np.ceil(desired_len / signal.shape[1]))
                signal = signal.repeat(1, factor)
                signal = signal[:, :desired_len]
            elif signal.shape[1] > desired_len:
                signal = signal[:, :desired_len]

        return signal