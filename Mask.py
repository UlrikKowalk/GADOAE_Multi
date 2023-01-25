import torch
import numpy as np

from matplotlib import pyplot as plt


class Mask():

    def __init__(self, device, coordinates, sample_rate):

        self.device = device
        self.coordinates = coordinates
        self.num_channels = self.coordinates.shape[0]
        self.sample_rate = sample_rate
        self.nC = 344.0

    @staticmethod
    def calculate_mic_distance(coord1, coord2):
        return np.sqrt((coord1[0] - coord2[0]) ** 2 +
                         (coord1[1] - coord2[1]) ** 2 +
                         (coord1[2] - coord2[2]) ** 2)

    def mask_2d_tau(self, max_difference_samples):

        mask = torch.zeros(size=(self.num_channels, self.num_channels, 2 * max_difference_samples),
                           device=self.device)

        for ii in range(self.num_channels):
            for jj in range(self.num_channels):

                tau_max = self.calculate_mic_distance(self.coordinates[ii, :], self.coordinates[jj, :])
                tau_max_samples = (np.ceil(tau_max / self.nC * self.sample_rate))

                # mask[ii, jj, int(self.max_difference_samples - tau_max_samples) ] = 0.5
                # mask[ii, jj, int(self.max_difference_samples - tau_max_samples+1) : int(self.max_difference_samples + tau_max_samples-1)] = 1.0
                # mask[ii, jj, int(self.max_difference_samples + tau_max_samples)] = 0.5

                mask[ii, jj, int(max_difference_samples - tau_max_samples): int(
                    max_difference_samples + tau_max_samples)] = 1.0

        # rows = 3
        # cols = 5
        # axes = []
        # fig1 = plt.figure()
        # for a in range(rows * cols):
        #     axes.append(fig1.add_subplot(rows, cols, a + 1))
        #     plt.imshow(mask[a, :, :])
        #     plt.axis('off')
        # fig1.tight_layout()
        # fig1.suptitle('Mask')
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=0.5, wspace=0.05, hspace=0)
        # plt.show()

        return mask


