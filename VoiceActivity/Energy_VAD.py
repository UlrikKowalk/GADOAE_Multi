import numpy as np
import torch


class Energy_VAD:

    def __init__(self, sample_rate, framesize, device):

        self.sample_rate = sample_rate
        self.frame_length_s = framesize
        self.frame_length = int(framesize * sample_rate)
        self.device = device
        self.threshold = -4.0

    @staticmethod
    def calculate_level(signal, channels):

        if channels == 1:
            return 10 * np.log10(np.std(signal) + 0.00001) #torch.finfo(torch.float32).eps)
        else:
            return 10 * np.log10(np.std(signal[:channels, :]))

    def get_vadvector_for_allframes(self, data):

        data = data.cpu().detach().numpy()
        timevec = np.arange(int(len(data) / self.sample_rate / self.frame_length_s)) * self.frame_length_s

        signal_energy = self.calculate_level(signal=data, channels=1)
        num_blocks = int(np.floor(data.shape[0] / self.frame_length))
        vad_vec = np.zeros(shape=(num_blocks, ))

        for block in range(num_blocks):
            idx_in = int(block * self.frame_length)
            idx_out = int(idx_in + self.frame_length)

            if self.calculate_level(signal=data[idx_in:idx_out], channels=1) >= self.threshold + signal_energy:
                vad_vec[block] = 1

        return timevec, vad_vec
