import glob
import torch
import torchaudio
import numpy as np

class NoiseSampled:

    # Class that manages sampled noise for VIWER-S array with varying number of microphones (shape samples)

    def __init__(self, directory, device):

        self.directory = directory
        self.device = device

    def get_noise(self, num_channels):

        file_list = glob.glob(self.directory + "/*.wav")
        rand_sample = torch.randint(low=0, high=len(file_list), size=(1,))

        noise, fs = torchaudio.load(file_list[rand_sample])
        # signal = F.resample(signal, fs, 8000)
        shift = np.random.randint(low=0, high=noise.shape[1])
        noise = torch.roll(input=noise, shifts=shift, dims=-1)

        return noise



