import numpy as np
import scipy.io as sio
import time
import torch
import math
import matplotlib.pyplot as plt
import pyroomacoustics as pra


class SRP_PHAT:

    def __init__(self, coordinates, parameters):

        self.sample_rate = parameters['sample_rate']
        self.frame_length = parameters['frame_length']
        self.num_classes = parameters['num_classes']
        self.base_dir = parameters['base_dir']
        self.percentile = parameters['mask_percentile']
        self.use_informed = parameters['use_informed']

        self.speed_of_sound = 344

        coordinates = coordinates
        self.num_channels = len(coordinates)

        theta = np.divide(range(self.num_classes), self.num_classes) * 2 * math.pi

        self.srp_phat = pra.doa.srp.SRP(L=np.transpose(coordinates), fs=self.sample_rate, nfft=self.frame_length, c=self.speed_of_sound, num_src=1,
                                        mode='far', r=None, azimuth=theta, colatitude=None, dim=2)

    @staticmethod
    def P2R(radii, angles):
        return radii * np.exp(1j * angles)

    @staticmethod
    def R2P(x):
        return np.abs(x), np.angle(x)

    def forward(self, data):
        if self.use_informed:
            return self.calculate_informed(signal=data)
        else:
            return self.calculate_uninformed(signal=data)

    def calculate_uninformed(self, signal):

        spec_frames = np.expand_dims(a=np.fft.rfft(signal[:self.num_channels, :], n=self.frame_length, axis=-1), axis=-1)

        self.srp_phat.locate_sources(X=spec_frames, freq_range=[1, self.sample_rate/2])
        return self.srp_phat.grid.values

    def calculate_informed(self, signal):

        spec_frames = np.expand_dims(a=np.fft.rfft(signal, n=self.frame_length, axis=-1), axis=-1)
        magnitudes = np.abs(spec_frames[self.num_channels, :, 0])

        threshold = np.quantile(magnitudes, self.percentile/100)
        mask = np.zeros(shape=(int(self.frame_length/2+1),))
        for freq in range(int(self.frame_length / 2 + 1)):
            if magnitudes[freq] >= threshold:
                mask[freq] = 1

        for channel in range(self.num_channels):
            spec_frames[channel, :, 0] *= np.transpose(mask)

        self.srp_phat.locate_sources(X=spec_frames[:self.num_channels, :], freq_range=[1, self.sample_rate/2])
        return self.srp_phat.grid.values
