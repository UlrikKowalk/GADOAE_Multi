from pathlib import Path
import torch
import torchaudio

class Librispeech:

    def __init__(self, directory, sample_rate, device):

        self.directory = directory
        self.sample_rate = sample_rate
        self.device = device
        self.file_list = []
        for path in Path(self.directory).rglob('*.flac'):
            self.file_list.append(path)
        self.length = len(self.file_list)

    def __len__(self):
        return self.length

    def get_random_sample(self):
        rand_sample = torch.randint(low=0, high=self.length, size=(1,))
        signal, fs = torchaudio.load(self.file_list[rand_sample])
        signal = torch.squeeze(signal)

        if fs != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=self.sample_rate, dtype=signal.dtype)
            return resampler(signal).to(self.device)
        else:
            return signal.to(self.device)
