import glob
import math

import numpy as np
import pyroomacoustics as pra
import torch
import torchaudio
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from NoiseTable import NoiseTable
from inv_sabine import inv_sabine
from NoiseSampled import NoiseSampled
from Librispeech import Librispeech
from VAD import VAD
from Mask import Mask
import scipy.io as sio


class CustomDataset(Dataset):

    def __init__(self, parameters, device):

        self.base_dir = parameters['base_dir']
        self.class_mapping = np.arange(0, parameters['num_classes'])
        self.num_classes = parameters['num_classes']
        self.sample_rate = parameters['sample_rate']
        self.num_sources = parameters['num_sources']
        self.device = device

        self.num_channels = 0
        self.max_sensor_spread = parameters['max_sensor_spread']
        self.min_array_width = parameters['min_array_width']
        self.frame_length = parameters['frame_length']
        self.len_s = parameters['signal_length']
        self.num_samples = parameters['num_samples']
        self.nC = 344.0
        self.rasterize = parameters['rasterize_array']
        self.sensor_grid_digits = parameters['sensor_grid_digits']
        self.use_tau_mask = parameters['use_tau_mask']

        self.leave_out_exact_values = parameters['leave_out_exact_values']
        self.use_in_between_doas = parameters['use_in_between_doas']

        # greatest distance between two microphones within the array
        self.max_dist_array = 2.0 * self.max_sensor_spread
        self.max_difference_samples = int(math.ceil(1.5 * self.max_dist_array / self.nC * self.sample_rate))

        # sample loading parameters
        self.sample_dir = parameters['sample_dir']
        self.proportion_noise = parameters['proportion_noise_input']
        self.threshold = -4.0
        self.hop_size = int(0.5 * self.frame_length)
        self.cut_style = parameters['cut_silence']
        self.librispeech = Librispeech(self.sample_dir, self.sample_rate, self.device)
        self.vad_name = parameters['vad_name']
        self.vad = VAD(name=self.vad_name, sample_rate=self.sample_rate, frame_length=self.frame_length/self.sample_rate, device=self.device)

        # Noise parameters
        self.min_snr = parameters['min_snr']
        self.max_snr = parameters['max_snr']
        self.noise_style = parameters['noise_style']
        self.noise_sampled_dir = parameters['noise_sampled_dir']
        if self.noise_style == 'table':
            self.noise_table = NoiseTable(parameters['noise_table'], self.sensor_grid_digits)
        elif self.noise_style == 'sampled':
            self.noise_sampled = NoiseSampled(directory=self.noise_sampled_dir, device=self.device)

        # Room acoustics parameters
        self.max_rt_60 = parameters['max_rt_60']
        self.min_rt_60 = parameters['min_rt_60']
        self.min_source_distance = parameters['min_source_distance']
        self.max_source_distance = parameters['max_source_distance']
        self.room_dim = parameters['room_dim']
        self.room_dim_delta = parameters['room_dim_delta']

        # Array parameters
        self.mic_center = np.array(parameters['mic_center'])
        self.mic_center_delta = np.array(parameters['mic_center_delta'])
        self.max_uncertainty = parameters['max_uncertainty']
        self.dimensions_array = parameters['dimensions_array']

        self.len_IR = 4096
        self.dist_samples = -255

        try:
            self.mic_coordinates_array = sio.loadmat(parameters['mic_array'])['coordinates']
            self.num_channels = len(self.mic_coordinates_array)

            # Adapt array dimensionality to number of dimensions used in the experiment
            self.mic_coordinates_array = self.reduce_array_dimensions(self.mic_coordinates_array, self.dimensions_array)

            if self.rasterize:
                for node, coords in enumerate(self.mic_coordinates_array):
                    self.mic_coordinates_array[node, :] = [round(i, self.sensor_grid_digits) for i in coords]

            self.max_dist_array = self.calculate_mic_distance(self.mic_coordinates_array[0, :],
                                                              self.mic_coordinates_array[-1, :])
            self.max_difference_samples = int(math.ceil(1.5 * self.max_dist_array / self.nC * self.sample_rate))

            mask = Mask(device=self.device, coordinates=self.mic_coordinates_array, sample_rate=self.sample_rate)
            self.tau_mask = mask.mask_2d_tau(max_difference_samples=self.max_difference_samples)
        except:
            pass

    def __len__(self):
        return self.num_samples

    def get_num_classes(self):
        return self.num_classes

    def get_coordinates(self):
        return self.mic_coordinates_array

    # @staticmethod
    # def calculate_level(signal, channels):
    #     if channels == 1:
    #         return torch.mul(10, torch.log10(torch.std(signal)))
    #     else:
    #         return torch.mul(10, torch.log10(torch.std(torch.mean(signal[:channels, :], dim=0))))

    @staticmethod
    def calculate_level(signal, channels):

        if channels == 1:
            return 10 * np.log10(np.std(signal) + 0.00001) #torch.finfo(torch.float32).eps)
        else:
            return 10 * np.log10(np.std(signal[:channels, :]))

    def normalise_level(self, signal, channels, use_vad=None):

        # control signal
        tmp_signal = signal.clone()

        # if multichannel, use average
        if channels > 1:
            tmp_signal = torch.mean(tmp_signal[:channels, :], dim=0)

        if use_vad:
            # use vad to exclude silent parts from level calculation
            tmp_signal = self.cut_signal(signal=tmp_signal, cut_style='silence')

        if channels == 1:
            return signal / torch.std(tmp_signal)
        else:
            signal[:channels, :] /= torch.std(tmp_signal)
            return signal

    def normalise_level_reference_channel(self, signal, reference_channel, use_vad=None):

        # control signal
        tmp_signal = signal[reference_channel, :]

        if use_vad:
            # use vad to exclude silent parts from level calculation
            tmp_signal = self.cut_signal(signal=tmp_signal, cut_style='silence')

        return signal / torch.std(tmp_signal)

    def change_level(self, signal, level, channels):
        factor = torch.pow(10, (level / 10)).to(self.device)
        if channels == 1:
            return signal * factor
        else:
            signal[:self.num_channels, :] *= factor
            return signal

    def set_level(self, signal, level, channels, use_vad=None):
        signal = self.normalise_level(signal=signal, channels=channels, use_vad=use_vad)
        return self.change_level(signal=signal, level=level, channels=channels)

    def generate_base_signal(self, training=False):

        # decision whether to use noise or speech
        if torch.rand(1) <= self.proportion_noise:
            base_signal = torch.randn(size=(int(self.len_s * self.sample_rate),), device=self.device)
            # shape: [N]
            signal_type = 'n'
        else:
            base_signal = self.get_signal(training=training)
            # shape: [N]
            signal_type = 's'
        return base_signal, signal_type

    def generate_random_noise(self, noise_level_desired, length):

        # Noise type: uncorrelated
        noise = torch.randn(size=(self.num_channels, length), device=self.device, dtype=torch.float32)
        noise = self.normalise_level(noise, channels=self.num_channels)
        factor = 10 ** (noise_level_desired / 10).type(torch.float32)

        return noise * factor

    def get_class_mapping(self):
        return self.class_mapping

    def _hann_poisson(self, m, alpha):
        mo2 = (m - 1) / 2
        n = torch.arange(-mo2, mo2 + 1, device=self.device)
        scl = alpha / mo2
        p = torch.exp(-scl * torch.abs(n))
        scl2 = torch.pi / mo2
        h = 0.5 * (1 + torch.cos(scl2 * n))
        return p * h

    def generate_room(self):
        # Randomised room dimensions
        room_dim_desired = self.room_dim + (2 * np.random.rand(3) - 1) * self.room_dim_delta

        # Randomised T60
        rt_60_desired = self.min_rt_60 + np.random.rand(1) * np.abs(self.max_rt_60 - self.min_rt_60)

        return room_dim_desired, rt_60_desired

    def generate_label(self):
        # Generate random label (aka source direction class)
        desired_label = np.random.randint(low=0, high=self.num_classes, size=1, dtype=np.longlong)

        return desired_label

    def generate_source(self, room_dim_desired, desired_label):

        # to test generalization of the dnn towards unseen DOA's
        if self.use_in_between_doas:
            desired_label = desired_label.astype(np.float) + 0.5

        # Calculate source direction from label
        source_direction = 2 * np.pi / self.num_classes * desired_label

        # Randomised source distance
        source_distance = self.min_source_distance + np.random.rand(1) * np.abs(
            self.max_source_distance - self.min_source_distance)

        # Calculate cartesian coordinates of source position relative to center of array
        source_position = [np.cos(source_direction[0]) * source_distance[0],
                           np.sin(source_direction[0]) * source_distance[0], 0.0]

        # Randomised center of array
        mic_center_desired = self.mic_center + (2 * np.random.rand(3) - 1) * self.mic_center_delta

        # Absolute source position
        source_position = source_position + mic_center_desired

        # Ensure that source is within boundaries of the room
        while (source_position[0] >= room_dim_desired[0] - 0.1) or (source_position[0] <= 0.1) or \
                (source_position[1] >= room_dim_desired[1] - 0.1) or (source_position[1] <= 0.1):
            source_distance -= 0.01

            # Calculate cartesian coordinates of source position relative to center of array
            source_position = [np.cos(source_direction[0]) * source_distance[0],
                               np.sin(source_direction[0]) * source_distance[0], 0.0]
            # Absolute source position
            source_position = source_position + mic_center_desired

        return source_position, mic_center_desired

    def add_signal_and_noise(self, signal, noise):

        return torch.add(signal, noise)

    def reduce_array_dimensions(self, mic_coordinates_array, dimensions_array):
        #
        # coordinates = torch.zeros(size=(len(mic_coordinates_array), 3), device=self.device)
        #
        # for idx, coord in enumerate(mic_coordinates_array):
        #     for dim in range(dimensions_array-1):
        #         coordinates[idx, dim] = coord[dim]

        # Flatten unnecessary dimensions
        if dimensions_array == 1:
            mic_coordinates_array[[i for i in range(self.num_channels)], 1:] = 0
            # coordinates[:][:2] = mic_coordinates_array[:][:2]
        elif dimensions_array == 2:
            # coordinates[:][:1] = mic_coordinates_array[:][:1]
            mic_coordinates_array[[i for i in range(self.num_channels)], 2] = 0
        elif dimensions_array == 3:
            pass

        # print(coordinates)
        # return coordinates
        return mic_coordinates_array

    def build_random_array(self, b_rescale):

        max_dist_sensors_from_center = np.random.rand(1) * self.max_sensor_spread

        # Build random 3D array of sensors
        self.mic_coordinates_array = 2 * (
                np.random.rand(self.num_channels, 3) - 0.5) * max_dist_sensors_from_center

        if b_rescale:
            # Rescale array to  width of self.min_array_width in x/y/z -> prevents arrays too small for accurate DOAE
            self.mic_coordinates_array = self.rescale(self.mic_coordinates_array)

    def generate_noise(self, shape):
        # Generate noise at specific level
        snr_desired = self.max_snr - torch.rand(1) * np.abs(self.max_snr - self.min_snr)
        # show that dnn generalizes towards unseen SNR's
        if self.leave_out_exact_values:
            while snr_desired == np.round(snr_desired):
                snr_desired = self.max_snr - torch.rand(1) * np.abs(self.max_snr - self.min_snr)

        if self.noise_style == 'table':
            # Fetch frozen noise
            noise = self.noise_table.lookup(self.mic_coordinates_array.copy(), self.len_s * self.sample_rate)
            noise = self.set_level(signal=noise, level=-snr_desired, channels=self.num_channels)

        elif self.noise_style == 'sampled':
            # Noise type: correlated
            noise = self.noise_sampled.get_noise(self.num_channels).to(self.device)
            noise = self.set_level(signal=noise, level=-snr_desired, channels=self.num_channels)

        else:
            noise = self.generate_random_noise(-snr_desired, shape[1])

        noise = self.cut_to_length(noise)

        return noise, snr_desired

    def cut_to_length(self, signal):

        if signal.dim() == 1:
            desired_len = int(self.sample_rate * self.len_s)
            if signal.shape[0] < desired_len:
                factor = int(np.ceil(desired_len / signal.shape[0]))
                signal = signal.repeat(factor)
                signal = signal[:desired_len]
            elif signal.shape[0] > desired_len:
                signal = signal[:desired_len]
        else:
            desired_len = int(self.sample_rate * self.len_s)
            if signal.shape[1] < desired_len:
                factor = int(np.ceil(desired_len / signal.shape[1]))
                signal = signal.repeat(1, factor)
                signal = signal[:, :desired_len]
            elif signal.shape[1] > desired_len:
                signal = signal[:, :desired_len]

        return signal

    def get_signal(self, training=False):
        # obtain random sample from librispeech corpus
        signal = self.librispeech.get_random_sample()

        # training excludes silence
        if training:
            # generate voice activity map
            _, vad_vector = self.get_vad(signal=signal)
            # cut sient passages
            signal = self.cut_silent_parts(signal, vad_vector)

        # Ensure that all signals have the same length (self.len_s) - includes truncation and repitition
        signal = self.cut_to_length(signal)

        return signal

    def cut_signal(self, signal, cut_style):

        _, vad_vector = self.get_vad(signal=signal)

        if cut_style == 'beginning':
            signal = self.cut_beginning(signal, vad_vector)
        elif cut_style == 'silence':
            signal = self.cut_silent_parts(signal, vad_vector)

        return signal

    def get_vad(self, signal):
        time_vector, vad_vector = self.vad.get_vad(data=signal)
        return time_vector, vad_vector

    def cut_beginning(self, signal, vad):
        start_at = 0
        for idx, item in enumerate(vad):
            if item == 1:
                start_at = idx
                break
        return signal[start_at * self.frame_length:]

    def cut_silent_parts(self, signal, vad):

        speech_frames = []
        for idx, frame in enumerate(vad):
            if frame == 1:
                speech_frames.append(idx)

        cut_signal = torch.zeros(size=(len(speech_frames) * self.frame_length,), device=self.device, dtype=torch.float)

        for idx, frame in enumerate(speech_frames):
            cut_signal[idx * self.frame_length:(idx+1) * self.frame_length] = \
                signal[frame * self.frame_length:(frame+1) * self.frame_length]

        return cut_signal

    def save_audio(self, filename, audio, normalize):
        if normalize:
            audio = audio / torch.max(torch.max(audio))
        torchaudio.save(filename, audio, self.sample_rate)

    def load_ir(self, label):
        file_list = glob.glob(f"{self.irs_dir}/{label.item():02d}/*.wav")
        rand_idx = torch.randint(low=0, high=len(file_list), size=(1,))
        # rand_idx = torch.tensor([0])
        ir, fs = torchaudio.load(file_list[rand_idx])
        # ir = F.resample(ir, fs, 8000)

        return ir

    # def generate_sample_GPU(self, room_dim, rt_60_desired, source_position, base_signal):
    #
    #     beta = gpuRIR.beta_SabineEstimation(room_sz=room_dim, T60=rt_60_desired)
    #     rirs = gpuRIR.simulateRIR(room_sz=room_dim,
    #                               beta=beta,
    #                               pos_src=source_position,
    #                               pos_rcv=(self.mic_coordinates + self.mic_center),
    #                               nb_img=(2, 2, 2),
    #                               Tmax=(self.len_IR / self.sample_rate),
    #                               fs=self.sample_rate)
    #
    #     x = np.zeros(shape=(self.num_channels + 1, base_signal.shape[1] + rirs.shape[2] - 1))
    #     for channel in range(self.num_channels):
    #         x[channel, :] = oaconvolve(rirs[0, channel, :], base_signal[0].cpu().detach().numpy(), mode='full')
    #
    #     # calculate time difference of source signal in samples
    #     dist_samples = int(self.calculate_mic_distance(self.mic_center, source_position)/self.nC*self.sample_rate)
    #     #  shift the base signal -> simulate wireless transmission
    #     base_signal = torch.roll(input=base_signal, shifts=dist_samples, dims=-1)
    #
    #     # last channel is clean signal
    #     x[self.num_channels, 0:len(base_signal[0])] = base_signal[0].cpu().detach().numpy()
    #
    #     x = torch.tensor(x, device=self.device, dtype=torch.float32)
    #
    #     return x

    def generate_sample_CPU(self, room_dim, rt_60_desired, source_position, base_signal, array, mic_center):

        e_absorption, max_order = inv_sabine(rt_60_desired, self.room_dim, self.nC)
        max_order = 2

        room = pra.ShoeBox(
            p=room_dim,
            fs=self.sample_rate,
            materials=pra.Material(energy_absorption=e_absorption[0]),
            max_order=max_order)
        room.add_source(source_position, signal=base_signal, delay=0)

        array += mic_center

        room.add_microphone_array(np.transpose(array))

        # print(label)
        #
        # plt.plot([0, room_dim[0], room_dim[0], 0, 0], [0, 0, room_dim[1], room_dim[1], 0], 'k')
        # plt.plot(source_position[0], source_position[1], 'rx')
        # for dim in array:
        #     plt.plot(dim[0], dim[1], 'bo')
        # plt.plot([0, 5*torch.cos(2*torch.pi*label/72)], [0, 5*torch.sin(2*torch.pi*label/72)], 'r')
        # plt.show()

        room.compute_rir()

        room.simulate()

        x = room.mic_array.signals
        x = torch.tensor(x, device=self.device, dtype=torch.float32)
        x = self.cut_to_length(x)

        return x

    @staticmethod
    def calculate_mic_distance(coord1, coord2):
        return math.sqrt((coord1[0] - coord2[0]) ** 2 +
                         (coord1[1] - coord2[1]) ** 2 +
                         (coord1[2] - coord2[2]) ** 2)

    @staticmethod
    def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

    def rescale(self, coordinates):

        # find largest distance in each dimension:
        for dim in range(self.dimensions_array):
            coordinates[:, dim] -= np.mean(coordinates[:, dim])
            distance = np.max(coordinates[:, dim]) - np.min(coordinates[:, dim])
            coordinates[:, dim] = coordinates[:, dim] / distance * self.min_array_width
            coordinates[:, dim] -= np.mean(coordinates[:, dim])

        return coordinates

    def __getitem__(self, index):

        pass


