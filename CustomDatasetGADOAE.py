import math
import random

# import gpuRIR
import numpy as np
import pyroomacoustics as pra
import scipy
import torch
from matplotlib import pyplot as plt
from scipy.signal import oaconvolve

from Coordinates import Coordinates
from CustomDataset import CustomDataset
from inv_sabine import inv_sabine


class CustomDatasetGADOAE(CustomDataset):

    def __init__(self, parameters, device):

        super().__init__(parameters, device)

        self.min_sensors = parameters['min_sensors']
        self.max_sensors = parameters['max_sensors']
        self.num_channels = parameters['num_channels']
        self.augmentation_style = parameters['augmentation_style']

        # greatest distance between two microphones within the array
        self.desired_width_samples = int(np.ceil(2.0 * self.max_sensor_spread / self.nC * self.sample_rate))
        self.max_dist_sensors_from_center = None

    def generate_sample_CPU(self, room_dim, rt_60_desired, source_position, base_signal, array, mic_center):

        e_absorption, max_order = inv_sabine(rt_60_desired, self.room_dim, self.nC)
        max_order = 2

        room = pra.ShoeBox(
            p=room_dim,
            fs=self.sample_rate,
            materials=pra.Material(energy_absorption=e_absorption[0]),
            max_order=max_order)
        room.add_source(source_position, signal=base_signal.cpu(), delay=0)

        # now 'denormalize' to be centered around desired array center
        array += mic_center

        room.add_microphone_array(np.transpose(array))

        # print(f'label= {label}, degrees: {label/72*360}')
        #
        # fig, ax = plt.subplots(1)
        # plt.plot([0, room_dim[0], room_dim[0], 0, 0], [0, 0, room_dim[1], room_dim[1], 0], 'k')
        # plt.plot(source_position[0], source_position[1], 'rx')
        # for dim in array_plus_direct:
        #     plt.plot(dim[0], dim[1], 'bo')
        # plt.plot([mic_center[0], 5*torch.cos(2*torch.pi*label/72)+mic_center[0]], [mic_center[1], 5*torch.sin(2*torch.pi*label/72)+mic_center[1]], 'r')
        # ax.axis('equal')
        # plt.show()

        room.compute_rir()
        room.simulate()
        x = room.mic_array.signals
        x = torch.tensor(x, device=self.device, dtype=torch.float32)
        x = self.cut_to_length(x)

        return x

    # def generate_sample_GPU(self, room_dim, rt_60_desired, source_position, base_signal, array, mic_center):
    #
    #     beta = gpuRIR.beta_SabineEstimation(room_sz=room_dim, T60=rt_60_desired)
    #
    #     # array centered around origin
    #     array_plus_direct = np.append(self.mic_coordinates_array, np.expand_dims(self.mic_coordinates_direct, axis=0),
    #                                   axis=0)
    #
    #     # now 'denormalize' to be centered around desired array center
    #     array_plus_direct += mic_center
    #
    #     rirs = gpuRIR.simulateRIR(room_sz=room_dim,
    #                               beta=beta,
    #                               pos_src=source_position,
    #                               pos_rcv=array_plus_direct,
    #                               nb_img=(2, 2, 2),
    #                               Tmax=(self.len_IR / self.sample_rate),
    #                               fs=self.sample_rate)
    #
    #     # x = gpuRIR.simulateTrajectory(source_signal=base_signal, RIRs=rirs)
    #
    #     x = np.zeros(shape=(self.num_channels + 1, base_signal.shape[0] + rirs.shape[2] - 1))
    #     # x = torch.zeros(size=(self.num_channels + 1, base_signal.shape[1] + rirs.shape[2] - 1), device='cuda')
    #
    #     for channel in range(self.num_channels):
    #         x[channel, :] = oaconvolve(rirs[0, channel, :], base_signal.cpu().detach().numpy(), mode='full')
    #
    #
    #     # estimate source distance from signals
    #     self.dist_samples = self.estimate_source_distance(x)
    #
    #     x = torch.from_numpy(x).float().to('cuda')
    #
    #     # last channel is clean signal
    #     x[self.num_channels, 0:len(base_signal)] = base_signal
    #
    #     #  shift the 'direct' signal -> simulate acoustic transmission
    #     x[self.num_channels, :] = torch.roll(input=x[self.num_channels, :], shifts=self.dist_samples, dims=-1)
    #     x = x[:, :len(base_signal)]
    #
    #     return x

    def add_signal_and_noise(self, signal, noise):

        signal[:self.num_channels, :] += noise

        return signal

    def get_desired_sir(self):
        return self.min_sir + torch.rand(1, device=self.device) * torch.abs(self.max_sir - self.min_sir)

    @staticmethod
    def augment_channels_repitition_last(coordinates, num_channels_desired, num_channels):
        # repeat the last channel (simulate multiple sensors at same position)
        for sensor in range(num_channels_desired - num_channels):
            coordinates = np.vstack([coordinates, coordinates[-1, :]])
        return coordinates

    @staticmethod
    def augment_channels_repetition_all(coordinates, num_channels_desired, num_channels):
        # repeat the whole array until num_channels is reached
        temp = np.zeros(shape=(num_channels_desired, 3))
        temp[:num_channels, :] = coordinates
        for sensor in range(num_channels, num_channels_desired):
            temp[sensor, :] = coordinates[sensor % num_channels, :]
        return temp

    @staticmethod
    def augment_channels_repetition_random(coordinates, num_channels_desired, num_channels):
        temp_coordinates = np.zeros(shape=(num_channels_desired, 3))
        temp_coordinates[:num_channels, :] = coordinates
        num_coordinates = num_channels
        # should another 'block' be attached?
        while num_coordinates < num_channels_desired:
            roll_index = np.random.randint(low=0, high=num_channels)
            temp = np.roll(a=coordinates, shift=roll_index, axis=0)

            temp_index = 0
            # fill the temporary array as long as size is not exceeded
            while(temp_index < num_channels and num_coordinates < num_channels_desired):
                temp_coordinates[num_coordinates, :] = temp[temp_index, :]

                temp_index += 1
                num_coordinates += 1

        return temp_coordinates

    def generate_test_signal(self, training=False):

        #############################################  Establish Room  #################################################

        # First establish room dimensions and array position
        room_dim_desired, rt_60_desired = self.generate_room()

        #########################################  Establish Primary Source  ###########################################

        # Randomly create a label (select one class)
        primary_label = self.generate_label()

        # Randomly create a source position and center of the microphone array
        source_position, mic_center_desired = self.generate_source(room_dim_desired, primary_label)

        # Generate a base signal (noise/speech)
        base_signal, signal_type = self.generate_base_signal(training=training)

        # Normalise signal (aka 0dB)
        base_signal_desired = self.normalise_level(signal=base_signal, channels=1)

        # Cut silent parts / beginnings / ... - based on desired way of cutting and vad
        _, voice_activity = self.get_vad(signal=base_signal_desired)

        # Build randomized array
        num_sensors = int(self.min_sensors + np.random.rand(1) * np.abs(self.max_sensors - self.min_sensors))
        self.max_dist_sensors_from_center = np.random.rand(1) * self.max_sensor_spread

        # Build random 3D array of sensors
        self.mic_coordinates_array = 2 * (np.random.rand(num_sensors, 3) - 0.5) * self.max_dist_sensors_from_center

        # Flatten unnecessary dimensions
        if self.dimensions_array == 1:
            self.mic_coordinates_array[[i for i in range(num_sensors)], 1:] = 0
        elif self.dimensions_array == 2:
            self.mic_coordinates_array[[i for i in range(num_sensors)], 2] = 0
        elif self.dimensions_array == 3:
            pass

        if self.augmentation_style == 'repeat_last':
            # Data augmentation: repeat the last channel (simulate multiple sensors at same position)
            self.mic_coordinates_array = self.augment_channels_repitition_last(
                coordinates=self.mic_coordinates_array,
                num_channels_desired=self.num_channels,
                num_channels=num_sensors)
        elif self.augmentation_style == 'repeat_all':
            # Data augmentation: repeat the whole array until num_channels is reached
            self.mic_coordinates_array = self.augment_channels_repetition_all(
                coordinates=self.mic_coordinates_array,
                num_channels_desired=self.num_channels,
                num_channels=num_sensors)
        elif self.augmentation_style == 'repeat_roll':
            # Data augmentation: fill array with randomly permuted coordinates
            self.mic_coordinates_array = self.augment_channels_repetition_random(
                coordinates=self.mic_coordinates_array,
                num_channels_desired=self.num_channels,
                num_channels=num_sensors)
        else:
            raise Exception(f'Unknown argument given for <augmentation_style>: {self.augmentation_style}')

        # Coordinates of array geometry but possibly with deviation
        coordinates = Coordinates(self.device, self.num_channels, self.dimensions_array,
                                  self.max_uncertainty).generate(torch.from_numpy(self.mic_coordinates_array).to(self.device))

        # Signals are generated according to deviant geometry
        # if self.device == 'cpu':
        x = self.generate_sample_CPU(room_dim=room_dim_desired, rt_60_desired=rt_60_desired,
                                     source_position=source_position, base_signal=base_signal_desired,
                                     array=coordinates, mic_center=mic_center_desired)
        # else:
        #     x = self.generate_sample_GPU(room_dim=room_dim_desired, rt_60_desired=rt_60_desired,
        #                                  source_position=source_position, base_signal=base_signal_desired,
        #                                  array=coordinates, mic_center=mic_center_desired)

        # normalise level of primary source measured at microphone array
        x_primary = self.normalise_level_reference_channel(signal=x,
                                                           reference_channel=int((self.num_channels-1)/2),
                                                           use_vad=True)

        #######################################  Establish Secondary Sources  ##########################################

        x_secondary = torch.zeros(size=(self.num_channels, self.len_s * self.sample_rate), device=self.device)

        for source in range(self.num_sources - 1):
            # Randomly create a label (select one class)
            secondary_label = self.generate_label()

            # Randomly create a source position and center of the microphone array
            source_position, _ = self.generate_source(room_dim_desired, secondary_label)

            # Generate a base signal (noise/speech)
            base_signal, signal_type = self.generate_base_signal(training=training)

            # Normalise signal (aka 0dB)
            base_signal = self.normalise_level(signal=base_signal, channels=1)

            # Signals are generated according to deviant geometry
            # if self.device == 'cpu':
            x = self.generate_sample_CPU(room_dim=room_dim_desired, rt_60_desired=rt_60_desired,
                                         source_position=source_position, base_signal=base_signal,
                                         array=coordinates.clone(), mic_center=mic_center_desired)
            # else:
            #     x = self.generate_sample_GPU(room_dim=room_dim_desired, rt_60_desired=rt_60_desired,
            #                                  source_position=source_position, base_signal=base_signal,
            #                                  array=coordinates.clone(), mic_center=mic_center_desired)

            # accumulate secondary sources
            x_secondary += x

        # only normalise if secondary source exists -> otherwise chaos
        if self.num_sources > 1:
            desired_sir = self.get_desired_sir()
            # secondary sources generate common sir (0 means, all secondary sources combined are as loud as primary
            # source) - measured at microphone array
            x_secondary = self.normalise_level_reference_channel(signal=x_secondary,
                                                                 reference_channel=int((self.num_channels-1)/2),
                                                                 use_vad=True)
            x_secondary *= torch.pow(10, -desired_sir / 10)

        #############################################  Overlay Sources  ################################################

        # overlay primary source and ALL secondary sources
        x = x_primary + x_secondary

        #############################################  Establish Noise  ################################################

        # Generate a noise signal (uncorrelated or frozen, based on main parameters) with internally specified level
        noise, snr_desired = self.generate_noise(x.shape)

        # Mix signal and noise to satisfy snr criterion (no noise in direct signal of course)
        x_complete = self.add_signal_and_noise(x, noise)

        # fig, ax = plt.subplots(2)
        # ax[0].plot(x_complete[-1, :])
        # ax[1].plot(voice_activity)
        # plt.show()

        parameters = {'snr': snr_desired,
                      'rt_60': rt_60_desired,
                      'signal_type': signal_type}

        return x_complete, primary_label, coordinates, voice_activity, parameters, base_signal_desired
