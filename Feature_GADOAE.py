import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import scipy


class Feature_GADOAE:

    def __init__(self, speed_of_sound, sample_rate, frame_length, num_dimensions, num_channels, coordinates, desired_width, device, tau_mask, zero_padding=False):

        self.speed_of_sound = speed_of_sound
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.num_dimensions = num_dimensions
        self.num_channels = num_channels
        self.coordinates = coordinates
        self.desired_width = desired_width
        self.device = device
        self.tau_mask = tau_mask
        self.zero_padding = zero_padding
        if self.zero_padding:
            self.principal_length = 2 * self.frame_length
        else:
            self.principal_length = self.frame_length

        self.num_permutations = int(self.num_channels * (self.num_channels - 1) / 2)

        self.unity_magnitudes = torch.ones(size=(self.principal_length,), device=self.device)
        self.window = torch.hann_window(self.frame_length, device=self.device).repeat((self.num_channels, 1))

    def allocate_max(self, num_frames):
        return torch.zeros(size=(num_frames, self.num_permutations +
                                 self.num_dimensions * self.num_channels,),
                                 device=self.device)

    def allocate_full(self, num_frames):
        return torch.zeros(size=(num_frames, self.num_permutations * 2 * self.desired_width +
                                 self.num_dimensions * self.num_channels,),
                                 device=self.device)

    @staticmethod
    def find_center(self, array):
        # find the geometrical centre of the array
        return [np.mean(np.unique(array, axis=0), axis=0)]

    def calculate_lags(self, array):
        distance = self.calculate_distances_from_center(array)
        lags = distance / self.speed_of_sound * self.sample_rate
        return lags

    def calculate_distances_from_center(self, array):
        center = self.find_center(array)
        distance = (array - center)
        return np.sqrt(distance[:, 0] ** 2 + distance[:, 1] ** 2 + distance[:, 2] ** 2)

    @staticmethod
    def calculate_distance(self, array, s1, s2):
        vec = array[s1] - array[s2]
        return np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

    def calculate_max(self, frames):

        final_feature = torch.zeros(size=(self.num_permutations +
                                          self.num_dimensions * self.num_channels,), device=self.device)

        frames = torch.mul(frames, self.window)

        if self.zero_padding:
            frames = F.pad(frames, pad=(0, self.frame_length))

        x_spec = torch.fft.fft(frames, dim=-1)
        x_angle = x_spec.angle()

        slice_idx = 0

        # now with magnitude weighting by direct signal
        for ii in range(self.num_channels):
            for jj in range(ii+1, self.num_channels):
                # Subtract in order to account for conjugation
                gcc_phase = x_angle[ii, :] - x_angle[jj, :]
                m_cross_phase_weighted = torch.polar(self.unity_magnitudes, gcc_phase)
                m_gcc_phat_weighted = torch.fft.ifft(m_cross_phase_weighted, dim=-1).real
                m_gcc_phat_weighted = torch.fft.fftshift(m_gcc_phat_weighted, dim=-1)
                m_gcc_phat_masked = m_gcc_phat_weighted[int(self.principal_length / 2 - self.desired_width):
                                                        int(self.principal_length / 2 + self.desired_width)] #\
                                # * self.tau_mask[ii, jj, :]

                # Compute argmax -> only forward the position of gcc-phat maximum
                # use argmax if argmax is not 0 -> 0 is result if no max can be found -> need to approve this!
                tmp_max = torch.argmax(m_gcc_phat_masked)

                # parabolic interpolation -> see if necessary
                if tmp_max > 0 and tmp_max < len(m_gcc_phat_masked)-1:
                    pos = self.quadratic_interpolation(m_gcc_phat_masked[tmp_max-1], m_gcc_phat_masked[tmp_max], m_gcc_phat_masked[tmp_max+1])
                    tmp_max = float(tmp_max) + pos

                final_feature[slice_idx] = (tmp_max - self.desired_width) if tmp_max != 0 else 0
                # final_feature[slice_idx + self.num_permutations] = self.calculate_distance(self.coordinates, ii, jj)

                # if slice_idx >= 100 and slice_idx < 110:
                #     print(final_feature[slice_idx])
                #     plt.plot(m_gcc_phat_masked)
                #     plt.show()
                slice_idx += 1

        # append array geometry to feature vector (depending on how many dimensions we train)
        # serially [x0, x1, ..., y0, y1, ...]
        for dim in range(self.num_dimensions):
            final_feature[slice_idx:slice_idx + self.num_channels] = self.coordinates[:, dim] * 40
            slice_idx += self.num_channels

        final_feature = torch.unsqueeze(final_feature, 0)

        return final_feature

    def calculate_full(self, frames):

        final_feature = torch.zeros(size=(self.num_permutations * 2 * self.desired_width +
                                          self.num_dimensions * self.num_channels,), device=self.device)

        frames = torch.mul(frames, self.window)

        if self.zero_padding:
            frames = F.pad(frames, pad=(0, self.frame_length))

        x_spec = torch.fft.fft(frames, dim=-1)
        x_angle = x_spec.angle()

        slice_idx = 0

        # now with magnitude weighting by direct signal
        for ii in range(self.num_channels):
            for jj in range(ii+1, self.num_channels):
                # Subtract in order to account for conjugation
                gcc_phase = x_angle[ii, :] - x_angle[jj, :]
                m_cross_phase_weighted = torch.polar(self.unity_magnitudes, gcc_phase)
                m_gcc_phat_weighted = torch.fft.ifft(m_cross_phase_weighted, dim=-1).real
                m_gcc_phat_weighted = torch.fft.fftshift(m_gcc_phat_weighted, dim=-1)
                final_feature[slice_idx:slice_idx+2*self.desired_width] = m_gcc_phat_weighted[int(self.principal_length / 2 - self.desired_width):
                                                        int(self.principal_length / 2 + self.desired_width)] #\
                                # * self.tau_mask[ii, jj, :]

                # final_feature[slice_idx + self.num_permutations] = self.calculate_distance(self.coordinates, ii, jj)

                # if slice_idx >= 100 and slice_idx < 110:
                #     print(final_feature[slice_idx])
                #     plt.plot(m_gcc_phat_masked)
                #     plt.show()
                slice_idx += 2 * self.desired_width

        # append array geometry to feature vector (depending on how many dimensions we train)
        # serially [x0, x1, ..., y0, y1, ...]
        for dim in range(self.num_dimensions):
            final_feature[slice_idx:slice_idx + self.num_channels] = torch.tensor(self.coordinates[:, dim] * 10)
            slice_idx += self.num_channels

        # plt.plot(final_feature)
        # plt.show()
        # print(final_feature)
        # print(self.coordinates)

        final_feature = torch.unsqueeze(final_feature, 0)

        return final_feature

    @staticmethod
    def matmul_complex(t1, t2):
        return torch.view_as_complex(
            torch.stack((t1.real @ t2.real - t1.imag @ t2.imag, t1.real @ t2.imag + t1.imag @ t2.real), dim=2))

    @staticmethod
    def quadratic_interpolation(mag_prev, mag, mag_next):
        return (mag_next - mag_prev) / (2.0 * (2.0 * mag - mag_next - mag_prev))