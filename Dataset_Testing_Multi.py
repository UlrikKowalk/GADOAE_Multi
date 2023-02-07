import numpy as np
import torch
from matplotlib import pyplot as plt

from CustomDatasetGADOAE import CustomDatasetGADOAE
from Feature_GADOAE import Feature_GADOAE
from Mask import Mask


class Dataset_Testing_Multi(CustomDatasetGADOAE):

    def __init__(self, parameters, device):

        super().__init__(parameters, device)

    def __getitem__(self, index):

        m_out_feature = torch.tensor(float('nan'), device=self.device)
        primary_label = 0
        coordinates = None
        parameters = None
        x_complete = None
        voice_activity = None

        while torch.isnan(m_out_feature).any():

            x_complete, primary_label, coordinates, voice_activity, parameters, _ = self.generate_test_signal(training=True)

            # self.save_audio('testaudio.wav', x_complete, normalize=True)


            # mask = Mask(self.device, coordinates, self.sample_rate).mask_2d_tau(self.desired_width_samples)

            feature = Feature_GADOAE(speed_of_sound=self.nC, sample_rate=self.sample_rate,
                                     frame_length=self.frame_length, num_dimensions=self.dimensions_array,
                                     num_channels=self.num_channels, coordinates=coordinates,
                                     desired_width=self.desired_width_samples, device=self.device,
                                     tau_mask=None, zero_padding=False)

            # Now go frame based
            num_frames = int(x_complete.shape[1] / self.frame_length)

            # Pre-allocate output feature
            m_out_feature = feature.allocate_max(num_frames)

            # self.save_audio(filename=f'{self.base_dir}/testaudio.wav', audio=x, normalize=True)

            for frame in range(num_frames):
                idx_in = int(frame * self.frame_length)
                idx_out = idx_in + self.frame_length
                x_frame = x_complete[:, idx_in:idx_out]

                m_out_feature[frame, :] = feature.calculate_max(frames=x_frame)


        # plt.imshow(m_out_feature)
        # plt.show()

        # rows = 1
        # cols = 5
        # axes = []
        # fig2 = plt.figure()
        # for a in range(rows * cols):
        #     axes.append(fig2.add_subplot(rows, cols, a + 1))
        #     plt.imshow(m_out_feature[a, :, :])
        #     plt.axis('off')
        # fig2.tight_layout()
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=0.5, wspace=0.05, hspace=0)
        # plt.show()

        return m_out_feature, primary_label, coordinates, parameters, x_complete, voice_activity



