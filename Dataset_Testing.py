import torch
from matplotlib import pyplot as plt

from CustomDatasetInformed import CustomDatasetInformed
from Feature_GCC import Feature_GCC


class Dataset_Testing_CNN_GCC(CustomDatasetInformed):

    def __init__(self, parameters, device):

        super().__init__(parameters, device)

        # self.tau_mask = self.tau_mask.to(device)

    def __getitem__(self, index):

        m_out_feature = torch.tensor(float('nan'), device=self.device)
        primary_label = 0
        coordinates = 0
        voice_activity = 0
        parameters = 0
        x_complete = 0

        while torch.isnan(m_out_feature).any():

            x_complete, primary_label, coordinates, voice_activity, parameters, desired = self.generate_test_signal()

        # plt.plot(self.mic_coordinates_array[:, 0]+self.mic_center[0], self.mic_coordinates_array[:, 1]+self.mic_center[1], 'ro')
        # plt.plot(coordinates[:, 0]+self.mic_center[0], coordinates[:, 1]+self.mic_center[1], 'bx')
        # plt.plot(source_position[0], source_position[1], 'gd')
        # plt.xlim([0, 9])
        # plt.ylim([0, 5])
        # plt.show()

            feature = Feature_GCC(frame_length=self.frame_length, num_channels=self.num_channels,
                                  max_difference_samples=self.max_difference_samples, device=self.device,
                                  percentile=self.mask_percentile, tau_mask=self.tau_mask.to(self.device), zero_padding=False)

            # Now go frame based
            num_frames = int(x_complete.shape[1] / self.frame_length)

            # Pre-allocate output feature
            m_out_feature = feature.allocate(num_frames)

            # self.save_audio(filename=f'{self.base_dir}/testaudio.wav', audio=x, normalize=True)

            for frame in range(num_frames):
                idx_in = int(frame * self.frame_length)
                idx_out = idx_in + self.frame_length

                m_out_feature[frame, :] = feature.calculate(data=x_complete[:, idx_in:idx_out],
                                                            use_informed=self.use_informed,
                                                            informed_style=self.informed_style,
                                                            use_tau_mask=self.use_tau_mask)

        # rows = 1
        # cols = 5
        # axes = []
        # fig2 = plt.figure()
        # for a in range(rows * cols):
        #     axes.append(fig2.add_subplot(rows, cols, a + 1))
        #     plt.imshow(m_out_feature[frame, a, :, :].cpu().detach().numpy())
        #     plt.axis('off')
        # fig2.tight_layout()
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=0.5, wspace=0.05, hspace=0)
        # plt.show()

        return m_out_feature, primary_label, coordinates, parameters, x_complete, voice_activity, desired



