import numpy as np
import torch
from CustomDatasetGADOAE import CustomDatasetGADOAE
from Feature_GADOAE import Feature_GADOAE
from Mask import Mask


class Dataset_Training_Multi(CustomDatasetGADOAE):

    def __init__(self, parameters, device):

        super().__init__(parameters, device)

    def __getitem__(self, index):

        m_out_feature = torch.tensor(float('nan'), device=self.device)
        primary_label = 0

        while torch.isnan(m_out_feature).any():

            x_complete, primary_label, coordinates, voice_activity, parameters, _ = self.generate_test_signal(training=True)

            # self.save_audio('testaudio.wav', x_complete, normalize=True)

            # mask = Mask(self.device, coordinates, self.sample_rate).mask_2d_tau(self.desired_width_samples)

            feature = Feature_GADOAE(speed_of_sound=self.nC, sample_rate=self.sample_rate,
                                     frame_length=self.frame_length, num_dimensions=self.dimensions_array,
                                     num_channels=self.num_channels, coordinates=coordinates,
                                     desired_width=self.desired_width_samples, device=self.device,
                                     tau_mask=None, zero_padding=False)

            # List all speech frames
            speech_frames = [i for i, e in enumerate(voice_activity) if e == 1]
            # Pick one speech frame
            random_frame = speech_frames[int(np.random.randint(low=0, high=int(len(speech_frames)), size=1))]

            idx_in = int(random_frame * self.frame_length)
            idx_out = idx_in + self.frame_length
            x_frame = x_complete[:, idx_in:idx_out]

            m_out_feature = feature.calculate_max(frames=x_frame)

        # plt.plot(torch.mean(x_complete, dim=0))
        # for idx, i in enumerate(voice_activity):
        #     if i:
        #         plt.plot([idx*self.frame_length, (idx+1)*self.frame_length],[1, 1], 'k')
        #     else:
        #         plt.plot([idx * self.frame_length, (idx + 1) * self.frame_length], [0, 0], 'k')
        # for fr in speech_frames:
        #     plt.plot([fr*self.frame_length, (fr+1)*self.frame_length], [0.9, 0.9], 'g')
        # plt.plot([frame*self.frame_length, (frame+1)*self.frame_length], [-1, -1], 'r')
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

        return m_out_feature, primary_label



