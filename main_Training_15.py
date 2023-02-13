import argparse
import os

import torch.nn
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from DNN_GADOAE_Multi import DNN_GADOAE_Multi
from Timer import Timer
from Dataset_Training_Multi import Dataset_Training_Multi
from Training import Training

writer = SummaryWriter("runs/gcc")

NUM_SAMPLES = 100000
EPOCHS = 500
NUM_CLASSES = 72
MAX_CHANNELS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.00001
NUM_WORKERS = 24

RATIO = 0.8
BASE_DIR_ML = os.getcwd()
SPEECH_DIR = "../LibriSpeech/train-clean-360"
NOISE_TABLE = "../NoiseLibrary/noise_table/noise_table.mat"
NOISE_SAMPLED_DIR = "../NoiseLibrary/noise_sampled_05"

PARAMETERS = {'base_dir': BASE_DIR_ML,
              'sample_dir': SPEECH_DIR,
              'noise_table': NOISE_TABLE,
              'noise_sampled_dir': NOISE_SAMPLED_DIR,
              'mic_array': None,
              'sample_rate': 8000,
              'signal_length': 1,
              'min_rt_60': 0.13,
              'max_rt_60': 1.0,
              'min_snr': 0,
              'max_snr': 30,
              'min_sir': None,
              'max_sir': None,
              'room_dim': [9, 5, 3],
              'room_dim_delta': [1.0, 1.0, 0.5],
              'mic_center': [4.5, 2.5, 1.0],
              'mic_center_delta': [0.5, 0.5, 0.5],
              'min_source_distance': 1.0, #1.0
              'max_source_distance': 3.0, #3.0
              'proportion_noise_input': 0.0, # 0.0: speech only, 1.0: noise only
              'noise_style': 'table', #'table', 'sampled', 'random'
              'vad_name': 'Energy', #Denk, Energy, MaKo, None, testNet
              'cut_silence': 'silence', #'none', 'beginning', 'silence'
              'frame_length': 256,
              'max_sensor_spread': 0.2, #lookup noise: only up to 0.2
              'min_array_width': 0.4,
              'rasterize_array': False,
              'use_tau_mask': True,
              'use_informed': False,
              'informed_style': 'ifft', # ifft, random
              'sensor_grid_digits': 2, #2: 0.01m
              'num_classes': 72,
              'num_samples': NUM_SAMPLES,
              'num_sources': 1,
              'max_uncertainty': 0.00,
              'dimensions_array': 2,
              'mask_percentile': None,
              'min_sensors': 15,
              'max_sensors': 15,
              'num_channels': MAX_CHANNELS,
              'augmentation_style': 'repeat_roll', # repeat_last, repeat_all, repeat_roll
              'leave_out_exact_values': False, #ATTENTION: During evaluation this MUST be set to False
              'use_in_between_doas': True}

is_training = True
is_continue = False


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':

    # torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('net', type=ascii)
    args = parser.parse_args()

    print(f'Net: {args.net[1:-1]}')

    # with Timer("CNN computation"):
    # if torch.cuda.is_available():
    #     device = 'cuda'
    # #     trained_net = "cnnfourth_GPU.pth"
    # else:
    device = "cpu"
    trained_net = f'{BASE_DIR_ML}/{args.net[1:-1]}'
    print(f"Using device '{device}'.")

    if is_training:

        dataset = Dataset_Training_Multi(parameters=PARAMETERS, device=device)

        # creating dnn and pushing it to CPU/GPU(s)
        dnn = DNN_GADOAE_Multi(num_channels=MAX_CHANNELS,
                               num_dimensions=PARAMETERS['dimensions_array'],
                               num_output_classes=NUM_CLASSES)

        if is_continue:
            sd = torch.load(trained_net)
            dnn.load_state_dict(sd)

        dnn.to(device)

        loss_fn = nn.CrossEntropyLoss()
        # loss_fn = CircularLoss(device, NUM_CLASSES)

        optimiser = torch.optim.Adam(dnn.parameters(), lr=LEARNING_RATE)

        Trainer = Training(model=dnn, loss_fn=loss_fn, optimiser=optimiser, dataset=dataset,
                               batch_size=BATCH_SIZE, ratio=RATIO, num_gpu=2, device=device,
                               filename=trained_net, num_workers=NUM_WORKERS)

        with Timer("Training online"):
            # train model
            Trainer.train(epochs=EPOCHS)

    writer.close()
    print("done.")


