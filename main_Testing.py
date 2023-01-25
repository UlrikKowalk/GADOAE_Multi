import argparse
import os
from termcolor import colored

import numpy as np
import pandas as pd
import torch.nn
import torchaudio
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import sys

import Evaluation
from CNN_GCC_05 import CNN_GCC_05
from Dataset_Testing_CNN_GCC import Dataset_Testing_CNN_GCC
from MUSIC import MUSIC
from SRP_PHAT import SRP_PHAT
from Timer import Timer

NUM_SAMPLES = 5000
BATCH_SIZE = 1
MAX_THETA = 360.0
NUM_CLASSES = 72
NUM_WORKERS = 1

LIST_SNR = [10]
LIST_T60 = [0.50]
LIST_SIR = [0]
LIST_UNCERTAINTY = [0.00]
LIST_PERCENTILE = [50]

BASE_DIR_ML = os.getcwd()
SPEECH_DIR = "../LibriSpeech/test-clean"
NOISE_TABLE = "../NoiseLibrary/noise_table/noise_table.mat"
NOISE_SAMPLED_DIR = "../NoiseLibrary/noise_sampled_05"
MIC_ARRAY = './array_viwers_05.mat'

PARAMETERS = {'base_dir': BASE_DIR_ML,
              'sample_dir': SPEECH_DIR,
              'noise_table': NOISE_TABLE,
              'noise_sampled_dir': NOISE_SAMPLED_DIR,
              'mic_array': MIC_ARRAY,
              'sample_rate': 8000,
              'signal_length': 5,
              'min_rt_60': None,
              'max_rt_60': None,
              'min_snr': None,
              'max_snr': None,
              'min_sir': None,
              'max_sir': None,
              'room_dim': [9, 5, 3],
              'room_dim_delta': [1.0, 1.0, 0.5],
              'mic_center': [4.5, 2.5, 1],
              'mic_center_delta': [0.5, 0.5, 0.5],
              'min_source_distance': 1.0, #1.0
              'max_source_distance': 3.0, #3.0
              'proportion_noise_input': 0.0, # 0.0: speech only, 1.0: noise only
              'noise_style': 'sampled', #'table', 'sampled', 'random'
              'vad_name': 'Energy', #Denk, Energy, MaKo, None, testNet
              'cut_silence': 'none', #'none', 'beginning', 'silence'
              'frame_length': 256,
              'max_sensor_spread': 0.2, # noise table: only up to 0.2
              'min_array_width': 0.4,
              'rasterize_array': False,
              'use_tau_mask': True,
              'use_informed': True,
              'informed_style': 'ifft', # ifft, random
              'sensor_grid_digits': 2, #2: 0.01m
              'num_classes': 72,
              'num_samples': NUM_SAMPLES,
              'num_sources': 2,
              'max_uncertainty': None,
              'dimensions_array': 2,
              'mask_percentile': None}


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('net', type=ascii)
    args = parser.parse_args()

    # print(f'Net: {args.net[1:-1]}')

    device = "cpu"
    if torch.cuda.is_available():
        device_inference = 'cuda'
    else:
        device_inference = device

    trained_net = f'{BASE_DIR_ML}/{args.net[1:-1]}'
    # print(f"Using device '{device}'.")

    for SNR in LIST_SNR:
        for T60 in LIST_T60:
            for SIR in LIST_SIR:
                for UNCERTAINTY in LIST_UNCERTAINTY:
                    for PERCENTILE in LIST_PERCENTILE:

                        PARAMETERS['min_snr'] = SNR
                        PARAMETERS['max_snr'] = SNR
                        PARAMETERS['min_rt_60'] = T60
                        PARAMETERS['max_rt_60'] = T60
                        PARAMETERS['min_sir'] = SIR
                        PARAMETERS['max_sir'] = SIR
                        PARAMETERS['max_uncertainty'] = UNCERTAINTY
                        PARAMETERS['mask_percentile'] = PERCENTILE

                        file_name = Evaluation.get_filename(trained_net, SNR, T60, SIR, UNCERTAINTY, PARAMETERS)
                        os.system('color')
                        print(colored(f'Testing: {file_name}', 'grey'))
                        dataset = Dataset_Testing_CNN_GCC(parameters=PARAMETERS, device=device)
                        # creating dnn and pushing it to CPU/GPU(s)
                        dnn = CNN_GCC_05(num_output_classes=dataset.get_num_classes())

                        map_location = torch.device(device_inference)
                        sd = torch.load(trained_net, map_location=map_location)

                        dnn.load_state_dict(sd)
                        dnn.to(device_inference)

                        class_mapping = dataset.get_class_mapping()
                        num_classes = dataset.get_num_classes()

                        n_true = 0
                        n_false = 0
                        list_rt_60_testing = []
                        list_snr_testing = []
                        list_signal_type_testing = []
                        list_ir_type_testing = []

                        # print('Testing')

                        list_predictions = []
                        list_predictions_srpphat = []
                        list_predictions_music = []
                        list_targets = []
                        list_var = []
                        list_kalman = []
                        list_error = []
                        list_error_srpphat = []
                        list_error_music = []

                        num_zeros = int(np.ceil(np.log10(NUM_SAMPLES)) + 1)

                        n_total = len(dataset)
                        n_test = int(n_total * 1)
                        n_val = n_total - n_test

                        test_data, val_data = torch.utils.data.random_split(dataset=dataset, lengths=[n_test, n_val])

                        test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                      persistent_workers=True, shuffle=True)

                        idx = 0

                        list_weird = []
                        list_occ = [0] * NUM_CLASSES

                        #with Timer('test_signals'):

                        for features, target, coordinates, parameters, signal, voice_activity, desired in test_data_loader:

                            # make signal 2D again (reverse dataloader transformation)
                            signal = signal[0, :, :]
                            voice_activity = voice_activity[0]
                            desired = desired[0]

                            predicted_music = 0
                            predicted_srpphat = 0

                            # dataset.save_audio(filename=f'testaudio.wav', audio=signal, normalize=True)

                            # load features to inference device (cpu/cuda)
                            features = features.to(device_inference)
                            # initialize SRP-PHAT
                            srp_phat = SRP_PHAT(coordinates=dataset.get_coordinates(),
                                                parameters=PARAMETERS)
                            # initialize MUSIC
                            music = MUSIC(coordinates=dataset.get_coordinates(),
                                          parameters=PARAMETERS)

                            predicted, predictions_cnn = Evaluation.estimate_cnn(model=dnn,
                                                                sample=features.squeeze(dim=0),
                                                                voice_activity=voice_activity)

                            # predicted_srpphat, predictions_srpphat = Evaluation.estimate_srpphat(model=srp_phat,
                            #                                                    sample=signal,
                            #                                                    voice_activity=voice_activity)
                            #
                            # predicted_music, predictions_music = Evaluation.estimate_music(model=music,
                            #                                               sample=signal,
                            #                                               voice_activity=voice_activity)

                            # predicted, predictions_cnn = Evaluation.estimate_cnn_with_interpolation(model=dnn,
                            #                                     sample=features.squeeze(dim=0),
                            #                                     MAX_THETA=MAX_THETA,
                            #                                     NUM_CLASSES=NUM_CLASSES,
                            #                                     voice_activity=voice_activity)
                            #
                            # predicted_srpphat, predictions_srpphat = Evaluation.estimate_srpphat_with_interpolation(model=srp_phat,
                            #                                                    sample=signal,
                            #                                               MAX_THETA=MAX_THETA,
                            #                                               NUM_CLASSES=NUM_CLASSES,
                            #                                                    voice_activity=voice_activity)
                            #
                            # predicted_music, predictions_music = Evaluation.estimate_music_with_interpolation(model=music,
                            #                                               sample=signal,
                            #                                               MAX_THETA=MAX_THETA,
                            #                                               NUM_CLASSES=NUM_CLASSES,
                            #                                               voice_activity=voice_activity)





                            expected = int(target)

                            list_occ[expected] += 1

############################################################

                            # fig, ax = plt.subplots(3)
                            # ax[0].fill_between(
                            #     x=range(len(voice_activity)),
                            #     y1=voice_activity * 72,
                            #     where=(voice_activity > 0),
                            #     color="r",
                            #     alpha=1.0)
                            # ax[0].plot(predictions_cnn, 'b', label='CNN')
                            # ax[0].legend(loc='upper right')
                            # ax[1].fill_between(
                            #     x=range(len(voice_activity)),
                            #     y1=voice_activity * 72,
                            #     where=(voice_activity > 0),
                            #     color="r",
                            #     alpha=1.0)
                            # ax[1].plot(predictions_srpphat, 'b', label='SRP-PHAT')
                            # ax[1].legend(loc='upper right')
                            # ax[2].fill_between(
                            #     x=range(len(voice_activity)),
                            #     y1=voice_activity * 72,
                            #     where=(voice_activity > 0),
                            #     color="r",
                            #     alpha=1.0)
                            # ax[2].plot(predictions_music, 'b', label='MUSIC')
                            # ax[2].legend(loc='upper right')
                            #
                            # plt.show()

############################################################

                            list_predictions.append(predicted)
                            list_predictions_srpphat.append(predicted_srpphat)
                            list_predictions_music.append(predicted_music)
                            list_targets.append(expected)
                            # list_var.append(variance)
                            # list_kalman.append(kalman)
                            list_rt_60_testing.append(parameters['rt_60'])
                            list_snr_testing.append(parameters['snr'])
                            list_signal_type_testing.append(parameters['signal_type'])

                            list_error.append(Evaluation.angular_error(expected, predicted, NUM_CLASSES) / NUM_CLASSES * MAX_THETA)
                            list_error_srpphat.append(
                                Evaluation.angular_error(expected, predicted_srpphat, NUM_CLASSES) / NUM_CLASSES * MAX_THETA)
                            list_error_music.append(
                                Evaluation.angular_error(expected, predicted_music, NUM_CLASSES) / NUM_CLASSES * MAX_THETA)

                            if list_error[idx] > 10:
                                list_weird.append((expected, predicted))

                            print(
                                f"{idx:0{num_zeros}d}/{NUM_SAMPLES:0{num_zeros}d} CNN: Angular error: {list_error[idx]} degrees")
                            print(
                                f"{idx:0{num_zeros}d}/{NUM_SAMPLES:0{num_zeros}d} SRP: Angular error: {list_error_srpphat[idx]} degrees")
                            print(
                                f"{idx:0{num_zeros}d}/{NUM_SAMPLES:0{num_zeros}d} MUSIC: Angular error: {list_error_music[idx]} degrees")

                            sys.stdout.write("\r{0}>".format("=" * round(50*idx/NUM_SAMPLES)))
                            sys.stdout.flush()

                            idx += 1

                        # Write results to pandas table
                        df = pd.DataFrame({
                            'Target': list_targets,
                            'Prediction': list_predictions,
                            'Prediction_SRPPHAT': list_predictions_srpphat,
                            'Prediction_MUSIC': list_predictions_music,
                            # 'Tracked': list_kalman,
                            'T60': list_rt_60_testing,
                            'SNR': list_snr_testing,
                            'Signal Type': list_signal_type_testing
                        })
                        # df.to_csv(
                        #     path_or_buf=file_name,
                        #     index=False)

                        MAE_CNN = np.mean(list_error)
                        MAE_SRPPHAT = np.mean(list_error_srpphat)
                        MAE_MUSIC = np.mean(list_error_music)

                        acc_model, acc_srpphat, acc_music = Evaluation.calculate_accuracy(df, NUM_CLASSES)

                        print(' ')

                        print(
                            f"CNN: Average angular error: {np.mean(list_error)} [{np.median(list_error)}] degrees, MAE: {MAE_CNN}, Accuracy: {acc_model}")
                        print(
                            f"SRP-PHAT: Average angular error: {np.mean(list_error_srpphat)} [{np.median(list_error_srpphat)}] degrees, MAE: {MAE_SRPPHAT}, Accuracy: {acc_srpphat}")
                        print(
                            f"MUSIC: Average angular error: {np.mean(list_error_music)} [{np.median(list_error_music)}] degrees, MAE: {MAE_MUSIC}, Accuracy: {acc_music}")

    Evaluation.plot_error(df=df, num_classes=NUM_CLASSES)
    print("done.")


    # os.system('shutdown -s')