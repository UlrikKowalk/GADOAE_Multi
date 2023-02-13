import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
import os

import FindPeaks
# from SourceManager import SourceManager


def get_filename(trained_net, snr, t60, uncertainty, parameters):
    return f'Results/{os.path.splitext(os.path.basename(trained_net))[0]}_SNR_{snr}_T60_{t60}_' \
            f'augmentation_{parameters["augmentation_style"]}_uncertainty_{uncertainty}_VAD_{parameters["vad_name"]}' \
            f'.csv'


def estimate_srpphat_with_interpolation(model, sample, MAX_THETA, NUM_CLASSES, voice_activity):

    num_frames = int(np.floor(sample.shape[1] / model.frame_length))
    predictions = np.zeros(shape=(num_frames,))
    predictions_vad_tmp = np.zeros(shape=(num_frames,))

    num_vad = 0
    for frame, vad in enumerate(voice_activity):
        idx_in = frame * model.frame_length
        idx_out = idx_in + model.frame_length
        prediction = model.forward(sample[:, idx_in:idx_out])
        # pred_max = np.argmax(prediction)

        # print(prediction)
        peaks = FindPeaks.find_peaks(prediction)
        estimates = FindPeaks.find_real_peaks(prediction, peaks, MAX_THETA, NUM_CLASSES)
        estimates = FindPeaks.sort_by_strength(estimates)[0]
        class_predicted = estimates[0]

        predictions[frame] = class_predicted

        if vad:
            predictions_vad_tmp[num_vad] = class_predicted
            num_vad += 1

    predictions_vad = predictions_vad_tmp[:num_vad]
    estimate = np.median(predictions_vad)

    return estimate, predictions


def estimate_srpphat(model, sample, voice_activity):

    num_frames = int(np.floor(sample.shape[1] / model.frame_length))
    predictions = np.zeros(shape=(num_frames,))
    predictions_vad_tmp = np.zeros(shape=(num_frames,))

    num_vad = 0
    for frame, vad in enumerate(voice_activity):
        idx_in = frame * model.frame_length
        idx_out = idx_in + model.frame_length
        prediction = model.forward(sample[:, idx_in:idx_out])
        pred_max = np.argmax(prediction)
        predictions[frame] = pred_max

        if vad:
            predictions_vad_tmp[num_vad] = pred_max
            num_vad += 1

    predictions_vad = predictions_vad_tmp[:num_vad]
    estimate = np.median(predictions_vad)

    return estimate, predictions


def estimate_music_with_interpolation(model, sample, MAX_THETA, NUM_CLASSES, voice_activity):

    num_frames = int(np.floor(sample.shape[1] / model.frame_length))
    predictions = np.zeros(shape=(num_frames,))
    predictions_vad_tmp = np.zeros(shape=(num_frames,))

    num_vad = 0
    for frame, vad in enumerate(voice_activity):
        idx_in = frame * model.frame_length
        idx_out = idx_in + model.frame_length
        prediction = model.forward(sample[:, idx_in:idx_out])
        # pred_max = np.argmax(prediction)

        peaks = FindPeaks.find_peaks(prediction)
        estimates = FindPeaks.find_real_peaks(prediction, peaks, MAX_THETA, NUM_CLASSES)
        estimates = FindPeaks.sort_by_strength(estimates)[0]
        class_predicted = estimates[0]

        predictions[frame] = class_predicted

        if vad:
            predictions_vad_tmp[num_vad] = class_predicted
            num_vad += 1

    predictions_vad = predictions_vad_tmp[:num_vad]
    estimate = np.median(predictions_vad)

    return estimate, predictions


def estimate_music(model, sample, voice_activity):
    num_frames = int(np.floor(sample.shape[1] / model.frame_length))
    predictions = np.zeros(shape=(num_frames,))
    predictions_vad_tmp = np.zeros(shape=(num_frames,))
    num_vad = 0

    for frame, vad in enumerate(voice_activity):
        idx_in = frame * model.frame_length
        idx_out = idx_in + model.frame_length
        prediction = model.forward(sample[:, idx_in:idx_out])
        pred_max = np.argmax(prediction)
        predictions[frame] = pred_max
        if vad:
            predictions_vad_tmp[num_vad] = pred_max
            num_vad += 1

    predictions_vad = predictions_vad_tmp[:num_vad]

    estimate = np.median(predictions_vad)

    return estimate, predictions


def estimate_cnn_with_interpolation(model, sample, MAX_THETA, NUM_CLASSES, voice_activity):
    model.eval()

    prediction_output = model.forward(sample)
    predictions = np.argmax(prediction_output.cpu().detach().numpy(), axis=1)
    predictions_vad_tmp = np.zeros(shape=(sample.shape[0],))

    num_vad = 0
    for frame, vad in enumerate(voice_activity):
        if vad:

            peaks = FindPeaks.find_peaks(prediction_output[frame, :])
            estimates = FindPeaks.find_real_peaks(prediction_output[frame, :], peaks, MAX_THETA, NUM_CLASSES)
            estimates = FindPeaks.sort_by_strength(estimates)[0]
            class_predicted = estimates[0]

            predictions_vad_tmp[num_vad] = class_predicted
            num_vad += 1

    predictions_vad = predictions_vad_tmp[:num_vad]

    estimate = np.median(predictions_vad)

    return estimate, predictions


def estimate_dnn(model, sample, voice_activity):

    model.eval()

    prediction_output = model.forward(sample)
    predictions = np.argmax(prediction_output.cpu().detach().numpy(), axis=1)
    predictions_vad_tmp = np.zeros(shape=(sample.shape[0],))

    num_vad = 0
    for frame, vad in enumerate(voice_activity):
        if vad:
            predictions_vad_tmp[num_vad] = predictions[frame]
            num_vad += 1

    predictions_vad = predictions_vad_tmp[:num_vad]
    # plt.fill_between(
    #     x=range(sample.shape[0]),
    #     y1=voice_activity*72,
    #     where=(voice_activity > 0),
    #     color="r",
    #     alpha=1.0)
    # plt.plot(predictions, 'b')
    # plt.show()

    estimate = np.median(predictions_vad)
    return estimate, predictions

def predict_with_interpolation_noVAD(model, sample, target, class_mapping, device, PARAMETERS, MAX_THETA, NUM_CLASSES):
    model.eval()

    predictions = model(sample)
    estimates_per_frame = []
    targets_per_frame = []
    for pred, tar in zip(predictions, target):
        cost = pred.cpu().detach().numpy()
        max_idx = np.argmax(cost)
        estimates = FindPeaks.find_real_peaks(cost, [max_idx], MAX_THETA, NUM_CLASSES)
        estimate = FindPeaks.sort_by_strength(estimates)[0][0]

        estimates_per_frame.append(estimate)
        targets_per_frame.append(float(tar))

    return np.median(estimates_per_frame), np.median(targets_per_frame), 0, 0


def predict(model, sample, target, class_mapping):
    model.eval()
    with torch.no_grad():
        prediction = model(sample)
        predicted_index = torch.argmax(prediction[0])
        class_predicted = class_mapping[predicted_index]
        class_expected = class_mapping[target]
    return class_predicted, class_expected


def predict_multiframe_with_source_management(model, sample, target, class_mapping, device, PARAMETERS, MAX_THETA, NUM_CLASSES):
    model.eval()
    with torch.no_grad():
        prediction = model(sample)

        logit_accu = np.zeros(shape=(PARAMETERS['num_classes'], sample.shape[0]))

        dt = PARAMETERS['frame_length'] / PARAMETERS['sample_rate']
        source_manager = SourceManager(dt, PARAMETERS['num_classes'], device)

        class_argmax = []
        class_expected = []
        class_all = []
        class_kalman = []
        class_kalman_all_sources = []
        all_estimates = []

        frame = 0
        for pred, tar in zip(prediction, target):

            pred = pred.cpu().detach().numpy()
            logit_accu[:, frame] = pred
            logits = pred.tolist()

            lin_logits = logits

            # lin_logits = FindPeaks.convert_logits_to_lin(logits)
            # lin_logits = logits

            # len_filter = 3
            # plt.plot(lin_logits, 'k')
            # lin_logits = scipy.signal.medfilt(volume=lin_logits, kernel_size=(len_filter,))
            # plt.plot(lin_logits, 'r')
            # plt.show()

            # estimates = FindPeaks.find_peaks(lin_logits, peak_importance=3)

            estimates = scipy.signal.find_peaks(lin_logits)[0]
            # estimates = [np.argmax(pred)]
            # print(estimates)



            # print(len(estimates))
            estimates = FindPeaks.find_real_peaks(lin_logits, estimates, MAX_THETA, NUM_CLASSES)
            estimates = FindPeaks.sort_by_strength(estimates)

            all_estimates.append(estimates)

            sources = source_manager.track_sources(estimates)

            if len(sources) > 0 and sources[0] != -255:
                frm = []
                srcs = []
                for src in sources:
                    frm.append((src.get_angle(), src.get_strength()))
                    srcs.append([src.get_angle(), src.get_strength()])
                class_kalman_all_sources.append(frm)
                srcs = sorted(srcs, key=lambda x: x[1], reverse=True)
                if np.isnan(srcs[0][0]):
                    print('NAN')
                class_kalman.append(srcs[0][0])
            else:
                class_kalman.append(-255)
                class_kalman_all_sources.append([])

            if tar != -255:

                # idx = np.argmax(pred)
                # Interpolated maximum (not really an index now)
                idx = estimates[0][0]

                class_argmax.append(idx)
                class_expected.append(class_mapping[tar])
                # class_all.append(idx)

            # else:
            #     class_all.append(None)

            frame += 1

        # fig, axs = plt.subplots(2)
        # axs[0].imshow(logit_accu, origin='lower', aspect='auto')
        #
        # for idx, frm in enumerate(all_estimates):
        #     axs[1].plot(idx, len(class_kalman_all_sources[idx]), 'k.')
        #     for esti in frm:
        #         axs[0].plot(idx, esti[0], 'wx', alpha=max([0.5, min([1.0, esti[1]])]))
        #
        # axs[0].plot(class_all, 'b--')
        #
        # # plot kalman sources
        # for idx, frm in enumerate(class_kalman_all_sources):
        #     if frm:
        #         for ifr, src in enumerate(frm):
        #             if ifr == 0:
        #                 axs[0].plot(idx, src[0], 'm.', alpha=max([0.5, min([1.0, src[1]])]))
        #             else:
        #                 axs[0].plot(idx, src[0], 'r.', alpha=max([0.5, min([1.0, src[1]])]))

        # plot argmax
        # for idx, estimates in enumerate(all_estimates):
        #     axs[0].plot(idx, estimates[0][0], 'm.')


        # axs[1].set_ylim([0, 6])
        # axs[0].set_xlim([0, frame-1])
        # axs[1].set_xlim([0, frame-1])
        # plt.show()



        class_var = np.var(class_argmax)
        class_argmax = np.median(class_argmax)
        class_expected = np.mean(class_expected)

        kalman_estimates = [i for i in class_kalman if i !=-255] # Fehlermaß ist nicht optimal weil wrap nicht berücksichtigt wird!
        if len(kalman_estimates) > 0:
            class_kalman = np.median(kalman_estimates)

        else:
            class_kalman = -255

    return class_argmax, class_expected, class_var, class_kalman


def predict_multiframe(model, sample, target, class_mapping):
    model.eval()
    with torch.no_grad():
        prediction = model(sample)
        predicted_index = torch.argmax(prediction, dim=-1)

        class_predicted = []
        class_expected = []
        class_all = []
        for idx, tar in zip(predicted_index, target):
            if tar != -255:
                class_predicted.append(class_mapping[idx])
                class_expected.append(class_mapping[tar])
                class_all.append(class_mapping[idx])
            else:
                class_all.append(-1)

        class_var = np.var(class_predicted)
        class_predicted = np.mean(class_predicted)
        class_expected = np.mean(class_expected)

    return class_predicted, class_expected, class_var


def predict_multisource(model, sample, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(sample)
        predicted_index = torch.argmax(predictions[0])
        class_predicted = class_mapping[predicted_index]

        class_expected = []
        for val in range(len(target)):
            class_expected.append(class_mapping[val])
    return class_predicted, class_expected


def predict_regression(model, sample, target):
    model.eval()
    with torch.no_grad():
        prediction = model(sample)
    return prediction.cpu().detach().numpy(), target.cpu().detach().numpy()


def angular_error(prediction, label, num_classes):
    error = np.abs(np.angle(np.exp(1j * 2 * np.pi * (prediction - label) / num_classes))) * num_classes / (2 * np.pi)
    return error


def plot_error(df, num_classes):

    targets = range(num_classes)
    errors = [[None]] * num_classes
    predictions = [[None]] * num_classes
    for target in targets:
        predictions[target] = df.loc[df['Target'] == target]['Prediction']
        tmp = df.loc[df['Target'] == target]['Prediction']
        tmp_err = []
        for err in tmp:
            tmp_err.append(angular_error(err, target, num_classes))
        errors[target] = tmp_err

    fig, ax = plt.subplots(2, 1)
    ax[0].boxplot(predictions)
    ax[1].boxplot(errors)
    ax[0].set_title('Predictions')
    ax[1].set_title('Errors')
    plt.gcf().subplots_adjust(bottom=0.1)
    plt.show()


def plot_distribution(labels, num_classes):

    target = range(num_classes)
    freq = np.zeros(num_classes)

    for label in labels:
         freq[label] += 1

    plt.bar(target, freq)
    plt.title('Target distribution')
    plt.show()


def angular_error_multisource(prediction, label, num_classes):

    print(prediction)
    print(label)

    error = 0
    for pred, lab in zip(prediction, label):
        error = error + np.abs(np.angle(np.exp(1j * 2 * np.pi * float(pred - lab) / num_classes))) * num_classes / (2 * np.pi)
    return error


def calculate_accuracy(df, num_classes):

    accuracy_model = 0
    accuracy_srpphat = 0
    accuracy_music = 0

    for idx in range(len(df)):

        tmp = df['Prediction'][idx]
        tmp_srpphat = df['Prediction_SRPPHAT'][idx]
        tmp_music = df['Prediction_MUSIC'][idx]
        target = df['Target'][idx]

        if angular_error(target, tmp, num_classes) <= 1:
            accuracy_model += 1
        if angular_error(target, tmp_srpphat, num_classes) <= 1:
            accuracy_srpphat += 1
        if angular_error(target, tmp_music, num_classes) <= 1:
            accuracy_music += 1

    return accuracy_model/len(df)*100, accuracy_srpphat/len(df)*100, accuracy_music/len(df)*100



