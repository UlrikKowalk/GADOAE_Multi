
import numpy as np
import matplotlib.pyplot as plt


def P2R(radii, angles):
    return radii * np.exp(1j*angles)


def R2P(x):
    return np.abs(x), np.angle(x)


def find_peaks(in_array, peak_importance: int=5): # Es scheint, als ob Peaks ganz am (unteren) Rand nicht immer gefunden werden?!?

    peaks = []
    in_len = len(in_array)

    if peak_importance > in_len:
        peak_importance = in_len - 1

    # Only a maximum of N/2 peaks possible (+1 to account for truncation)
    # prov = (in_len + 1) / 2
    accu = []#ArrayList()

    # Edge case: Array has either 1 or 0 entries -> return 0
    if in_len < 2:
        peaks.append(0)
        return peaks

    # Edge case: Array has 2 entries 1>2 / 2>1 / 1==2
    if in_len == 2:
        if in_array[0] > in_array[1]:
            accu.append(0)
        elif in_array[0] < in_array[1]:
            accu.append(1)
        else:
            accu.append(0)
            accu.append(1)

    # Padding/repeating the original array at the start and end by peak importance # samples
    # tmp_array = ArrayList(size=(2 * peak_importance + in_len), val=0.0)
    tmp_array = [0.0] * (2 * peak_importance + in_len)
    for idx in range(peak_importance):
        tmp_array[idx] = in_array[in_len - peak_importance + idx]
        tmp_array[in_len + peak_importance + idx] = in_array[idx]
    for idx in range(in_len):
        tmp_array[idx + peak_importance] = in_array[idx]

    # peak_appeal = [0] * (in_len + 2*peak_importance)
    # for idx in range(peak_importance, in_len + peak_importance):
    #     pad = 1
    #     while idx-pad >= 0 and idx+pad <= in_len+peak_importance and tmp_array[idx] > tmp_array[idx - pad] and tmp_array[idx] > tmp_array[idx + pad]:
    #         peak_appeal[idx] += 1
    #         pad += 1

    peak_appeal = [0] * (in_len)
    for idx in range(in_len):
        pad = 1
        while pad <= in_len and in_array[idx] > in_array[(idx - pad) % in_len] and in_array[idx] > in_array[(idx + pad) % in_len]:
            peak_appeal[idx] += 1
            pad += 1

    accu = []
    for idx in range(in_len):
        if peak_appeal[idx] >= peak_importance:
            accu.append(idx)

    # fig, axs = plt.subplots(2)
    # axs[0].plot(np.log(in_array))
    # for idx, pk in enumerate(peak_appeal):
    #     if pk >= peak_importance:
    #         axs[0].plot(idx, np.log(pk), 'ro')
    # axs[1].plot(peak_appeal)
    # plt.show()

    # Search the array from its original beginning to its original end
    # for idx in range(peak_importance, in_len):
    #
    #     b_peak = [0.0] * peak_importance
    #     for ipad in range(1, peak_importance + 1):
    #         if tmp_array[idx] > tmp_array[idx - ipad] and tmp_array[idx] > tmp_array[idx + ipad]:
    #             b_peak[ipad - 1] = 0
    #         else:
    #             b_peak[ipad - 1] = 1
    #
    #     print()
    #     test = 0
    #     for ipad in range(len(b_peak)):
    #         test += b_peak[ipad]
    #     if test == 0:
    #         accu.append(idx - peak_importance)





    #
    # plt.plot(in_array)
    # for p in accu:
    #     plt.plot(p, in_array[p], 'ro')
    # plt.show()
    # print(in_array)

    return accu


def quadratic_interpolation(mag_prev, mag, mag_next):
    pos = (mag_next - mag_prev) / (2.0 * (2.0 * mag - mag_next - mag_prev))
    val = mag - 0.25 * (mag_prev - mag_next) * pos
    return [pos, val]


# def sharpen_logits(logits):
#
#     spec = np.fft.fft(logits)
#     spec_new = np.divide(spec, np.abs(spec))
#     result = np.real(np.fft.ifft(spec_new))
#
#     return result


def find_real_peaks(logits, peaks, max_theta, num_classes):

    step_theta = max_theta / num_classes
    real_peaks = [-255] * len(peaks)

    for ipk, peak in enumerate(peaks):

        # Avoid circular confusion
        if peak == 0:
            tmp = quadratic_interpolation(logits[-1], logits[0], logits[1])
        elif peak == len(logits) - 1:
            tmp = quadratic_interpolation(logits[-2], logits[-1], logits[0])
        else:
            tmp = quadratic_interpolation(logits[peak - 1], logits[peak], logits[peak + 1])

        real_peaks[ipk] = (tmp[0] + peak, tmp[1])

        if real_peaks[ipk][0] < 0.0:
            real_peaks[ipk] = (0.0, tmp[1])
        elif real_peaks[ipk][0] > max_theta / step_theta - 1.0:
            real_peaks[ipk] = (max_theta / step_theta - 1.0, tmp[1])

    return real_peaks


def sort_by_strength(estimates):
    return sorted(estimates, key=lambda pk: pk[1], reverse=True)


def convert_strength_to_lin(estimates):
    for idx in range(len(estimates)):
        estimates[idx] = (estimates[idx][0], np.power(10, estimates[idx][1] / 20))
    return estimates


def convert_logits_to_lin(logits):
    for idx in range(len(logits)):
        logits[idx] = np.power(10, logits[idx] / 20)
    return logits
