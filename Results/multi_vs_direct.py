import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import COLORS


list_files = [
              'gadoae_multi_2d_SNR_0_T60_0.5_uncertainty_0.0_VAD_Energy',
              'gadoae_multi_2d_SNR_5_T60_0.5_uncertainty_0.0_VAD_Energy',
              'gadoae_multi_2d_SNR_10_T60_0.5_uncertainty_0.0_VAD_Energy',
              'gadoae_multi_2d_SNR_15_T60_0.5_uncertainty_0.0_VAD_Energy',
              'gadoae_multi_2d_SNR_20_T60_0.5_uncertainty_0.0_VAD_Energy',
              'gadoae_15_2d_SNR_0_T60_0.5_uncertainty_0.0_VAD_Energy',
              'gadoae_15_2d_SNR_5_T60_0.5_uncertainty_0.0_VAD_Energy',
              'gadoae_15_2d_SNR_10_T60_0.5_uncertainty_0.0_VAD_Energy',
              'gadoae_15_2d_SNR_15_T60_0.5_uncertainty_0.0_VAD_Energy',
              'gadoae_15_2d_SNR_20_T60_0.5_uncertainty_0.0_VAD_Energy',
            ]

filename = 'Multi'

# new_list = []
# for item in list_files:
#     new_list.append(item.replace('0.5', '0.13'))
# list_files = new_list
# filename = filename.replace('0-50','0-13')

def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)


def angular_error(prediction, label, num_classes):
    if len(prediction) > 0:
        error = [0] * len(prediction)
        for idx in range(len(prediction)):
            error[idx] = fac * np.abs(np.angle(np.exp(1j * 2 * np.pi * float(prediction[idx] - label[idx]) / num_classes))) * num_classes / (2 * np.pi)
        return error
    else:
        return None


def calculate_accuracy(prediction, label, num_classes, margin):
    if len(prediction) > 0:
        error_list = angular_error(prediction, label, num_classes)
        accuracy = np.zeros(len(error_list))
        for idx, error in enumerate(error_list):
            accuracy[idx] = (error <= (margin / num_classes * 360))
        return sum(accuracy) / len(prediction) * 100


def calculate_rmse(prediction, label, num_classes):
    if len(prediction) > 0:
        error_list = angular_error(prediction, label, num_classes)
        return np.sqrt(np.mean(np.power(error_list, 2)))


def calculate_mae(prediction, label, num_classes):
    if len(prediction) > 0:
        error_list = angular_error(prediction, label, num_classes)
        return np.mean(error_list)


num_classes = 72
margin = 1

data_dnn_acc = []
data_srp_acc = []
data_music_acc = []
data_dnn_rmse = []
data_srp_rmse = []
data_music_rmse = []
data_dnn_mae = []
data_srp_mae = []
data_music_mae = []

fac = 5.0

for idx, item in enumerate(list_files):
    df = pd.read_csv(f'./{item}.csv')

    data_dnn_acc.append(calculate_accuracy(df['Prediction'].tolist(), df['Target'].tolist(), num_classes, 1))
    data_dnn_rmse.append(calculate_rmse(df['Prediction'].tolist(), df['Target'].tolist(), num_classes))
    data_dnn_mae.append(calculate_mae(df['Prediction'].tolist(), df['Target'].tolist(), num_classes))

    data_srp_acc.append(calculate_accuracy(df['Prediction_SRPPHAT'].tolist(), df['Target'].tolist(), num_classes, 1))
    data_music_acc.append(calculate_accuracy(df['Prediction_MUSIC'].tolist(), df['Target'].tolist(), num_classes, 1))
    data_srp_rmse.append(calculate_rmse(df['Prediction_SRPPHAT'].tolist(), df['Target'].tolist(), num_classes))
    data_music_rmse.append(calculate_rmse(df['Prediction_MUSIC'].tolist(), df['Target'].tolist(), num_classes))
    data_srp_mae.append(calculate_mae(df['Prediction_SRPPHAT'].tolist(), df['Target'].tolist(), num_classes))
    data_music_mae.append(calculate_mae(df['Prediction_MUSIC'].tolist(), df['Target'].tolist(), num_classes))

# mean_cnn_chaakrabarty = np.mean(data_dnn_mae)
# mean_dnn_full = np..mean()

# ACCURACY

fig1, axs = plt.subplots(2, sharex=True)
axs[0].plot(data_dnn_acc[0:5], 'x-', color='tab:orange', label='DNN (multi)')
axs[0].plot(data_dnn_acc[5:10], '*-', color='tab:red', label='DNN (M=15)')
axs[0].plot(data_srp_acc[5:10], '--o', color='tab:blue', label='SRP-PHAT')
axs[0].plot(data_music_acc[0:5], ':d', color='tab:green', label='MUSIC')
axs[0].set_xticks(range(0, 5))
axs[0].set_xticklabels([0, 5, 10, 15, 20])
# axs[0].set_yticks([0, 25, 50, 75, 100])
axs[0].set_axisbelow(True)
axs[0].set(ylim = [35, 110])
axs[0].yaxis.grid(color='lightgrey', linestyle='dashed')
axs[0].set(ylabel='Accuracy [%]')
# axs[0].legend(loc='lower right')
# axs[0].legend(loc='lower right', bbox_to_anchor=(1.0, -0.4), framealpha=1.0)
# axs[0].title.set_text('Trained using variable M (multi)')
plt.gcf().subplots_adjust(bottom=0.2)
#
axs[1].plot(data_dnn_mae[0:5], 'x-', color='tab:orange', label='DNN (multi)')
axs[1].plot(data_dnn_mae[5:10], '*-', color='tab:red', label='DNN (M=15)')
axs[1].plot(data_srp_mae[5:10], '--o', color='tab:blue', label='SRP-PHAT')
axs[1].plot(data_music_mae[0:5], ':d', color='tab:green', label='MUSIC')
axs[1].set_xticks(range(0, 5))
axs[1].set_xticklabels([0, 5, 10, 15, 20])
axs[1].set_axisbelow(True)
axs[1].set(ylim=[-2, 24])
axs[1].yaxis.grid(color='lightgrey', linestyle='dashed')
axs[1].set(ylabel='MAE [°]')
axs[1].set(xlabel='SNR [dB]')
# axs[1].title.set_text('Trained using M=15')
# axs[1].legend(loc='upper right')
axs[1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.4), framealpha=1.0)
plt.gcf().subplots_adjust(bottom=0.1)
plt.suptitle('Accuracy and MAE of localization algorithms, evaluated using M=15, T60=0.5s')
# plt.show()

plt.savefig('DNN_multi_vs_direct_Accuracy_MAE_SNR', bbox_inches='tight', transparent="True", pad_inches=0.1, dpi=300)
plt.close()


# MAE
#
# fig1, axs = plt.subplots(2, sharex=True)
# axs[0].plot(data_dnn_mae[0:5], 'x-', color='tab:orange', label='DNN')
# axs[0].plot(data_srp_mae[0:5], '--o', color='tab:blue', label='SRP-PHAT')
# axs[0].plot(data_music_mae[0:5], ':d', color='tab:green', label='MUSIC')
# axs[0].set_xticks(range(0, 5))
# axs[0].set_xticklabels([0, 5, 10, 15, 20])
# # axs[0].set_yticks([0, 25, 50, 75, 100])
# axs[0].set_axisbelow(True)
# axs[0].set(ylim = [-2, 24])
# axs[0].yaxis.grid(color='lightgray', linestyle='dashed')
# axs[0].legend(loc='upper right')
# axs[0].set(ylabel ='MAE [°]')
# axs[0].title.set_text('Trained using variable M (multi)')
# plt.gcf().subplots_adjust(bottom=0.2)
# #
# axs[1].plot(data_dnn_mae[5:10], 'x-', color='tab:orange', label='DNN')
# axs[1].plot(data_srp_mae[5:10], '--o', color='tab:blue', label='SRP-PHAT')
# axs[1].plot(data_music_mae[5:10], ':d', color='tab:green', label='MUSIC')
# axs[1].set_xticks(range(0, 5))
# axs[1].set_xticklabels([0, 5, 10, 15, 20])
# axs[1].set_axisbelow(True)
# axs[1].set(ylim = [-2, 24])
# axs[1].yaxis.grid(color='lightgray', linestyle='dashed')
# axs[1].set(ylabel ='MAE [°]')
# axs[1].set(xlabel = 'SNR [dB]')
# axs[1].title.set_text('Trained using M=15')
#
# # axs[1].legend(loc='center right', bbox_to_anchor=(1.0, 1.15), framealpha=1.0)
# axs[1].legend(loc='upper right')
# plt.gcf().subplots_adjust(bottom=0.1)
# # plt.suptitle('Accuracy and MAE of DNN trained using variable M (multi), evaluated using M=15, T60=0.5s')
# # plt.show()
#
# plt.savefig('DNN_multi_vs_direct_MAE_SNR', bbox_inches='tight', transparent="True", pad_inches=0.1, dpi=300)
# plt.close()
