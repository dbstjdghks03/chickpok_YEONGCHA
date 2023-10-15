import os
from nptdms import TdmsFile
import re
import torch
import matplotlib.pyplot as plt
import json

dir1 = "/content/drive/MyDrive/data set/train_tdms/221108_차세대전동차/S206"
dir2 = "/content/drive/MyDrive/data set/train_tdms/221108_차세대전동차/BATCAM2"


def getTdms(path):
    tdms_file = TdmsFile(path)
    L = list(name for name in tdms_file['RawData'].channels())
    L_str = list(map(str, L))
    data_lst = []
    peak_lst = []
    for string in L_str:
        num = re.sub(r'[^0-9]', '', string)
        if num:
            selected_data = tdms_file['RawData'][f'Channel{num}']
            data_lst.append(selected_data.data)
            peak_lst.append(max(abs(selected_data.data)))
    data_sum = sum(data_lst)
    peakAmp = max(abs(data_sum))
    maxPeak = max(peak_lst)
    arr = (data_sum / peakAmp) * maxPeak
    return arr


for dirpath, dirnames, files in os.walk('//home/gritte/workspace/MyCanvas/data/data_set/train_json'):
    print(f'Found directory: {dirpath}')
    for file_name in files:
        if file_name.endswith(".json"):
            with open(dirpath + '/' + file_name, 'r') as f:
                data = json.load(f)
            folder_name = dirpath.split("/")[-1]
            s206_path = os.path.join('/home/gritte/workspace/MyCanvas/data/data_set/train_tdms', folder_name, 'S206',
                                     data['title_s206'])
            batcam_path = os.path.join('/home/gritte/workspace/MyCanvas/data/data_set/train_tdms', folder_name, 'BATCAM2',
                                       data['title_batcam2'])

            print(s206_path, batcam_path)

            try:
                arr = getTdms(s206_path)
            except:
                arr = []

            try:
                arr2 = getTdms(batcam_path)
            except:
                arr2 = []

            fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns, and figure size

            # 첫 번째 그래프
            axs[0].plot(arr)
            axs[0].set_title('s206')
            axs[0].set_xlabel(folder_name + 'S206' + data['title_s206'])

            # 두 번째 그래프
            axs[1].plot(arr2)
            axs[1].set_title('batcam')
            axs[1].set_xlabel(folder_name + 'BATCAM2' + data['title_batcam2'])

            # 그래프를 화면에 보이기
            plt.tight_layout()  # 자동으로 레이아웃을 조절하여 오버랩 방지
            plt.savefig(folder_name + '/'+data['title_s206'].split('.')[0]+'.png')