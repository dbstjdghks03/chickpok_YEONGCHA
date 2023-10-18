import os
import json
from nptdms import TdmsFile
import re
import numpy as np

root = "/content/drive/MyDrive/data set"

class_to_idx = {
    'Yes': 0,
    'No': 1
}


def tdms_preprocess(tdms_path):
    tdms_file = TdmsFile(tdms_path)

    if ('Channel97' in tdms_file['RawData']) & ('Channel98' in tdms_file['RawData']):
        triggerA = tdms_file['RawData']['Channel97']
        triggerB = tdms_file['RawData']['Channel98']
        lst = []
        for i in range(len(triggerA.data)):
            if (triggerA.data[i] == 1) or (triggerB.data[i] == 1):
                lst.append(1)
            else:
                lst.append(0)

        indices = [index for index, value in enumerate(lst) if value == 1]

    elif 'Channel97' in tdms_file['RawData']:
        triggerA = tdms_file['RawData']['Channel97']
        lst = []
        for i in range(len(triggerA.data)):
            if triggerA.data[i] == 1:
                lst.append(1)
        indices = [index for index, value in enumerate(lst) if value == 1]

    elif 'Channel98' in tdms_file['RawData']:
        triggerB = tdms_file['RawData']['Channel98']
        lst = []
        for i in range(len(triggerB.data)):
            if triggerB.data[i] == 1:
                lst.append(1)
        indices = [index for index, value in enumerate(lst) if value == 1]

    L = list(name for name in tdms_file['RawData'].channels())
    L_str = list(map(str, L))
    data_lst = []
    peak_lst = []
    for string in L_str:
        num = re.sub(r'[^0-9]', '', string)
        if num and num != 97 and num != 98:
            selected_data = tdms_file['RawData'][f'Channel{num}']
            data_lst.append(selected_data.data)
            peak_lst.append(max(abs(selected_data.data)))

    data_sum = sum(data_lst)
    peakAmp = max(abs(data_sum))
    maxPeak = max(peak_lst)

    y = (data_sum / peakAmp) * maxPeak

    y = y[indices[0]:indices[-1]]

    y = np.pad(y, (0, 638825 - len(y)), mode='constant')

    return y


skip_list =["/content/drive/MyDrive/data set/train_json/221102_H", "/content/drive/MyDrive/data set/train_json/221107_N", "/content/drive/MyDrive/data set/train_json/221110_N"]
for dirpath, dirnames, files in os.walk(root + '/train_json'):
    print(f'Found directory: {dirpath}')
    if dirpath in skip_list:
        continue
    for file_name in files:
        if file_name.endswith(".json"):
            with open(dirpath + '/' + file_name, 'r') as f:
                data = json.load(f)
            folder_name = dirpath.split("/")[-1]
            s206_path = os.path.join(root+'/train_tdms/'+folder_name+'/S206/'+data['title_s206'])
            # batcam_path = os.path.join(root, '/train_tdms', folder_name, 'BATCAM2',
            #                            data['title_batcam2'])
            horn = class_to_idx[data['Horn']]
            if horn == "Yes":
                position = int(data['Position'])
            else:
                position = -1

            s206 = tdms_preprocess(s206_path)
            np.save(s206_path.split('.')[0], s206)
            # batcam = tdms_preprocess(batcam_path)
