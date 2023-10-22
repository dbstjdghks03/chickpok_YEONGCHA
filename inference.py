# train model with pca

import torch
import numpy as np
import re
from nptdms import TdmsFile

from src.datasets.dataset import YoungDataSet, PreProcess
from src.models.PCAmodel import PCAModel

def get_numpy_from_nonfixed_2d_array(input, fixed_length=638825, padding_value=0):
    output = np.pad(input, (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length]
    return output


def tdms_preprocess(tdms_path):
    tdms_file = TdmsFile(tdms_path)
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

    y = get_numpy_from_nonfixed_2d_array((data_sum / peakAmp) * maxPeak)

    return y


if __name__ == '__main__':

    s206_audio = tdms_preprocess('./dataset/train_tdms/221102_H/S206/Test_10.tdms')
    s206 = PreProcess(s206_audio)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_components = 10

    model = PCAModel(n_components).to(device)
    model.load_state_dict(torch.load('./valid_best_model_withpca.pt'))
    model.eval()
    # optimizer로는 Adam 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# +
#stft = torch.tensor(s206.get_stft()).to(device).float()
# -

mfcc = s206.get_mfcc()
mfct = mfcc
mfcc = np.array([mfcc,mfct])
mfcc = torch.tensor(mfcc).to(device).float()

sc = s206.get_sc()

sc.shape

st = sc
sc = np.concatenate((st,sc))
sc = torch.tensor(sc).to(device).float()
sc = torch.reshape(sc, (2, 20001, -1))



output = model(mfcc, sc)

pred = output.max(1, keepdim=True)[1]
print(pred)

output

# train model with out pca

# +
import torch
import numpy as np
import re
from nptdms import TdmsFile

from src.datasets.dataset import YoungDataSet, PreProcess
from src.models.PCAmodel import PCAModel


# -

def get_numpy_from_nonfixed_2d_array(input, fixed_length=4000000, padding_value=0):
    output = np.pad(input, (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length]
    return output


def tdms_preprocess(tdms_path):
    tdms_file = TdmsFile(tdms_path)
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

    y = get_numpy_from_nonfixed_2d_array((data_sum / peakAmp) * maxPeak)

    return y


if __name__ == '__main__':

    s206_audio = tdms_preprocess('../dataset/train_tdms/221102_H/S206/Test_10.tdms')
    s206 = PreProcess(s206_audio)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_components = 30

    model = PCAModel(n_components).to(device)
    model.load_state_dict(torch.load('./valid_best_model.pt'))
    model.eval()
    # optimizer로는 Adam 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    
    s206_audio = np.load('../dataset/train_tdms/221107_N/S206/test_07.npy')
    s206 = PreProcess(s206_audio)

    mfcc = s206.get_mfcc()
    #mfct = mfcc
    #mfcc = np.array([mfcc,mfct])
    mfcc = torch.tensor(mfcc).to(device).float()
    mfcc = torch.reshape(mfcc, (1, 3, 10, 2324))
    

mfcc.shape

if __name__ == '__main__':
    
    s206_audio = np.load('../dataset/train_tdms/221107_N/S206/test_07.npy')
    s206 = PreProcess(s206_audio)

    mfcc = s206.get_mfcc()
    mfct = mfcc
    mfcc = np.array([mfcc,mfct])
    mfcc = torch.tensor(mfcc).to(device).float()

    sc = s206.get_sc()
    st = sc
    sc = np.concatenate((st,sc))
    sc = torch.tensor(sc).to(device).float()
    sc = torch.reshape(sc, (2, 3195, -1))

    sc.shape

    n_components = 40

    model = PCAModel(n_components).to(device)
    model.load_state_dict(torch.load('../overfitted_model.pt'))
    model.eval()
    
    # optimizer로는 Adam 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    output = model(mfcc, sc)
    print(output)
    if output[0][0] < 0:
        print(f"horn_O and position is {output[0][1]}")
    else:
        print("horn_X")

# +
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


for dirpath, dirnames, files in os.walk(root + '/train_json'):
    print(f'Found directory: {dirpath}')
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

