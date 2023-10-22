import torch
import numpy as np
import re
from nptdms import TdmsFile
import argparse

from src.datasets.dataset import YoungDataSet, PreProcess, TestYoungDataSet
from src.models.PCAmodel import PCAModel

parser = argparse.ArgumentParser()

# 3. parser.add_argument로 받아들일 인수를 추가해나간다.
parser.add_argument('--filepath', type=str)

args = parser.parse_args()

# filepath = args.filepath
filepath = "../dataset/train_tdms/221102_H/S206/Test_10.tdms"


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_components = 40

if __name__ == '__main__':
    s206_audio = tdms_preprocess(filepath)
    s206 = PreProcess(s206_audio)

    mfcc = s206.get_mfcc()
    mfct = mfcc
    mfcc = np.array([mfcc, mfct])
    mfcc = torch.tensor(mfcc).to(device).float()

    sc = s206.get_sc()
    st = sc
    sc = np.concatenate((st, sc))
    sc = torch.tensor(sc).to(device).float()
    sc = torch.reshape(sc, (2, 3195, -1))

    n_components = 40

    model = PCAModel(n_components).to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    position_mse = 0
    test_len = 0
    correct_predictions = 0

    test_dataset = YoungDataSet(root=root, is_npy=True, transform=None)

    for i, (mfcc, sc, horn, position) in enumerate(test_loader):
        mfcc, sc, horn, position = mfcc.to(device).float(), sc.to(
            device).float(), horn.to(device), position.to(device).float()
        output = model(mfcc, sc)
        epoch_test_loss += loss(output, horn, position).item()

        predictions = torch.tensor([1 if out[0] > 0 else -1 for out in output]).to(device)
        correct_predictions += (predictions == horn).float().sum()
        position_mse += MSELoss(output[:, 1], position)
        test_len += position.shape[0]

    accuracy = correct_predictions / test_len

    print(f'train_loss: {epoch_train_loss / train_len}')
    print('Horn Accuracy: {}/{} ({:.2f}%), Position MSE: {}\n'.format(
        correct_predictions, test_len,
        accuracy, position_mse / test_len))


    output = model(mfcc, sc)
    print(output)

    if output[0][0] > 1:
        print("위험 수치가")

    elif output[0][0] < 1:
        pred = 0

    else:
        pred = 1

    print(pred)


