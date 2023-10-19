import torch
import numpy as np
import re
from nptdms import TdmsFile

from src.datasets.dataset import YoungDataSet, PreProcess
from src.models.PCAmodel import PCAModel

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


#def get_model_for_eval():
#  """Gets the broadcasted model."""
#  model = models.resnet50(pretrained=True)
#  model.load_state_dict(bc_model_state.value)
#  model.eval()
#  return model

if __name__ == '__main__':

    s206_audio = tdms_preprocess('file path')
    s206 = PreProcess(s206_audio)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_components = 12

    model = PCAModel(n_components).to(device)
    model.load_state_dict(torch.load('Model path'))
    model.eval()
    # optimizer로는 Adam 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    stft = torch.tensor(s206.get_stft())
    mfcc = torch.tensor(s206.get_mfcc())
    sc = torch.tensor(s206.get_sc())

    output = model(mfcc, sc)
    pred = output.max(1, keepdim=True)[1]
    print(pred)
