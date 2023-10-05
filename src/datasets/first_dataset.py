#데이터 로딩 및 DataLoader 생성:
# 예를 들어, stft_data, mfcc_data, sc_data, labels를 준비한 후
tils.data.DataLoader(custom_dataset, batch_size=32, shuffle=True)