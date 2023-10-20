import numpy as np
import librosa
import random
import scipy
import torch
import torch.nn.functional as F
from src.datasets.utils.helper_funcs import AugBasic

# class change_pitch(object):
#     def __init__(self, sampling_rate, pitch_factor) -> None:
#         self.sampling_rate = sampling_rate
#         self.pitch_factor = pitch_factor
#
#     def __call__(self, data):
#         """
#         원본 데이터의 피치를 조절합니다.
#         """
#         return librosa.effects.pitch_shift(data, self.sampling_rate, self.pitch_factor)


class RandomLPHPFilter(AugBasic):
    def __init__(self, fs, p=0.5, fc_lp=None, fc_hp=None):
        self.p = p
        self.fs = fs
        self.fc_lp = fc_lp
        self.fc_hp = fc_hp
        self.num_taps = 15

    def __call__(self, sample):
        if random.random() < self.p:
            a = 0.25
            if random.random() < 0.5:
                fc = 0.5 + random.random() * 0.25
                filt = scipy.signal.firwin(self.num_taps, fc, window='hamming')
            else:
                fc = random.random() * 0.25
                filt = scipy.signal.firwin(self.num_taps, fc, window='hamming', pass_zero=False)
            filt = torch.from_numpy(filt).float()
            filt = filt / filt.sum()
            sample = F.pad(sample.view(1, 1, -1), (filt.shape[0] // 2, filt.shape[0] // 2), mode="reflect")
            sample = F.conv1d(sample, filt.view(1, 1, -1), stride=1, groups=1)
            sample = sample.view(-1)
        return sample


class RandomAmp(AugBasic):
    def __init__(self, low, high, p=0.5):
        self.low = low
        self.high = high
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            amp = torch.FloatTensor(1).uniform_(self.low, self.high)
            sample.mul_(amp)
        return sample


class RandomFlip(AugBasic):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample.data = torch.flip(sample.data, dims=[-1, ])
        return sample


class RandomAdd180Phase(AugBasic):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample.mul_(-1)
        return sample


class RandomAdditiveWhiteGN(AugBasic):
    def __init__(self, p=0.5, snr_db=30):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            w = torch.randn_like(sample).mul_(sgm)
            sample.add_(w)
        return sample


class RandomAdditiveUN(AugBasic):
    def __init__(self, snr_db=35, p=0.5):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.) * np.sqrt(3)
            w = torch.rand_like(sample).mul_(2 * sgm).add_(-sgm)
            sample.add_(w)
        return sample


class RandomAdditivePinkGN(AugBasic):
    def __init__(self, snr_db=35, p=0.5):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            n = sample.shape[-1]
            w = torch.randn(n)
            nn = n // 2 + 1
            k = torch.arange(1, nn + 1, 1).float()
            W = torch.fft.fft(w)
            W = W[:nn] / k.sqrt()
            W = torch.cat((W, W.flip(dims=(-1,))[1:-1].conj()), dim=-1)
            w = torch.fft.ifft(W).real
            w.add_(w.mean()).div_(w.std())
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            sample.add_(w.mul_(sgm))
        return sample


class RandomAdditiveVioletGN(AugBasic):
    def __init__(self, p=0.5, snr_db=35):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            n = sample.shape[-1]
            w = torch.randn(n)
            nn = n // 2 + 1
            k = torch.arange(1, nn + 1, 1).float()
            W = torch.fft.fft(w)
            W = W[:nn] * k
            W = torch.cat((W, W.flip(dims=(-1,))[1:-1].conj()), dim=-1)
            w = torch.fft.ifft(W).real
            w.add_(w.mean()).div_(w.std())
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            sample.add_(w.mul_(sgm))
        return sample


class RandomAdditiveRedGN(AugBasic):
    def __init__(self, p=0.5, snr_db=35):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            n = sample.shape[-1]
            w = torch.randn(n)
            nn = n // 2 + 1
            k = torch.arange(1, nn + 1, 1).float()
            W = torch.fft.fft(w)
            W = W[:nn] / k
            W = torch.cat((W, W.flip(dims=(-1,))[1:-1].conj()), dim=-1)
            w = torch.fft.ifft(W).real
            w.add_(w.mean()).div_(w.std())
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            sample.add_(w.mul_(sgm))
        return sample


class RandomAdditiveBlueGN(AugBasic):
    def __init__(self, p=0.5, snr_db=35):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            n = sample.shape[-1]
            w = torch.randn(n)
            nn = n // 2 + 1
            k = torch.arange(1, nn + 1, 1).float()
            W = torch.fft.fft(w)
            W = W[:nn] * k.sqrt()
            W = torch.cat((W, W.flip(dims=(-1,))[1:-1].conj()), dim=-1)
            w = torch.fft.ifft(W).real
            w.add_(w.mean()).div_(w.std())
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            sample.add_(w.mul_(sgm))
        return sample


class RandomFreqShift(AugBasic):
    def __init__(self, sgm, fs, p=0.5):
        super().__init__(fs=fs)
        self.sgm = sgm
        self.p = p
        self.fft_params = {}
        self.fft_params['win_len'] = [512, 1024, 2048]
        self.fft_params['hop_len'] = [128, 256, 1024]
        self.fft_params['n_fft'] = [512, 1024, 2048]

    def __call__(self, sample):
        if random.random() < self.p:
            win_idx = random.randint(0, len(self.fft_params['win_len']) - 1)
            df = self.fs / self.fft_params['win_len'][win_idx]
            f_shift = torch.randn(1).mul_(self.sgm * df)
            t = torch.arange(0, self.fft_params['win_len'][win_idx], 1).float()
            w = torch.real(torch.exp(-1j * 2 * np.pi * t * f_shift))
            X = torch.stft(sample,
                           win_length=self.fft_params['win_len'][win_idx],
                           hop_length=self.fft_params['hop_len'][win_idx],
                           n_fft=self.fft_params['n_fft'][win_idx],
                           window=w,
                           return_complex=True)
            sample = torch.istft(X,
                                 win_length=self.fft_params['win_len'][win_idx],
                                 hop_length=self.fft_params['hop_len'][win_idx],
                                 n_fft=self.fft_params['n_fft'][win_idx])

        return sample


class RandomAddSine(AugBasic):
    def __init__(self, fs, snr_db=35, max_freq=50, p=0.5):
        self.snr_db = snr_db
        self.max_freq = max_freq
        self.min_snr_db = 30
        self.p = p
        self.fs = fs

    def __call__(self, sample):
        n = torch.arange(0, sample.shape[-1], 1)
        f = self.max_freq * torch.rand(1) + 3 * torch.randn(1)
        if random.random() < self.p:
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            t = n * 1. / self.fs
            s = (sample ** 2).mean().sqrt()
            sgm = s * np.sqrt(2) * 10 ** (-snr_db / 20.)
            b = sgm * torch.sin(2 * np.pi * f * t + torch.rand(1) * np.pi)
            sample.add_(b)

        return sample


class RandomAmpSegment(AugBasic):
    def __init__(self, low, high, max_len=None, p=0.5):
        self.low = low
        self.high = high
        self.max_len = max_len
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            if self.max_len is None:
                self.max_len = sample.shape[-1] // 10
            idx = random.randint(0, self.max_len)
            amp = torch.FloatTensor(1).uniform_(self.low, self.high)
            sample[idx: idx + self.max_len]@(amp)
        return sample


class RandomPhNoise(AugBasic):
    def __init__(self, fs, sgm=0.01, p=0.5):
        super().__init__(fs=fs)
        self.sgm = sgm
        self.p = p
        self.fft_params = {}
        self.fft_params['win_len'] = [512, 1024, 2048]
        self.fft_params['hop_len'] = [128, 256, 1024]
        self.fft_params['n_fft'] = [512, 1024, 2048]

    def __call__(self, sample):
        if random.random() < self.p:
            win_idx = random.randint(0, len(self.fft_params['win_len']) - 1)
            sgm_noise = self.sgm + 0.01 * torch.rand(1)
            X = torch.stft(sample,
                           win_length=self.fft_params['win_len'][win_idx],
                           hop_length=self.fft_params['hop_len'][win_idx],
                           n_fft=self.fft_params['n_fft'][win_idx],
                           return_complex=True)
            w = sgm_noise * torch.rand_like(X)
            phn = torch.exp(1j * w)
            X.mul_(phn)
            sample = torch.istft(X,
                                 win_length=self.fft_params['win_len'][win_idx],
                                 hop_length=self.fft_params['hop_len'][win_idx],
                                 n_fft=self.fft_params['n_fft'][win_idx])
        return sample


class AudioAugs(object):
    def __init__(self, k_augs, fs, p=0.5, snr_db=30):
        self.noise_vec = ['awgn', 'abgn', 'apgn', 'argn', 'avgn', 'aun', 'phn', 'sine']
        augs = {}
        for aug in k_augs:
            if aug == 'amp':
                augs['amp'] = RandomAmp(p=p, low=0.5, high=1.3)
            elif aug == 'flip':
                augs['flip'] = RandomFlip(p)
            elif aug == 'neg':
                augs['neg'] = RandomAdd180Phase(p)
            elif aug == 'awgn':
                augs['awgn'] = RandomAdditiveWhiteGN(p=p, snr_db=snr_db)
            elif aug == 'abgn':
                augs['abgn'] = RandomAdditiveBlueGN(p=p, snr_db=snr_db)
            elif aug == 'argn':
                augs['argn'] = RandomAdditiveRedGN(p=p, snr_db=snr_db)
            elif aug == 'avgn':
                augs['avgn'] = RandomAdditiveVioletGN(p=p, snr_db=snr_db)
            elif aug == 'apgn':
                augs['apgn'] = RandomAdditivePinkGN(p=p, snr_db=snr_db)
            elif aug == 'sine':
                augs['sine'] = RandomAddSine(p=p, fs=fs)
            elif aug == 'ampsegment':
                augs['ampsegment'] = RandomAmpSegment(p=p, low=0.5, high=1.3, max_len=int(0.1 * fs))
            elif aug == 'aun':
                augs['aun'] = RandomAdditiveUN(p=p, snr_db=snr_db)
            elif aug == 'phn':
                augs['phn'] = RandomPhNoise(p=p, fs=fs, sgm=0.01)
            elif aug == 'fshift':
                augs['fshift'] = RandomFreqShift(fs=fs, sgm=1, p=p)
            else:
                raise ValueError("{} not supported".format(aug))
        self.augs = augs
        self.augs_signal = [a for a in augs if a not in self.noise_vec]
        self.augs_noise = [a for a in augs if a in self.noise_vec]

    def __call__(self, sample, **kwargs):
        augs = self.augs_signal.copy()
        augs_noise = self.augs_noise
        random.shuffle(augs)
        if len(augs_noise) > 0:
            i = random.randint(0, len(augs_noise) - 1)
            augs.append(augs_noise[i])
        for aug in augs:
            sample = self.augs[aug](sample)
        return sample