import numpy
import os
import random
import soundfile
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from tools import *


class BaseDataset(Dataset):
    def __init__(self, train_list=None, train_path=None, num_frames=None, **kwargs):
        self.num_frames = num_frames
        self.train_path = train_path
        self.data_list = []
        self.data_label = []
        print("datasetPath is {}".format(train_path))
        if train_path is None:
            print('train path is none')
        if train_list is None:
            print('create a empty dataset')
        else:
            lines = open(train_list).read().splitlines()
            dictkeys = list(set([x.split()[0] for x in lines]))
            dictkeys.sort()
            dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
            for index, line in enumerate(lines):
                speaker_label = dictkeys[line.split()[0]]
                file_name = os.path.join(train_path,line.split()[1])
                self.data_label.append(speaker_label)
                self.data_list.append(file_name)
            print('speaker number:{}'.format(self.get_speaker_number()))
            print('utterance number:{}'.format(len(self.data_label)))
        self.data_label = torch.tensor(self.data_label, dtype=torch.long)

    def __getitem__(self, index):
        audio, sr = soundfile.read(self.data_list[index])
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)

    class BaseDataset(Dataset):
        def __init__(self, train_list=None, train_path=None, num_frames=None, **kwargs):
            self.num_frames = num_frames
            self.train_path=train_path
            self.data_list = []
            self.data_label = []
            print (train_path)
            if train_list is None:
                print('create a empty dataset')
            else:
                lines = open(train_list).read().splitlines()
                dictkeys = list(set([x.split()[0] for x in lines]))
                dictkeys.sort()
                dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
                for index, line in enumerate(lines):
                    speaker_label = dictkeys[line.split()[0]]
                    file_name = os.path.join(train_path,line.split()[1])
                    self.data_label.append(speaker_label)
                    self.data_list.append(file_name)
                print('speaker number:{}'.format(self.get_speaker_number()))
                print('utterance number:{}'.format(len(self.data_label)))
            self.data_label = torch.tensor(self.data_label, dtype=torch.long)

        def __getitem__(self, index):
            audio, sr = soundfile.read(self.data_list[index])
            length = self.num_frames * 160 + 240
            if audio.shape[0] <= length:
                shortage = length - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
            audio = audio[start_frame:start_frame + length]
            audio = numpy.stack([audio], axis=0)
            return torch.FloatTensor(audio[0]), self.data_label[index]

        def __len__(self):
            return len(self.data_list)
    def get_speaker_number(self):
        return len(numpy.unique(self.data_label))


class Evaluator(object):
    def __init__(self, eval_list, eval_path, **kwargs):
        self.eval_list = eval_list
        self.eval_path = eval_path

    def eval(self, model):
        model.eval()
        files = []
        embeddings = {}
        lines = open(self.eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm(enumerate(setfiles), desc='test', total=len(setfiles), ncols=100):
            audio, _ = soundfile.read(os.path.join(self.eval_path, file))
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

            # Spliited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])
            feats = numpy.stack(feats, axis=0).astype(numpy.float64)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = model.speaker_encoder.forward(data_1, aug=False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = model.speaker_encoder.forward(data_2, aug=False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        scores, labels = [], []

        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF_01, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)
        minDCF_001, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.001, 1, 1)

        return EER, minDCF_01, minDCF_001

