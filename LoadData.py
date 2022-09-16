import torch

import os
import logging
import numpy as np
import re
from torch.utils.data import Dataset
import copy
import pickle

from torch.nn import functional
import io
label_classes = 4
segment = {'hap': 0, 'sad': 0, 'neu': 0, 'ang': 0}
speakersegment={'M':0,'F':1}
class Loadtest(Dataset):
    def __init__(self):
        self.len = 1

    def __getitem__(self, item):
        return torch.ones(3, 3)


class LoadDiaData(Dataset):
    def __init__(self, train_or_test,dataset="ground_truth",classify='emotion'):
        if classify=="vad":
            if dataset == "v":
                file=open("IEMOCAP_features_BertText_4ClassGT_v.pkl", 'rb')
                #file = open("IEMOCAP_features_BertText4_ASR_Google_v.pkl", 'rb')
                #file = open("IEMOCAP_features_BertText4_ASR_v.pkl", 'rb')
            if dataset == "a":
                file=open("IEMOCAP_features_BertText_4ClassGT_a.pkl", 'rb')
                #file = open("IEMOCAP_features_BertText4_ASR_Google_a.pkl", 'rb')
                #file = open("IEMOCAP_features_BertText4_ASR_a.pkl", 'rb')
            if dataset == "d":
                file = open("IEMOCAP_features_BertText_4ClassGT_d.pkl", 'rb')
                #file = open("IEMOCAP_features_BertText4_ASR_Google_d.pkl", 'rb')
                #file = open("IEMOCAP_features_BertText4_ASR_d.pkl", 'rb')
        if classify=="emotion":
            if dataset=="speech_recognition":
                file = open('IEMOCAP_features_BertText4_ASR.pkl', 'rb')
            elif dataset=="google_cloud":
                file = open('IEMOCAP_features_BertText4_ASR_Google.pkl', 'rb')
            elif dataset=="ground_truth":
                #file = open('IEMOCAP_features_Wav2VecBertText4_Class.pkl', 'rb')
                #file = open('IEMOCAP_features_Wav2VecTimeBertText4_Class.pkl', 'rb')
                #file = open('IEMOCAP_features_Wav2VecNoiseBertText4_Class.pkl', 'rb')
                #file = open('IEMOCAP_features_FbankTimeBertText4_Class.pkl', 'rb')
                #file = open('IEMOCAP_features_Fbank25BertText4_Class.pkl', 'rb')
                #file = open('IEMOCAP_features_BARTText4_Class.pkl', 'rb')
                #file = open('IEMOCAP_features_RobertaText4_Class.pkl','rb')
                #file = open('IEMOCAP_features_GPT2Text4_Class.pkl', 'rb')
                #file = open('IEMOCAP_features_XLNetText4_Class.pkl', 'rb')
                file = open('IEMOCAP_features_BertText_4Class.pkl', 'rb')
                #file = open('IEMOCAP_features_HubertBertText4_Class.pkl', 'rb')
                #file = open('Choose_modal_traindata.pkl', 'rb')
                #file = open('Choose_modal_traindata_origininput.pkl', 'rb')
                #file = open('IEMOCAP_features_GloVemean_4Class.pkl','rb')
                #file = open('IEMOCAP_features_4Cate.pkl','rb')
                #file = open("IEMOCAP_features_Word2vecmean_4Class.pkl",'rb')
                #file = open('IEMOCAP_features_IS13BertText4_Class.pkl', 'rb')
                #file = open('IEMOCAP_features_IS09BertText4_Class.pkl', 'rb')
                #file = open('IEMOCAP_features_100DimBertText4_Class.pkl', 'rb')
            elif dataset == "Bert":
                file = open('IEMOCAP_features_BertText_4Class.pkl', 'rb')
            elif dataset == "gpt2":
                file = open('IEMOCAP_features_GPT2Text4_Class.pkl', 'rb')
            elif dataset == "BART":
                file = open('IEMOCAP_features_BARTText4_Class.pkl', 'rb')
            elif dataset == "Roberta":
                file = open('IEMOCAP_features_RobertaText4_Class.pkl', 'rb')
            elif dataset == "XLNet":
                file = open('IEMOCAP_features_XLNetText4_Class.pkl', 'rb')
            elif dataset == "Albert":
                file = open('IEMOCAP_features_AlbertText4_Class.pkl', 'rb')
            elif dataset == "Ernie2":
                file = open('IEMOCAP_features_Ernie2Text4_Class.pkl', 'rb')
            elif dataset == "Word2Vec":
                #file = open('IEMOCAP_features_4Cate.pkl', 'rb')
                file = open("IEMOCAP_features_Word2vecmean_4Class.pkl", 'rb')
            elif dataset == "Elmo":
                file = open('IEMOCAP_features_ElmoWhiteText_4Class.pkl', 'rb')
            elif dataset=="resources":
                file = open("IEMOCAP_features_4Cate.pkl", 'rb')
        self.filename=file.name
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, self.videoAudio, self.videoVisual, \
        self.videoSentence, self.trainVid, self.testVid = pickle.load(file, encoding='latin1')
        self.indexes = np.arange(len(self.videoIDs))
        self.trainVid = list(self.trainVid)
        self.testVid = list(self.testVid)
        self.text_audio_max = 0
        self.train_or_test = train_or_test
        for vid in self.trainVid:
            if len(self.videoText[vid]) > self.text_audio_max:
                self.text_audio_max = len(self.videoText[vid])

    def __getitem__(self, batch_index):

        indexes = self.indexes[batch_index]
        # 处理返回各种特征值

        if self.train_or_test == 'train':
            vid = self.trainVid[indexes]
        if self.train_or_test == 'test':
            self.testVid.sort()
            vid = self.testVid[indexes]

        tmp = np.array(self.videoAudio[vid]).reshape(
            [np.shape(self.videoAudio[vid])[0], np.shape(self.videoAudio[vid])[1], 1])
        # 将音频特征处理为统一长度方便放入batch。
        audio_len = len(self.videoAudio[vid])
        gap = self.text_audio_max - audio_len
        audio_feat = np.pad(tmp, [(0, gap), (0, 0), (0, 0)], mode='constant')
        audio = [torch.tensor(audio_feat[:, :, 0]), torch.tensor(audio_len)]

        # 将文本特征处理为统一长度方便放入batch。

        self.videoText[vid]=np.array(self.videoText[vid])
        if len(np.shape(self.videoText[vid]))!=2:
            self.videoText[vid]=self.videoText[vid].squeeze()

        tmp = np.array(self.videoText[vid]).reshape(
            [np.shape(self.videoText[vid])[0], np.shape(self.videoText[vid])[1], 1])
        text_len = len(self.videoText[vid])
        gap = self.text_audio_max - text_len
        text_feat = np.pad(tmp, [(0, gap), (0, 0), (0, 0)], mode='constant')
        text = [torch.tensor(text_feat[:, :, 0]), torch.tensor(text_len)]
        # 将label处理为统一长度方便放入batch。
        tmp = np.array(self.videoLabels[vid]).reshape(
            [np.shape(self.videoLabels[vid])[0], 1])
        speaker=[]
        for speakertmp in self.videoSpeakers[vid]:
            if speakertmp == 'M':
                speaker.append(1)
            else:
                speaker.append(0)
        speaker = np.array(speaker).reshape(
            [np.shape(speaker)[0], 1])
        labels = np.pad(tmp, [(0, gap), (0, 0)], mode='constant', constant_values=(3, 3))
        labels = torch.LongTensor(labels)
        speaker_labels = np.pad(speaker, [(0, gap), (0, 0)], mode='constant', constant_values=(1, 1))
        speaker_labels = torch.LongTensor(speaker_labels)
        mask = np.zeros(np.shape(audio[0])[0])
        mask[:text[1]] = 1

        labels=functional.one_hot(labels, num_classes=label_classes).squeeze(1)
        return self.videoVisual, audio[0], text[0], mask, labels, text[1],vid,speaker_labels#多一个vid用于提取音频特征，需要删除
        # return audio[0], audio[0], text[0], mask, labels, text[1]

    def __len__(self):
        if self.train_or_test == 'train':
            return len(self.trainVid)
        if self.train_or_test == 'test':
            return len(self.testVid)



if __name__ == '__main__':
    test = Loadtest()
