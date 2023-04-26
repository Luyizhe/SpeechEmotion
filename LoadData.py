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
label_classes =4 

class LoadDiaData(Dataset):
    def __init__(self, train_or_test):
        # features file
        file=open('IEMOCAP_features_BertText_4Class.pkl','rb')
        # load data
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, self.videoAudio, self.videoVisual, \
        self.videoSentence, self.trainVid, self.testVid = pickle.load(file, encoding='latin1')
        
        self.indexes = np.arange(len(self.videoIDs))
        self.trainVid = list(self.trainVid)
        self.testVid = list(self.testVid)
        self.text_audio_max = 0
        self.train_or_test = train_or_test
        # get max dialogue length for padding
        for vid in self.trainVid+self.testVid:
            if len(self.videoText[vid]) > self.text_audio_max:
                self.text_audio_max = len(self.videoText[vid])

    def __getitem__(self, batch_index):

        indexes = self.indexes[batch_index]

        if self.train_or_test == 'train':
            vid = self.trainVid[indexes]
        if self.train_or_test == 'test':
            vid = self.testVid[indexes]
            
        # pad audio features
        tmp = np.array(self.videoAudio[vid]).reshape(
            [np.shape(self.videoAudio[vid])[0], np.shape(self.videoAudio[vid])[1], 1])
        audio_len = len(self.videoAudio[vid])
        gap = self.text_audio_max - audio_len
        audio_feat = np.pad(tmp, [(0, gap), (0, 0), (0, 0)], mode='constant')
        audio = [torch.tensor(audio_feat[:, :, 0]), torch.tensor(audio_len)]

        # pad text features
        self.videoText[vid]=np.array(self.videoText[vid])

        tmp = np.array(self.videoText[vid]).reshape(
            [np.shape(self.videoText[vid])[0], np.shape(self.videoText[vid])[1], 1])
        text_len = len(self.videoText[vid])
        gap = self.text_audio_max - text_len
        text_feat = np.pad(tmp, [(0, gap), (0, 0), (0, 0)], mode='constant')
        text = [torch.tensor(text_feat[:, :, 0]), torch.tensor(text_len)]
        
        # pad labels
        tmp = np.array(self.videoLabels[vid]).reshape(
            [np.shape(self.videoLabels[vid])[0], 1])
        labels = np.pad(tmp, [(0, gap), (0, 0)], mode='constant', constant_values=(0, 0))
        labels = torch.LongTensor(labels)
        mask = np.zeros(np.shape(audio[0])[0])
        mask[:text[1]] = 1
        labels=functional.one_hot(labels, num_classes=label_classes).squeeze(1)
        return  audio[0], text[0], mask, labels, text[1]

    def __len__(self):
        if self.train_or_test == 'train':
            return len(self.trainVid)
        if self.train_or_test == 'test':
            return len(self.testVid)

