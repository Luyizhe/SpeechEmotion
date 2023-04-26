from sklearn.metrics import confusion_matrix
import torch

import os
import logging
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn

import seaborn as sn

import pickle
import LoadData
from LoadData import label_classes

torch.set_printoptions(threshold=np.inf)
dropout = 0.2
batch_size = 20
epochs = 150
audio_feature_Dimension = 100
audio_Linear = 100
text_embedding_Dimension = 100
Bert_text_embedding_Dimension = 768
text_Linear = 100
gru_hidden = 50
attention_weight_num = 100
attention_head_num = 4
bidirectional = 2  # 2表示双向LSTM,1表示单向

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.set_printoptions(precision=5, suppress=True)
import matplotlib.pyplot as plt


def plot_matrix(matrix):
    '''
    matrix: confusion matrix
    '''
    labels_order = ['hap', 'sad', 'neu', 'ang']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels_order)
    ax.set_yticklabels([''] + labels_order)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    for x in range(len(matrix)):
        for y in range(len(matrix)):
            plt.annotate(matrix[y, x], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    return plt

def weighted_accuracy(list_y_true, list_y_pred):
    '''
    list_y_true: a list of groundtruth labels.
    list_y_pred: a list of labels predicted by the model.
    '''
    assert (len(list_y_true) == len(list_y_pred))

    y_true = np.array(list_y_true)
    y_pred = np.array(list_y_pred)

    w = np.ones(y_true.shape[0])
    for idx, i in enumerate(np.bincount(y_true)):
        w[y_true == idx] = float(1 / i)

    return accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=w)

class After_fusion(nn.Module):
    def __init__(self, fusion="ADD"):
        super(After_fusion, self).__init__()
        self.fusion = fusion
                        
        # 并联concat
        self.Concat_Linear = torch.nn.Linear(label_classes, label_classes,
                                              bias=False)
        self.Omega_f = torch.normal(mean=torch.full((label_classes, 1), 0.0),
                                    std=torch.full((label_classes, 1), 0.01)).to(device)
        # 特征直接相加

        self.Bert_Linear_text = torch.nn.Linear(Bert_text_embedding_Dimension, text_Linear, bias=True)
        self.Linear_audio = torch.nn.Linear(attention_weight_num, 100, bias=True)
        self.Linear_text = torch.nn.Linear(attention_weight_num, 100, bias=True)
        self.Linear_fusion = torch.nn.Linear(attention_weight_num, 100, bias=True)
        self.Classify_Linear_audio = torch.nn.Linear(100, label_classes, bias=True)
        self.Classify_Linear_text = torch.nn.Linear(100, label_classes, bias=True)
        self.Classify_Linear = torch.nn.Linear(100, label_classes, bias=True)
        self.Softmax = torch.nn.Softmax(dim=2)
        # self.LN=torch.nn.functional.layer_norm([batch_size,])
        self.GRU_audio = torch.nn.GRU(input_size=audio_Linear, hidden_size=gru_hidden, num_layers=1,
                                      bidirectional=True)
        self.GRU_text = torch.nn.GRU(input_size=audio_Linear, hidden_size=gru_hidden, num_layers=1,
                                     bidirectional=True)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.Attention_audio = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num, dropout=0.2,
                                                           bias=True)
        self.Attention_text = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num, dropout=0.2,
                                                          bias=True)



    def forward(self, Audio_Features, Texts_Features, Seqlen, Mask):
        '''
        Audio_Features: 100 dimensions audio feature which affined from IS13.
                        shape is (B, L, D)
        Texts_Features: 768 dimensions text feature    
                        shape is (B, L, D)    
        Seqlen:         Every sample length in the batch.   
                        shape is (B)
        Mask:           Sample mask. '1' indicates that the corresponding position is allowed to attend.
                        shape is (B, L)
        Where is :
            B: Batch size
            L: Dialogue lengths
            D: Feature dimensions
        '''
        # Convert text to 100 dimensions
        input_text = self.Bert_Linear_text(Texts_Features)
        input_audio = Audio_Features

        # audio flow
        Audio_Padding = torch.nn.utils.rnn.pack_padded_sequence(input_audio, Seqlen,
                                                                batch_first=True, enforce_sorted=False)
            # Audio_GRU_Out shape is (L, B, D)
        Audio_GRU_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU_audio(Audio_Padding)[0])      
        Audio_MinMask = Mask[:, :Audio_GRU_Out.shape[0]]
        Audio_GRU_Out = self.dropout(Audio_GRU_Out)
            # Audio_Attention_Out shape is (L, B, D)
        Audio_Attention_Out, Audio_Attention_Weight = self.Attention_audio(Audio_GRU_Out, Audio_GRU_Out,
                                                                          Audio_GRU_Out,
                                                                          key_padding_mask=(~Audio_MinMask.to(torch.bool)),
                                                                          need_weights=True)                                                            
        Audio_Dense1 = torch.relu(self.Linear_audio(Audio_Attention_Out.permute([1, 0, 2])))
            # Audio_Dropouted_Dense1 shape is (B, L, D) 
        Audio_Dropouted_Dense1 = self.dropout(Audio_Dense1 * Audio_MinMask[:, :, None])      
            # Audio_Emotion_Output shape is (L, B, D)
        Audio_Emotion_Output = self.Classify_Linear_audio(Audio_Dropouted_Dense1.permute([1, 0, 2]))

        # text flow 
        Text_Padding = torch.nn.utils.rnn.pack_padded_sequence(input_text, Seqlen,
                                                               batch_first=True, enforce_sorted=False)
            # Text_GRU_Out shape is (L, B, D)
        Text_GRU_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU_text(Text_Padding)[0])
        Text_MinMask = Mask[:, :Text_GRU_Out.shape[0]]
        Text_GRU_Out = self.dropout(Text_GRU_Out)
            # Text_Attention_Out shape is (L, B, D)
        Text_Attention_Out, Text_Attention_Weight = self.Attention_text(Text_GRU_Out, Text_GRU_Out,
                                                                       Text_GRU_Out,
                                                                       key_padding_mask=(~Text_MinMask.to(torch.bool)),
                                                                       need_weights=True)
        Text_Dense1 = torch.relu(self.Linear_text(Text_Attention_Out.permute([1, 0, 2])))
            # Text_Dropouted_Dense1 shape is (B, L, D)
        Text_Dropouted_Dense1 = self.dropout(Text_Dense1 * Text_MinMask[:, :, None])
            # Text_Emotion_Output shape is (L, B, D)
        Text_Emotion_Output = self.Classify_Linear_text(Text_Dropouted_Dense1.permute([1, 0, 2]))

        # different fusion methods 
        # Emotion_Output shape is (L, B, D)
        if self.fusion == "ADD":
            Emotion_Output = Audio_Emotion_Output + Text_Emotion_Output
        elif self.fusion == "Concat":
            # Concat shape is (L, B, 2xD)
            Concat = torch.cat([Audio_Emotion_Output[:, :, :], Text_Emotion_Output[:, :, :]], 2).permute([1, 0, 2])
            Emotion_Output = self.Concat2_Linear(Concat).permute([1,0,2])
        elif self.fusion == "AT_fusion":
            # Concat shape is (L, B, 2, D)
            Concat = torch.cat([Text_Emotion_Output[:, :, None, :], Audio_Emotion_Output[:, :, None, :]], 2)
            u_cat = self.Concat_Linear(Concat)
            NonLinear = self.dropout(torch.relu(u_cat))
            alpha_fuse = torch.matmul(NonLinear, self.Omega_f).squeeze(3)
            normalized_alpha = self.Softmax(alpha_fuse)
            Emotion_Output = torch.matmul(u_cat.permute([0, 1, 3, 2]), normalized_alpha[:, :, :, None]).squeeze(dim=3)
        # Emotion_Predict shape is (L, B, D)
        Emotion_Predict = self.Softmax(Emotion_Output)

        return Emotion_Predict, Text_Emotion_Output, Audio_Emotion_Output

def train_and_test_afterfusion(train_loader, test_loader, model, optimizer, num_epochs, loss_num,
                               savefile=None):
    '''
    train_loader:   Torch dataloader in trainset.
    test_loader:    Torch dataloader in testset.
    model:          Torch model.
    optimizer:      Optimizer for training model.
    num_epochs:     Train epochs.
    loss_num:       '1' means the single loss and 3 indicates the Perspective Loss Function.
    savefile:       Where to save confusion matrix and WA.
    '''
    Best_WA = 0
    Loss_Function = nn.CrossEntropyLoss(reduction='none').to(device)

    for epoch in range(num_epochs):
        confusion_Ypre = []
        confusion_Ylabel = []
        confusion_TrainYlabel = []
        text_confusion_Ypre = []
        audio_confusion_Ypre = []
        model.train()
        # train model
        for i, features in enumerate(train_loader):
            audio_train, text_train, train_mask, train_label, seqlen_train= features
            train_mask = train_mask.to(torch.int).to(device)
            audio_train = audio_train.to(device)
            train_label = train_label.to(device)
            text_train = text_train.to(device)
            # inference
            outputs, text_outputs, audio_outputs = model.forward(audio_train, text_train, seqlen_train,
                                                                       train_mask)

            text_outputs = torch.nn.functional.softmax(text_outputs, dim=2)
            audio_outputs = torch.nn.functional.softmax(audio_outputs, dim=2)

            train_label = train_label[:, 0:outputs.shape[0]]
            outputs = outputs.permute([1, 2, 0])
            train_label = train_label.permute([0, 2, 1])
            Loss_Label = torch.argmax(train_label, dim=1)

            # calculate loss and update model parameters
            optimizer.zero_grad()

            loss = Loss_Function(outputs, Loss_Label)
            audio_outputs = audio_outputs.permute([1, 2, 0])
            text_outputs = text_outputs.permute([1, 2, 0])
            loss_audio = Loss_Function(audio_outputs, Loss_Label)
            loss_text = Loss_Function(text_outputs, Loss_Label)
            
            if loss_num == '1':
                # Single Loss Function
                total_loss_ = loss
            elif loss_num == '3':
                # Perspective Loss Function
                total_loss_ = loss + loss_audio + loss_text
            True_loss = total_loss_ * train_mask[:, :loss.shape[1]]
            total_loss = torch.sum(True_loss, dtype=torch.float)
            total_loss.backward()
            optimizer.step()
                
        # test model
        with torch.no_grad():
            model.eval()
            correct = 0
            text_correct = 0
            audio_correct = 0
            total = 0
            text_scale=0
            audio_scale=0
            for i, features in enumerate(test_loader):
                
                audio_test, text_test, test_mask, test_label, seqlen_test = features
                test_mask = test_mask.to(torch.int).to(device)
                audio_test = audio_test.to(device)
                test_label = test_label.to(device)
                text_test = text_test.to(device)
                # inference
                outputs,text_outputs, audio_outputs= model.forward(audio_test, text_test, seqlen_test,
                                                                                  test_mask)

                # (predict shape is (B, L)
                predict = torch.max(outputs, 2)[1].permute([1, 0])
                text_predict = torch.max(text_outputs, 2)[1].permute([1, 0])
                audio_predict = torch.max(audio_outputs, 2)[1].permute([1, 0])
                test_label = test_label[:, :predict.shape[1]]

                # different sample has various length, so we need to mask redundant position
                test_mask = test_mask[:, :predict.shape[1]]
                predict = predict * test_mask
                text_predict = text_predict * test_mask
                audio_predict = audio_predict * test_mask
                test_label = torch.argmax(test_label, dim=2)
                test_label = test_label * test_mask
                
                # count numbers
                total += test_mask.sum()
                correct += ((predict == test_label) * test_mask).sum()

                # record confusion matrix
                for i in range(predict.shape[0]):
                    confusion_Ypre.extend(predict[i][:seqlen_test[i]].cpu().numpy())
                    text_confusion_Ypre.extend(text_predict[i][:seqlen_test[i]].cpu().numpy())
                    audio_confusion_Ypre.extend(audio_predict[i][:seqlen_test[i]].cpu().numpy())
                    confusion_Ylabel.extend(test_label[i][:seqlen_test[i]].cpu().numpy())
                    
            WA=weighted_accuracy(confusion_Ylabel,confusion_Ypre)
            text_WA=weighted_accuracy(confusion_Ylabel,text_confusion_Ypre)
            audio_WA=weighted_accuracy(confusion_Ylabel,audio_confusion_Ypre)
            if WA > Best_WA:
                # fusion confusion matrix
                Best_WA = WA
                matrix = confusion_matrix(confusion_Ylabel, confusion_Ypre)
                total_num = np.sum(matrix, axis=1)
                acc_matrix = np.round(matrix / total_num[:, None], decimals=4)
                # text confusion matrix
                text_Best_WA = text_WA
                matrix = confusion_matrix(confusion_Ylabel, text_confusion_Ypre)
                total_num = np.sum(matrix, axis=1)
                text_acc_matrix = np.round(matrix / total_num[:, None], decimals=4)
                # audio confusion matrix
                audio_Best_WA = audio_WA
                matrix = confusion_matrix(confusion_Ylabel, audio_confusion_Ypre)
                total_num = np.sum(matrix, axis=1)
                audio_acc_matrix = np.round(matrix / total_num[:, None], decimals=4)
                torch.save(model, "best.pt")
        print(
            'Epoch: %d/%d; total utterance: %d ; correct utterance: %d ; WA: %.2f%%; AudioWA: %.2f%%; TextWA: %.2f%%' % (
            epoch + 1, num_epochs, total.item(), correct.item(), WA*100, 100 * audio_WA, 100 * text_WA))

    print("Best Valid WA: %0.2f%%" % (100 * Best_WA))
    print("Best Text Valid WA: %0.2f%%" % (100 * text_WA))
    print("Best Audio Valid WA: %0.2f%%" % (100 * audio_WA))
    print(acc_matrix)
    print(audio_acc_matrix)
    print(text_acc_matrix)
    if savefile != None:
        np.savez(savefile, matrix=acc_matrix, ACC=Best_WA, text_matrix=text_acc_matrix, text_ACC=text_WA,
                 audio_matrix=audio_acc_matrix, audio_ACC=audio_WA)
