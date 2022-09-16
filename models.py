import torch

import os
import logging
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
# 绘制多分类混淆矩阵
from sklearn.metrics import confusion_matrix
from data_analyze import PCA
import seaborn as sn

import pickle
import LoadData

torch.set_printoptions(threshold=np.inf)
label_classes = 4
dropout = 0.2
batch_size = 16
max_audio_length = 700
max_text_length = 100
epochs = 150
audio_feature_Dimension = 100
audio_Linear = 140
audio_lstm_hidden = 140
text_embedding_Dimension = 100
Bert_text_embedding_Dimension = 768
Word2vec_text_embedding_Dimension=300
Elmo_text_embedding_Dimension = 1024
MFCC_Dimension = 40
Fbank_Dimension = 40
Wav2vec_Dimension=512
IS13_Dimension=6373
IS09_Dimension=384
Hubert_Dimension=1024
text_Linear = 140
text_lstm_hidden = 140
mix_lstm_hidden = 140
gru_hidden = 70
attention_weight_num = 140
attention_head_num = 1
bidirectional = 2  # 2表示双向LSTM,1表示单向

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt

def plot_matrix(matrix):
    labels_order = ['hap', 'sad', 'neu', 'ang']
    # labels_order = ['1', '2', '3', '4', '5']
    # 利用matplot绘图
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


class Early_fusion(nn.Module):
    def __init__(self, modal="text", fusion="AT_fusion"):
        super(Early_fusion, self).__init__()
        self.modal = modal
        self.fusion = fusion
        # 串联concat
        self.Concat2_Linear_Bert = torch.nn.Linear(Bert_text_embedding_Dimension + IS13_Dimension, audio_Linear,
                                                   bias=False)
        self.Concat2_Linear = torch.nn.Linear(text_embedding_Dimension + audio_Linear, audio_Linear,
                                              bias=False)
        # 并联concat
        self.Concat1_Linear = torch.nn.Linear(text_embedding_Dimension, audio_Linear,
                                              bias=False)
        self.Omega_f = torch.normal(mean=torch.full((text_embedding_Dimension, 1), 0.0),
                                    std=torch.full((text_embedding_Dimension, 1), 0.01))
        # 特征直接相加
        # self.ADD= torch.nn.Linear(text_embedding_Dimension, audio_Linear,bias=False)
        self.Audio_Up_Sample=torch.nn.Linear(audio_feature_Dimension, audio_Linear, bias=True)
        self.Audio_Linear = torch.nn.Linear(audio_feature_Dimension, audio_Linear, bias=True)
        self.Text_Linear = torch.nn.Linear(text_embedding_Dimension, text_Linear, bias=True)
        self.Glove_up_Linear_text=torch.nn.Linear(text_embedding_Dimension, 140, bias=True)
        self.Glove_down_Linear_text = torch.nn.Linear(140, text_Linear, bias=True)
        self.Bert_Linear_text = torch.nn.Linear(Bert_text_embedding_Dimension, text_Linear, bias=True)
        self.Word2vec_Linear_text = torch.nn.Linear(Word2vec_text_embedding_Dimension, text_Linear, bias=True)
        self.Elmo_Text_Linear = torch.nn.Linear(Elmo_text_embedding_Dimension, text_Linear, bias=True)
        self.MFCC_Linear = torch.nn.Linear(MFCC_Dimension, audio_Linear, bias=True)
        self.Fbank_Linear = torch.nn.Linear(Fbank_Dimension, audio_Linear, bias=True)
        self.Wav2Vec_Linear=torch.nn.Linear(Wav2vec_Dimension, audio_Linear, bias=True)
        self.IS13_Linear = torch.nn.Linear(IS13_Dimension, audio_Linear, bias=True)
        self.IS09_Linear = torch.nn.Linear(IS09_Dimension, audio_Linear, bias=True)
        self.Hubert = torch.nn.Linear(Hubert_Dimension, audio_Linear, bias=True)
        self.Linear = torch.nn.Linear(attention_weight_num, 100, bias=True)
        self.Linear2 = torch.nn.Linear(gru_hidden*2, 140, bias=True)
        self.Classify_Linear = torch.nn.Linear(100, label_classes, bias=True)
        self.Classify_Linear_audio = torch.nn.Linear(140, label_classes, bias=True)
        self.Classify_Linear_text = torch.nn.Linear(140, label_classes, bias=True)
        self.Softmax = torch.nn.Softmax(dim=2)
        self.GRU = torch.nn.GRU(input_size=audio_Linear, hidden_size=gru_hidden, num_layers=1,
                                bidirectional=True)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.Attention = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num, dropout=0.2, bias=True)
        self.Speaker_Linear=torch.nn.Linear(100,2,bias=True)
        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()

    def forward(self, Audio_Features, Texts_Embedding, Seqlen, Mask):
        '''

        :param Audio_Features: 音频特征，维度待补充
        :param Texts_Embedding: 文本特征，维度待补充
        :param Seqlen: batch的每个序列长度，维度待补充
        :param Mask: mask矩阵，维度待补充
        :return:
        '''
        if self.modal == "text":
            #input = self.Glove_down_Linear_text(torch.relu(self.Glove_up_Linear_text(Texts_Embedding)))
            #input=Texts_Embedding
            input = self.Bert_Linear_text(Texts_Embedding)
            #input = self.Elmo_Text_Linear(Texts_Embedding)
            #input=self.Word2vec_Linear_text(Texts_Embedding.to(torch.float32))
        if self.modal == "audio":
            #input = Audio_Features
            #input= self.Fbank_Linear(Audio_Features.to(torch.float32))
            #input = self.Wav2Vec_Linear(Audio_Features)
            #input = self.relu(self.IS13_Linear(Audio_Features.to(torch.float32)))
            input = self.relu(self.Hubert(Audio_Features.to(torch.float32)))
            #input = self.relu(self.IS09_Linear(Audio_Features.to(torch.float32)))
            #input = self.gelu(self.IS13_Linear(Audio_Features.to(torch.float32)))
            #input = self.MFCC_Linear(Audio_Features)

        if self.modal == "multi":
            # 使用AT_fusion或者ADD需要先将特征仿射到同一个维度
            #tmp = self.Word2vec_Linear_text(Texts_Embedding.to(torch.float32))
            tmp = self.Bert_Linear_text(Texts_Embedding)
            #tmp=Texts_Embedding
            #input = self.Wav2Vec_Linear(Audio_Features)
            #input = self.relu(self.IS13_Linear(Audio_Features.to(torch.float32)))
            input=Audio_Features
            if self.fusion == "AT_fusion":
                # 并联的方式（AT-fusion）
                Concat = torch.cat([input[:, :, None, :], tmp[:, :, None, :]], 2)
                u_cat = self.Concat1_Linear(Concat)

            elif self.fusion == "Concat":
                # 串联
                Concat = torch.cat([Audio_Features[:, :, None, :].to(torch.float32), Texts_Embedding[:, :, None, :]], 3)
                # u_cat = self.Concat2_Linear_Bert(Concat)
                u_cat = self.Concat2_Linear(Concat)
            if self.fusion != "ADD":
                NonLinear = torch.tanh(u_cat)
                alpha_fuse = torch.matmul(NonLinear, self.Omega_f)
                alpha_fuse = alpha_fuse.squeeze(3)
                normalized_alpha = self.Softmax(alpha_fuse)
                input = torch.matmul(u_cat.permute([0, 1, 3, 2]), normalized_alpha[:, :, :, None]).squeeze(dim=3)
            if self.fusion == "ADD":
                # ADD的方式
                Audio_Features=self.Audio_Up_Sample(Audio_Features[:, :, :]).to(torch.float32)
                Audio_output=self.relu(self.Classify_Linear_audio(Audio_Features))
                Texts_Embedding=self.Bert_Linear_text(Texts_Embedding[:, :, :])
                Text_output=self.relu(self.Classify_Linear_text(Texts_Embedding))

                input=Audio_Features+Texts_Embedding
                # input = self.Audio_Up_Sample(Audio_Features[:, :, :]).to(torch.float32) + self.Bert_Linear_text(
                #     Texts_Embedding[:, :, :])


        # 为了batch统一长度后标记原长度。
        #GRU层
        Fusion_Padding = torch.nn.utils.rnn.pack_padded_sequence(input, Seqlen,
                                                                batch_first=True, enforce_sorted=False)
        Fusion_GRU_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU(Fusion_Padding)[0])
        MinMask = Mask[:, :Fusion_GRU_Out.shape[0]]
        Fusion_Contribute = self.dropout(Fusion_GRU_Out*MinMask.permute([1,0])[:,:,None])
        #atttention层
        Fusion_Attention_Out, _ = self.Attention(Fusion_Contribute, Fusion_Contribute, Fusion_Contribute,
                                          key_padding_mask=(1 - MinMask))
        #一层全连接
        Dense1 = torch.tanh(self.Linear((Fusion_Attention_Out).permute([1, 0, 2])))
        Masked_Dense1 = Dense1 * MinMask[:, :, None]

        Dropouted_Dense1 = self.dropout(Masked_Dense1)
        #分类层
        Emotion_Output = self.Classify_Linear((Dropouted_Dense1).to(torch.float32).permute([1, 0, 2]))

        Emotion_Predict = self.Softmax(Emotion_Output)

        if self.fusion=='AT_fusion':
            return Emotion_Predict,normalized_alpha,Dropouted_Dense1,Audio_output,Text_output
        else:
            return Emotion_Predict,False,Dropouted_Dense1,Audio_output,Text_output


def train_and_test_earlyfusion(train_loader, test_loader, model, optimizer, num_epochs, savefile=None):
    Best_Valid = 0

    Loss_Function = nn.CrossEntropyLoss(reduction='none')
    for epoch in range(num_epochs):
        confusion_Ypre = []
        confusion_Ylabel = []
        confusion_TrainYlabel = []
        outputs_total = []
        test_label_total = []
        text_confusion_Ypre = []
        audio_confusion_Ypre = []
        model.train()
        #########计算训练loss以及梯度反传###################
        for i, features in enumerate(train_loader):
            video_train, audio_train, text_train, train_mask, train_label, seqlen_train,_ ,_= features
            train_mask = train_mask.to(torch.int)
            audio_train = audio_train.to(device)
            train_label = train_label.to(device)
            text_train = text_train.to(device)
            #################################模型inference#######################################
            outputs,_,_,text_outputs, audio_outputs= model.forward(audio_train, text_train, seqlen_train, train_mask)

            audio_outputs=audio_outputs.permute([1,0,2])[:outputs.shape[0],:,:]
            text_outputs = text_outputs.permute([1, 0, 2])[:outputs.shape[0], :, :]

            text_outputs = torch.nn.functional.softmax(text_outputs, dim=2)
            audio_outputs = torch.nn.functional.softmax(audio_outputs, dim=2)


            outputs = outputs.permute([1, 2, 0])
            train_label = train_label[:, 0:outputs.shape[2]]
            train_label = train_label.permute([0, 2, 1])
            Loss_Label = torch.argmax(train_label, dim=1)
            optimizer.zero_grad()

            audio_outputs = audio_outputs.permute([1, 2, 0])
            text_outputs = text_outputs.permute([1, 2, 0])
            loss_audio = Loss_Function(audio_outputs, Loss_Label)
            loss_text = Loss_Function(text_outputs, Loss_Label)
            loss = Loss_Function(outputs, Loss_Label)
            ############################选择loss计算方式#######################
            True_loss = (loss+loss_audio+loss_text) * train_mask[:, :loss.shape[1]]
            #True_loss = loss * train_mask[:, :loss.shape[1]]
            total_loss=torch.sum(True_loss, dtype=torch.float)
            total_loss.backward()
            optimizer.step()
            for i in range(Loss_Label.shape[0]):
                confusion_TrainYlabel.extend(Loss_Label[i][:seqlen_train[i]].numpy())

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            text_correct = 0
            audio_correct = 0

            for i, features in enumerate(test_loader):
                video_test, audio_test, text_test, test_mask, test_label, seqlen_test,_,_ = features
                test_mask = test_mask.to(torch.int)
                audio_test = audio_test.to(device)
                test_label = test_label.to(device)
                text_test = text_test.to(device)

                original_outputs,_,_,text_output, audio_output= model.forward(audio_test, text_test, seqlen_test, test_mask)

                text_output = text_output[:,:original_outputs.shape[0],:]
                audio_output = audio_output[:,:original_outputs.shape[0],:]
                _, text_predict = torch.max(text_output, 2)
                _, audio_predict = torch.max(audio_output, 2)


                outputs=original_outputs
                outputs_original = outputs.permute([1, 0, 2])
                _, predict = torch.max(outputs_original, 2)

                test_label_original = test_label[:, :predict.shape[1]]

                test_mask_ = test_mask[:, :predict.shape[1]]
                predict = predict * test_mask_
                text_predict = text_predict * test_mask_
                audio_predict = audio_predict * test_mask_
                test_label = torch.argmax(test_label_original, dim=2)
                test_label = test_label * test_mask_
                total += test_mask_.sum()

                text_correct += ((text_predict == test_label) * test_mask_).sum()
                audio_correct += ((audio_predict == test_label) * test_mask_).sum()
                correct += ((predict == test_label) * test_mask_).sum()

                for i in range(predict.shape[0]):
                    confusion_Ypre.extend(predict[i][:seqlen_test[i]].numpy())
                    text_confusion_Ypre.extend(text_predict[i][:seqlen_test[i]].numpy())
                    audio_confusion_Ypre.extend(audio_predict[i][:seqlen_test[i]].numpy())
                    confusion_Ylabel.extend(test_label[i][:seqlen_test[i]].numpy())
                    outputs_total.extend(outputs_original[:, :seqlen_test[i], :].reshape(-1))
                    test_label_total.extend(test_label_original[:, :seqlen_test[i], :].reshape(-1))

            print(
                'Epoch: %d/%d; total utterance: %d ; correct utterance: %d ; Acc: %.2f%%' % (epoch + 1, num_epochs,
                                                                                             total.item(),
                                                                                             correct.item(), 100 * (
                                                                                                     correct / total).item()))

        if correct / total > Best_Valid:
            Best_Valid = correct / total
            matrix = confusion_matrix(confusion_Ylabel, confusion_Ypre)
            total_num = np.sum(matrix, axis=1)
            acc_matrix = np.round(matrix / total_num[:, None], decimals=4)
            # ###################文本的混淆矩阵#################################
            text_Best_Valid = text_correct / total
            matrix = confusion_matrix(confusion_Ylabel, text_confusion_Ypre)
            total_num = np.sum(matrix, axis=1)
            text_acc_matrix = np.round(matrix / total_num[:, None], decimals=4)
            #######################音频的混淆矩阵########################
            audio_Best_Valid = audio_correct / total
            matrix = confusion_matrix(confusion_Ylabel, audio_confusion_Ypre)
            total_num = np.sum(matrix, axis=1)
            audio_acc_matrix = np.round(matrix / total_num[:, None], decimals=4)
            torch.save(model,"best.pt")

    #plot_matrix(acc_matrix).show()
    dir = "confusion_matrix_try"
    figure = plot_matrix(acc_matrix)
    filename = "%s\\After_fusion_%s_%s.png" % (dir, 'multi', 'Bert')
    figure.savefig(fname=filename, dpi=300)

    print("Best Valid Accuracy: %0.2f%%" % (100 * Best_Valid))
    #print("Best Vaild Speaker Accuracy: %0.2f%%" % (100 * Best_Speaker_Valid))
    #np.savez(savefile, matrix=acc_matrix,AUC_label=test_label_total,AUC_outputs=outputs_total)
    np.savez(savefile, matrix=acc_matrix,ACC=Best_Valid)
