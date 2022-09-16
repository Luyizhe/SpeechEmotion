import torch

import os
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from models import plot_matrix
from data_analyze import PCA
# 绘制多分类混淆矩阵
from sklearn.metrics import confusion_matrix

import pickle
import LoadData
from LoadData import label_classes

torch.set_printoptions(threshold=np.inf)
dropout = 0.2
batch_size = 20
max_audio_length = 700
max_text_length = 100
epochs = 150
audio_feature_Dimension = 100
audio_Linear = 100
audio_lstm_hidden = 100
text_embedding_Dimension = 100
Bert_text_embedding_Dimension = 768
Elmo_text_embedding_Dimension = 1024
Word2vec_text_embedding_Dimension = 300
MFCC_Dimension = 40
Fbank_Dimension = 80
Wav2Vec_Dimension = 512
IS13_Dimension = 6373
text_Linear = 100
text_lstm_hidden = 100
mix_lstm_hidden = 100
gru_hidden = 50
attention_weight_num = 100
attention_head_num = 1
trans_attention_head = 4
bidirectional = 2  # 2表示双向LSTM,1表示单向

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(precision=5, suppress=True)
import matplotlib.pyplot as plt

class After_fusion(nn.Module):
    def __init__(self, fusion="ADD"):
        super(After_fusion, self).__init__()
        self.fusion = fusion
        # 串联concat

        self.Concat2_Linear = torch.nn.Linear(2 * label_classes, label_classes,
                                              bias=False)
        # 并联concat
        self.Concat1_Linear = torch.nn.Linear(label_classes, label_classes,
                                              bias=False)
        self.Omega_f = torch.normal(mean=torch.full((label_classes, 1), 0.0),
                                    std=torch.full((label_classes, 1), 0.01))
        # 特征直接相加
        # self.ADD= torch.nn.Linear(text_embedding_Dimension, audio_Linear,bias=False)
        self.Bert_Linear_text = torch.nn.Linear(Bert_text_embedding_Dimension, text_Linear, bias=True)
        self.Elmo_Linear_text = torch.nn.Linear(Elmo_text_embedding_Dimension, text_Linear, bias=True)
        self.Word2vec_Linear_text = torch.nn.Linear(Word2vec_text_embedding_Dimension, text_Linear, bias=True)
        self.MFCC_Linear = torch.nn.Linear(MFCC_Dimension, text_Linear, bias=True)
        self.Wav2Vec_Audio_Linear = torch.nn.Linear(Wav2Vec_Dimension, audio_Linear, bias=True)
        self.IS13_Linear = torch.nn.Linear(IS13_Dimension, audio_Linear, bias=True)
        self.Audio_Linear = torch.nn.Linear(audio_feature_Dimension, audio_Linear, bias=True)
        self.Text_Linear = torch.nn.Linear(text_embedding_Dimension, text_Linear, bias=True)
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
        self.GRU_fusion = torch.nn.GRU(input_size=label_classes, hidden_size=label_classes // 2, num_layers=1,
                                       bidirectional=True)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.Attention_audio = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num, dropout=0.2,
                                                           bias=True)
        self.Attention_audio_logits_weight = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num,
                                                                         dropout=0.2,
                                                                         bias=True)
        self.Linear_audio_logits_weight = torch.nn.Linear(attention_weight_num, label_classes, bias=True)

        self.Attention_text = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num, dropout=0.2,
                                                          bias=True)
        self.Attention_text_logits_weight = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num,
                                                                        dropout=0.2,
                                                                        bias=True)
        self.Linear_text_logits_weight = torch.nn.Linear(attention_weight_num, label_classes, bias=True)
        self.Linear_Concat_to_100 = torch.nn.Linear(gru_hidden * 4, 100, bias=True)
        self.Linear_Concat_to_100_1 = torch.nn.Linear(gru_hidden * 4, 100, bias=True)
        self.Linear_100_to_weight = torch.nn.Linear(100, label_classes, bias=True)
        self.Speaker_Linear = torch.nn.Linear(100, 2, bias=True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, Audio_Features, Texts_Embedding, Seqlen, Mask):
        '''

        :param Audio_Features: 音频特征，维度待补充
        :param Texts_Embedding: 文本特征，维度待补充
        :param Seqlen: batch的每个序列长度，维度待补充
        :param Mask: mask矩阵，维度待补充
        :return: 待定
        '''
        ##########################将音频、文本仿射到同一维度#####################
        # input_text=Texts_Embedding
        input_text = self.Bert_Linear_text(Texts_Embedding)
        # input_text = self.Elmo_Linear_text(Texts_Embedding)
        # input_text = self.Word2vec_Linear_text(Texts_Embedding.to(torch.float32))
        input_audio = Audio_Features
        # input_audio = self.Wav2Vec_Audio_Linear(Audio_Features)
        # input_audio = self.relu(self.IS13_Linear(Audio_Features.to(torch.float32)))
        ################使用audio进行分类训练############################
        # 将音频通过GRU
        Audio_Padding = torch.nn.utils.rnn.pack_padded_sequence(input_audio, Seqlen,
                                                                batch_first=True, enforce_sorted=False)
        # Audio_LSTM_Out、Text_LSTM_Out的维度分别是audio序列长度，batchsize，特征维度。
        Audio_GRU_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU_audio(Audio_Padding)[0])

        Audio_MinMask = Mask[:, :Audio_GRU_Out.shape[0]]
        Audio_Contribute = self.dropout(Audio_GRU_Out * Audio_MinMask.permute([1, 0])[:, :, None])
        # 音频attention
        Audio_Attention_Out, Audio_Attention_Weight = self.Attention_audio(Audio_Contribute, Audio_Contribute,
                                                                           Audio_Contribute,
                                                                           key_padding_mask=(1 - Audio_MinMask),
                                                                           need_weights=True)
        # 音频attention输出通过线性层仿射。
        Audio_Dense1 = torch.tanh(self.Linear_audio(Audio_Attention_Out.permute([1, 0, 2])))
        Audio_Masked_Dense1 = Audio_Dense1 * Audio_MinMask[:, :, None]
        Audio_Dropouted_Dense1 = self.dropout(Audio_Masked_Dense1)
        # 音频预测logits
        Audio_Emotion_Output = self.Classify_Linear_audio(Audio_Dropouted_Dense1.permute([1, 0, 2]))
        ################使用text进行分类训练############################
        # 将文本通过GRU
        Text_Padding = torch.nn.utils.rnn.pack_padded_sequence(input_text, Seqlen,
                                                               batch_first=True, enforce_sorted=False)
        # Audio_LSTM_Out、Text_LSTM_Out的维度分别是audio序列长度，batchsize，特征维度。
        Text_GRU_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU_text(Text_Padding)[0])

        Text_MinMask = Mask[:, :Text_GRU_Out.shape[0]]
        Text_Contribute = self.dropout(Text_GRU_Out * Text_MinMask.permute([1, 0])[:, :, None])
        # 文本attention
        Text_Attention_Out, Text_Attention_Weight = self.Attention_text(Text_Contribute, Text_Contribute,
                                                                        Text_Contribute,
                                                                        key_padding_mask=(1 - Text_MinMask),
                                                                        need_weights=True)

        # 音频attention输出通过线性层仿射。
        Text_Dense1 = torch.tanh(self.Linear_text(Text_Attention_Out.permute([1, 0, 2])))
        Text_Masked_Dense1 = Text_Dense1 * Text_MinMask[:, :, None]
        Text_Dropouted_Dense1 = self.dropout(Text_Masked_Dense1)
        # 文本预测logits
        Text_Emotion_Output = self.Classify_Linear_text(Text_Dropouted_Dense1.permute([1, 0, 2]))

        ##################SOTA之上的一些额外操作：拼接/不拼接两个LSTM输出的特征，再通过attention##########################
        # max_seqlen = max(Seqlen)
        # Concat_Input = torch.concat([input_text[:, :max_seqlen, :], input_audio[:, :max_seqlen, :]], dim=1).permute(
        #     [1, 0, 2])
        # input_text_for_weight = input_text[:, :max_seqlen, :].permute([1, 0, 2])
        # input_audio_for_weight = input_audio[:, :max_seqlen, :].permute([1, 0, 2])
        #
        # Audio_Attention_logits_weight, _ = self.Attention_audio_logits_weight(input_audio_for_weight, Concat_Input,
        #                                                                       Concat_Input,
        #                                                                       key_padding_mask=(1 - Concat_mask))
        #
        # # Audio_Attention_logits_weight = self.Linear_text_logits_weight(Audio_Attention_logits_weight.permute([1, 0, 2]))
        # Text_Attention_logits_weight, _ = self.Attention_text_logits_weight(input_text_for_weight, Concat_Input,
        #                                                                     Concat_Input,
        #                                                                     key_padding_mask=(1 - Concat_mask))
        # # Text_Attention_logits_weight = self.Linear_text_logits_weight(Text_Attention_logits_weight.permute([1, 0, 2]))
        # Concat_attention_out = torch.concat(
        #     [Audio_Attention_logits_weight.permute([1, 0, 2]), Text_Attention_logits_weight.permute([1, 0, 2])], dim=2)
        # Concat_attention_out = self.relu(
        #     self.dropout(self.Linear_Concat_to_100_1(Concat_attention_out) * Text_MinMask[:, :, None]))
        # Sigmoid_weight = self.sigmoid(self.dropout(self.Linear_100_to_weight(Concat_attention_out)))
        ################fusion方式################################
        Text_Emotion_Predict = self.Softmax(Text_Emotion_Output)
        Audio_Emotion_Predict = self.Softmax(Audio_Emotion_Output)
        if self.fusion == "ADD":
            # 将attention输出进行fusion
            fusion_dense = torch.tanh(
                (self.Linear_fusion(Text_Attention_Out.permute([1, 0, 2]) + Audio_Attention_Out.permute([1, 0, 2]))))
            fusion_Dropouted_Dense1 = self.dropout((fusion_dense * Text_MinMask[:, :, None]))
            Emotion_Output = self.Classify_Linear(fusion_Dropouted_Dense1.permute([1, 0, 2]))

            # SOTA的方案
            # Emotion_Output = Audio_Emotion_Output + Text_Emotion_Output

        elif self.fusion == "Dot":
            Emotion_Output = Audio_Emotion_Output * Text_Emotion_Output
        elif self.fusion == "Concat":
            Concat = torch.cat([Audio_Emotion_Output[:, :, :], Text_Emotion_Output[:, :, :]], 2)
            Emotion_Output = self.Concat2_Linear(Concat)
        elif self.fusion == "AT_fusion":
            Concat = torch.cat([Text_Emotion_Output[:, :, None, :], Audio_Emotion_Output[:, :, None, :]], 2)
            u_cat = self.Concat1_Linear(Concat)
            NonLinear = self.dropout(torch.tanh(u_cat))
            alpha_fuse = torch.matmul(NonLinear, self.Omega_f)
            alpha_fuse = alpha_fuse.squeeze(3)
            normalized_alpha = self.Softmax(alpha_fuse)
            Emotion_Output = torch.matmul(u_cat.permute([0, 1, 3, 2]), normalized_alpha[:, :, :, None]).squeeze(dim=3)

        Emotion_Predict = self.Softmax(Emotion_Output)

        return Emotion_Predict, Text_Emotion_Output, Audio_Emotion_Output, Text_Attention_Out, Audio_Attention_Out, False, fusion_Dropouted_Dense1


def train_and_test_afterfusion(train_loader, test_loader, model, optimizer, num_epochs, delta,
                               savefile=None):
    '''

    :param train_loader: 训练数据迭代器
    :param test_loader: 测试数据迭代器
    :param model: torch的模型
    :param optimizer: 优化器
    :param num_epochs: 训练的epoch
    :param delta:
    :param savefile: 保存的文件的名字
    :return: 无
    '''
    Best_Valid = 0
    Best_Speaker_Valid = 0

    Loss_Function = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(num_epochs):
        confusion_Ypre = []
        confusion_Ylabel = []
        confusion_TrainYlabel = []
        text_confusion_Ypre = []
        audio_confusion_Ypre = []
        model.train()
        for i, features in enumerate(train_loader):
            ############################从迭代器中拿数据#####################################
            video_train, audio_train, text_train, train_mask, train_label, seqlen_train, _, speaker_labels = features
            train_mask = train_mask.to(torch.int)
            audio_train = audio_train.to(device)
            train_label = train_label.to(device)
            text_train = text_train.to(device)
            ############################模型inference#########################################
            outputs, text_outputs, audio_outputs, _, _, _, _ = model.forward(audio_train, text_train, seqlen_train,
                                                                             train_mask)

            ##########################计算各个分类头的损失###########################
            text_outputs = torch.nn.functional.softmax(text_outputs, dim=2)
            audio_outputs = torch.nn.functional.softmax(audio_outputs, dim=2)

            train_label = train_label[:, 0:outputs.shape[0]]
            outputs = outputs.permute([1, 2, 0])
            train_label = train_label.permute([0, 2, 1])
            Loss_Label = torch.argmax(train_label, dim=1)
            ###########################   梯度回传   ###############################
            optimizer.zero_grad()
            loss = Loss_Function(outputs, Loss_Label)
            audio_outputs = audio_outputs.permute([1, 2, 0])
            text_outputs = text_outputs.permute([1, 2, 0])
            loss_audio = Loss_Function(audio_outputs, Loss_Label)
            loss_text = Loss_Function(text_outputs, Loss_Label)
            ####################注释选择训练使用哪些loss，delta默认为1#######################
            # total_loss_ = loss + delta * (loss_audio + loss_text)
            total_loss_ = loss

            True_loss = total_loss_ * train_mask[:, :loss.shape[1]]

            total_loss = torch.sum(True_loss, dtype=torch.float)

            total_loss.backward()
            optimizer.step()
            ######################记录绘图数据##################
            for i in range(Loss_Label.shape[0]):
                confusion_TrainYlabel.extend(Loss_Label[i][:seqlen_train[i]].numpy())
        ###############################   模型测试   ###############################
        with torch.no_grad():
            model.eval()
            correct = 0
            text_correct = 0
            audio_correct = 0
            total = 0

            for i, features in enumerate(test_loader):
                ############################从迭代器中拿数据#####################################
                video_test, audio_test, text_test, test_mask, test_label, seqlen_test, _, test_speaker = features
                test_mask = test_mask.to(torch.int)
                audio_test = audio_test.to(device)
                test_label = test_label.to(device)
                text_test = text_test.to(device)
                ############################模型inference#########################################
                original_outputs, text_output, audio_output, _, _, _, _ = model.forward(audio_test, text_test,
                                                                                        seqlen_test,
                                                                                        test_mask)
                ############################计算各个模态预测的准确率#########################
                outputs = original_outputs
                text_outputs = text_output
                audio_outputs = audio_output
                outputs_original = outputs.permute([1, 0, 2])
                text_output = text_outputs.permute([1, 0, 2])
                audio_output = audio_outputs.permute([1, 0, 2])
                _, predict = torch.max(outputs_original, 2)
                _, text_predict = torch.max(text_output, 2)
                _, audio_predict = torch.max(audio_output, 2)
                test_label_original = test_label[:, :predict.shape[1]]

                test_mask = test_mask[:, :predict.shape[1]]
                predict = predict * test_mask
                text_predict = text_predict * test_mask
                audio_predict = audio_predict * test_mask

                test_label = torch.argmax(test_label_original, dim=2)
                test_label = test_label * test_mask
                total += test_mask.sum()

                correct += ((predict == test_label) * test_mask).sum()
                text_correct += ((text_predict == test_label) * test_mask).sum()
                audio_correct += ((audio_predict == test_label) * test_mask).sum()
                #########################记录绘图所需数据###############################
                for i in range(predict.shape[0]):
                    confusion_Ypre.extend(predict[i][:seqlen_test[i]].numpy())
                    text_confusion_Ypre.extend(text_predict[i][:seqlen_test[i]].numpy())
                    audio_confusion_Ypre.extend(audio_predict[i][:seqlen_test[i]].numpy())
                    confusion_Ylabel.extend(test_label[i][:seqlen_test[i]].numpy())

            if correct / total > Best_Valid:
                torch.save(model, 'best.pt')
                ####################总的混淆矩阵##################
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

        print(
            'Epoch: %d/%d; total utterance: %d ; correct utterance: %d ; Acc: %.2f%%; AudioAcc: %.2f%%; TextAcc: %.2f%%' % (
                epoch + 1, num_epochs, total.item(), correct.item(), 100 * (correct / total).item(),
                100 * (audio_correct / total).item(), 100 * (text_correct / total).item()))

    # #####################展示和保存图像####################
    # dir = "confusion_matrix_try"
    # figure = plot_matrix(acc_matrix)
    # filename = "%s\\After_fusion_%s_%s_%f.png" % (dir, 'ADD', 'Bert', delta)
    # figure.savefig(fname=filename, dpi=300)
    #
    # figure = plot_matrix(text_acc_matrix)
    # filename = "%s\\After_fusion__text_%s_%s_%f.png" % (dir, 'ADD', 'Bert', delta)
    # figure.savefig(fname=filename, dpi=300)
    #
    # figure = plot_matrix(audio_acc_matrix)
    # filename = "%s\\After_fusion__audio_%s_%s_%f.png" % (dir, 'ADD', 'Bert', delta)
    # figure.savefig(fname=filename, dpi=300)
    # figure.show()
    # figure.close()
    print("Best Valid Accuracy: %0.2f%%" % (100 * Best_Valid))
    print("Best Text Valid Accuracy: %0.2f%%" % (100 * text_Best_Valid))
    print("Best Audio Valid Accuracy: %0.2f%%" % (100 * audio_Best_Valid))
    if savefile != None:
        np.savez(savefile, matrix=acc_matrix, ACC=Best_Valid, text_matrix=text_acc_matrix, text_ACC=text_Best_Valid,
                 audio_matrix=audio_acc_matrix, audio_ACC=audio_Best_Valid)


