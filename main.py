from models import *
from try_models_advanced import *

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str, default="tmp.txt", help='output confusion matrix to a file')
    parser.add_argument('--classify', type=str, default="emotion", help='choose train vad or emotion')
    parser.add_argument('--modal', type=str, default="multi", help='choose "text","audio","multi"')
    parser.add_argument('--fusion', type=str, default="ADD",
                        help='choose "AT_fusion" "Concat" "ADD" ,or "ADD" "Dot" in try models')
    parser.add_argument('--dataset', type=str, default="ground_truth",
                        help='choose "google_cloud" "speech_recognition" "ground_truth" "resources" or "v" "a" "d"')
    parser.add_argument('--criterion', type=str, default="CrossEntropyLoss", help='choose "MSELoss" "CrossEntropyLoss"')
    parser.add_argument('--loss_delta', type=float, default=1, help='change loss proportion')
    args = parser.parse_args()
    classify = args.classify  # "emotion" "vad"
    modal = args.modal  # "text","audio","multi"
    fusion = args.fusion  # "AT_fusion" "Concat" "ADD"
    dataset = args.dataset  # "google cloud" "speech recognition" "ground truth" "v" "a" "d"
    # matrix_save_file=sys.argv[1]

    batch_data_train = LoadData.LoadDiaData('train', dataset, classify)
    train_loader = DataLoader(dataset=batch_data_train, batch_size=batch_size, drop_last=False, shuffle=True)
    batch_data_test = LoadData.LoadDiaData('test', dataset, classify)
    test_loader = DataLoader(dataset=batch_data_test, batch_size=batch_size, drop_last=False, shuffle=False)

    model = Early_fusion(modal, fusion).to(device)
    #model = After_fusion(fusion).to(device)


    lr,num_epochs = 1e-3,epochs
    #######################设置不同学习率################
    all_params = model.parameters()
    Text_params = []
    Audio_params = []
    Speaker_params=[]
    # 根据自己的筛选规则  将所有网络参数进行分组
    for pname, p in model.named_parameters():
       # print(pname.split('.')[0])zhang
        if pname.split('.')[0].endswith('audio'):
            Audio_params += [p]
        elif pname.split('.')[0].endswith('text'):
            Text_params += [p]
        if pname.split('.')[0].startswith('Speaker'):
            Speaker_params+=[p]
        # 取回分组参数的id
    params_id = list(map(id, Text_params)) + list(map(id, Audio_params))+list(map(id, Speaker_params))
    # 取回剩余分特殊处置参数的id
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))

    #######################构建优化器################
    optimizer  = torch.optim.Adam([
        {'params': other_params,'lr':lr},
        {'params': Audio_params, 'lr': lr},
        {'params': Text_params, 'lr':lr},
        {'params': Speaker_params, 'lr': 0}],
        weight_decay=1e-5
    )

    #############################选择融合方式###########################################
    train_and_test_earlyfusion(train_loader, test_loader, model, optimizer, num_epochs, args.outfile)
    #train_and_test_afterfusion(train_loader, test_loader, model, optimizer, num_epochs, args.loss_delta,args.outfile)