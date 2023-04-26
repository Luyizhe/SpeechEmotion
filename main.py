from models import *
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str, default="tmp.txt", help='output confusion matrix to a file')
    parser.add_argument('--fusion', type=str, default="ADD",
                        help='choose "AT_fusion" "Concat" "ADD" ')
    parser.add_argument('--loss_num', type=str, default='3', help='1loss or 3loss')

    args = parser.parse_args()
    fusion = args.fusion  
    loss_num = args.loss_num 
    
    # train and test dataloader
    batch_data_train = LoadData.LoadDiaData('train')
    train_loader = DataLoader(dataset=batch_data_train, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=4)
    batch_data_test = LoadData.LoadDiaData('test')
    test_loader = DataLoader(dataset=batch_data_test, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=4)
    # torch model
    model = After_fusion(fusion).to(device)

    lr,num_epochs = 1e-3,epochs

    optimizer  = torch.optim.Adam(model.parameters(),lr = lr,weight_decay=1e-5)
    # train and test model
    train_and_test_afterfusion(train_loader, test_loader, model, optimizer, num_epochs,loss_num, args.outfile)
   
