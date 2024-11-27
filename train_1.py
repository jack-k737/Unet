import time

#from pytorch_metric_learning import losses

import numpy as np

import utils.Data_Loader as Data_Loader
import utils.Models as Models
import utils.losses as losses
import os, torch, torchvision, glob, natsort
import torch.utils.data as data
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, StepLR, ReduceLROnPlateau
import matplotlib.pyplot as plt

from utils.myKits import Accumulator, create_dir, create_dir1

if __name__ == '__main__':
    ###############################################
    # 训练数据集 + 验证数据集
    ###############################################
    Train_data_folder = os.path.join(os.getcwd(), 'ODOC', 'Domain1', 'train', '')

    Training_Data = Data_Loader.Images_Dataset_folder(Train_data_folder)

    ###############################################
    # 参数
    ###############################################
    epochs = 300
    batch_size = 4
    lr = 0.001
    ce_weight = 0.13
    ###############################################
    # validation dataset sample & train dataset sample
    ###############################################
    val_split = int(0.3 * len(Training_Data))
    data_idx = list(range(len(Training_Data)))
    valid_idx, train_idx = data_idx[:val_split], data_idx[val_split:]
    valid_sampler = data.sampler.SubsetRandomSampler(valid_idx)
    train_sampler = data.sampler.SubsetRandomSampler(train_idx)

    train_loader = data.DataLoader(Training_Data,batch_size=batch_size, pin_memory=True, sampler=train_sampler)
    valid_loader = data.DataLoader(Training_Data, batch_size=batch_size, pin_memory=True, sampler=valid_sampler)

    ###############################################
    # 模型
    ###############################################
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Models.UNet(3,3)
    model.to(device)
    opt = Adam(model.parameters(), lr=lr)#, weight_decay=1e-4)
    #opt = SGD(model.parameters(), lr=lr, momentum=0.9)
    loss = losses.calc_loss
    scheduler = CosineAnnealingLR(opt, int(epochs * 1.2), eta_min=1e-5)
    #scheduler = ReduceLROnPlateau(opt, 'min', patience=5)
    #scheduler = StepLR(opt, int(epochs*0.3), gamma=0.5)
    #scheduler = CyclicLR(opt, 1e-5, 1e-2, step_size_up=100, step_size_down=100, mode='triangular')
    ###############################################
    # 创建保存路径
    ###############################################
    New_folder = './model'
    read_model_path = './model/Unet_D_' + str(epochs) + '_' + str(batch_size)

    create_dir1(New_folder)
    create_dir(read_model_path)

    ###############################################
    # 模型训练+验证
    ###############################################
    val_loss_min = np.inf
    acc = Accumulator(2)

    train_loss_history = []
    val_loss_history = []  # 记录验证损失

    for epoch in range(epochs):
        since = time.time()
        acc.reset()
        model.train()
        #######################################
        #训练
        #######################################
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            y_hat = model(x)
            ce_lossV, dice_lossV = loss(y_hat, y)
            lossT = dice_lossV*(1 - ce_weight)+ ce_lossV * ce_weight
            lossT.backward()
            opt.step()
            acc.add(lossT.item() * x.shape[0], x.shape[0])

        scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        train_loss = acc[0] / acc[1]
        train_loss_history.append(train_loss)
        acc.reset()
        model.eval()
        ######################################################
        #测试
        ######################################################
        with torch.no_grad():
            for x1, y1 in valid_loader:
                x1, y1 = x1.to(device), y1.to(device)
                y_hat1 = model(x1)
                ce_lossV, dice_lossV = loss(y_hat1, y1)
                lossV = dice_lossV
                acc.add(lossV.item() * x1.shape[0], x1.shape[0])

            val_loss = acc[0] / acc[1]

            val_loss_history.append(val_loss)  # 记录每个 epoch 的验证损失

            print(f'Epoch: {epoch + 1}/{epochs}, time consumed: {time.time() - since:.1f}s, Training loss: {train_loss:.6f}, Validation dice_loss: {val_loss:.6f}')
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                print(f'Validatcion loss decreased ({val_loss_min:.6f} --> {val_loss:.6f}). Saving model')
                torch.save(model.state_dict(), read_model_path + '/Unet_epoch_' + str(epochs) + '_batchsize_' + str(batch_size) + '.pth')


    ###############################################
    # 绘制验证损失图
    ###############################################
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), val_loss_history, label='Validation Loss', color='blue', marker='.')
    plt.plot(range(1, epochs + 1), train_loss_history, label='Training Loss', color='red', marker='.')
    plt.title('Validation Loss & Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

