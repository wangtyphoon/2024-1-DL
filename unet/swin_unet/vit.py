# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn

from torch.nn.modules.utils import _pair
from swin_unet.model_48_ver1 import SwinTransformerSys
from torchsummary import summary

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self,  img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.swin_unet = SwinTransformerSys()

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUnet().to(device)
    summary(model, input_size=(10, 48, 48))  # Assuming an input image size of 320x320 with 3 channels
    # import torch
    # import torch.nn as nn
    # import torch.optim as optim
    # from torch.utils.data import DataLoader, TensorDataset

    # # 定义合成数据的参数
    # batch_size = 4
    # num_samples = 100
    # input_channels = 10
    # image_size = 48  # 输入图像大小：40x40
    # n_classes = 1  # 二分类分割任务

    # # 生成合成数据（输入图像和掩码）
    # inputs = torch.rand((num_samples, input_channels, image_size, image_size))  # 随机输入图像
    # masks = (torch.rand((num_samples, n_classes, image_size, image_size)) > 0.5).float()  # 随机二进制掩码

    # # 创建 DataLoader
    # dataset = TensorDataset(inputs, masks)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # # 初始化模型、损失函数和优化器
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SwinUnet(img_size=image_size, num_classes=n_classes).to(device)
    # criterion = nn.BCEWithLogitsLoss()  # 用于二分类的损失函数
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # # 训练循环
    # epochs = 100
    # for epoch in range(epochs):
    #     model.train()
    #     epoch_loss = 0
    #     for batch_inputs, batch_masks in dataloader:
    #         batch_inputs, batch_masks = batch_inputs.to(device), batch_masks.to(device)
            
    #         # 前向传播
    #         outputs = model(batch_inputs)
            
    #         # 计算损失
    #         loss = criterion(outputs, batch_masks)
            
    #         # 反向传播和优化
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #         epoch_loss += loss.item()
        
    #     print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    # # 在单个样本上测试模型
    # model.eval()
    # test_input = torch.rand((1, input_channels, image_size, image_size)).to(device)
    # test_output = model(test_input)
    # print("Test output shape:", test_output.shape)


