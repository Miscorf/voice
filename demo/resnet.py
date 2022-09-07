import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
batch_size  = 50
class BasicBlock(nn.Module):
    channel_expansion = 1  # {扩展后的最终输出通道数} / {扩展前的输出通道数（blk_mid_channels）}
    
    def __init__(self, blk_in_channels, blk_mid_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=blk_in_channels,  # blk_in_channels：block 中第一个 conv 层的输入通道数
                               out_channels=blk_mid_channels,  # blk_mid_channels：block 中第一个 conv 层的输出通道数
                               kernel_size=3,
                               padding=1,
                               stride=stride)  # stride 可以任意指定
        self.bn1 = nn.BatchNorm2d(blk_mid_channels)
        
        self.conv2 = nn.Conv2d(in_channels=blk_mid_channels,  # block 中第二个 conv 层的输入通道数
                               out_channels=blk_mid_channels*self.channel_expansion,  # 扩展后的最终输出通道数
                               kernel_size=3, 
                               padding=1, 
                               stride=1)  # stride 恒为 1
        self.bn2 = nn.BatchNorm2d(blk_mid_channels*self.channel_expansion)
        
        # 实现 shortcut connection：
        # 假如 block 的输入 x 和 conv2/bn2 的输出形状相同：直接相加
        # 假如 block 的输入 x 和 conv2/bn2 的输出形状不同：在 shortcut connection 上增加一次对 x 的 conv/bn 变换
        if stride != 1 or blk_in_channels != self.channel_expansion*blk_mid_channels:  # 形状不同
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=blk_in_channels,
                          out_channels=self.channel_expansion*blk_mid_channels,  # 变换通道数
                          kernel_size=1,
                          padding=0,
                          stride=stride),  # 变换空间维度
                nn.BatchNorm2d(self.channel_expansion*blk_mid_channels)
            )
        else:  # 形状相同
            self.shortcut = nn.Sequential()
            
        
    def forward(self, t):
        
        # conv1
        out = self.conv1(t)
        out = self.bn1(out)
        out = F.relu(out) 
        
        ################### Please finish the following code ###################
        
        # conv2 & shortcut        
        out = self.conv2(out)
        out = self.bn2(out)
        out = out+self.shortcut(t)
        out = F.relu(out)
        
        ########################################################################
        
        return out

class BottleneckBlock(nn.Module):
    channel_expansion = 4  # {扩展后的最终输出通道数} / {扩展前的输出通道数（blk_mid_channels）}
    
    def __init__(self, blk_in_channels, blk_mid_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=blk_in_channels,  # blk_in_channels：block 中第一个 conv 层的输入通道数
                               out_channels=blk_mid_channels,  # blk_mid_channels：block 中第一个 conv 层的输出通道数
                               kernel_size=1,
                               padding=0,
                               stride=1)  # stride 恒为 1
        self.bn1 = nn.BatchNorm2d(blk_mid_channels)
        
        self.conv2 = nn.Conv2d(in_channels=blk_mid_channels,  # block 中第二个 conv 层的输入通道数
                               out_channels=blk_mid_channels,  # block 中第二个 conv 层的输出通道数
                               kernel_size=3,
                               padding=1,
                               stride=stride)  # stride 可以任意指定
        self.bn2 = nn.BatchNorm2d(blk_mid_channels)
        
        self.conv3 = nn.Conv2d(in_channels=blk_mid_channels,  # block 中第三个 conv 层的输入通道数
                               out_channels=blk_mid_channels*self.channel_expansion,  # 扩展后的最终输出通道数
                               kernel_size=1,
                               padding=0,
                               stride=1)  # stride 恒为 1
        self.bn3 = nn.BatchNorm2d(blk_mid_channels*self.channel_expansion)
        
        # 实现 shortcut connection：
        # 假如 block 的输入 x 和 conv3/bn3 的输出形状相同：直接相加
        # 假如 block 的输入 x 和 conv3/bn3 的输出形状不同：在 shortcut connection 上增加一次对 x 的 conv/bn 变换
        if stride != 1 or blk_in_channels != blk_mid_channels*self.channel_expansion:  # 形状不同
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=blk_in_channels,
                          out_channels=blk_mid_channels*self.channel_expansion,  # 变换空间维度
                          kernel_size=1,
                          padding=0,
                          stride=stride),  # 变换空间维度
                nn.BatchNorm2d(blk_mid_channels*self.channel_expansion)
            )
        else:  # 形状相同
            self.shortcut = nn.Sequential()
            
        
    def forward(self, t):
        
        ################### Please finish the following code ###################

        # conv1
        out = self.conv1(t)
        out = self.bn1(out)
        out = F.relu(out)
        
        # conv2
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        # conv3 & shortcut
        out = self.conv3(out)
        out = self.bn3(out)
        out = out+self.shortcut(t)
        out = F.relu(out)
        
        ########################################################################
        
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        
        self.residual_layers = 4  # 每个 "residual layer" 含多个 blocks，对应上面列表中的一行 (即 conv2_x, conv3_x, conv4_x 或 conv5_x)
        self.blk1_in_channels = 32  # 按照上面的列表，此处应填 64，但由于大网络训练起来耗时太长，此处我们酌情把全部通道都除以 2
        self.blk_mid_channels = [32, 64, 128, 256]  # 原先的通道数：[64, 128, 256, 512]
        self.blk_channels = [self.blk1_in_channels] + self.blk_mid_channels  # [32, 32, 64, 128, 256]
        self.blk_stride = [1,2,2,2]  # 每个 residual layer 的 stride
        
        self.blk_channel_expansion = block.channel_expansion
        
        # 第一个卷积层（独立于 residual layers 之外）
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.blk_channels[0], kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(self.blk1_in_channels) 
        
        # residual layers (打包在 self.layers 中)
        self.layers = nn.Sequential()
        for i in range(self.residual_layers):
            blk_in_channels = self.blk_channels[i] if i==0 else self.blk_channels[i]*block.channel_expansion
            blk_mid_channels = self.blk_channels[i+1]
            self.layers.add_module(f"residule_layer{i}", 
                                   self._make_layer(block=block,  # block 种类：BasicBlock 或 BottleneckBlock
                                                    blk_in_channels=blk_in_channels,
                                                    blk_mid_channels=blk_mid_channels, 
                                                    num_blocks=num_blocks[i],  # 该 residual layer 有多少个 blocks
                                                    stride=self.blk_stride[i])
            )
        # 最后的全连接层
        self.linear = nn.Linear(in_features=self.blk_channels[self.residual_layers]*block.channel_expansion, 
                                out_features=num_classes)
        
        
    def _make_layer(self, block, blk_in_channels, blk_mid_channels, num_blocks, stride):
        block_list = []
        stride_list = [stride] + [1]*(num_blocks-1)  # 每个 block 的 stride
        
        for block_idx in range(num_blocks):
            if block_idx != 0:  # 对于 residual layer 中非第一个 block: 调整其 blk_in_channels
                blk_in_channels = blk_mid_channels*block.channel_expansion
            block_list.append(
                block(blk_in_channels=blk_in_channels, 
                      blk_mid_channels=blk_mid_channels, 
                      stride=stride_list[block_idx])
            )
        
        return nn.Sequential(*block_list)  # 返回一个 residual layer
    
    
    def forward(self, t):
        
        ################### Please finish the following code ###################

        # conv1
        # ...
        out = self.conv1(t)
        out = self.bn1(out)
        out = F.relu(out)
        
        # "residual layers"（打包在 self.layers 中）

        out = self.layers(out)
        
        # average pooling
        out = F.avg_pool2d(out, 4)  # shape of "out" before pooling (ResNet18): (batch_size, 256, 4, 4)

        # linear layer
        # out = out.reshape(XXX, XXX)
        # out = self.linear(out)
        out = out.reshape(batch_size, 256)
        out = self.linear(out)
        
        ########################################################################

        return out

################### Please finish the following code ###################

# def ResNet18():
#     return ResNet(block=XXX, num_blocks=XXX, num_classes=XXX)
def ResNet18():
    return ResNet(block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10)

# def ResNet34():
#     return ResNet(block=XXX, num_blocks=XXX, num_classes=XXX)
def ResNet34():
    return ResNet(block=BasicBlock, num_blocks=[3,4,6,3], num_classes=10)

# def ResNet50():
#     return ResNet(block=XXX, num_blocks=XXX, num_classes=XXX)
def ResNet50():
    return ResNet(block=BottleneckBlock, num_blocks=[3,4,6,3], num_classes=10)

# def ResNet101():
#     return ResNet(block=XXX, num_blocks=XXX, num_classes=XXX)
def ResNet101():
    return ResNet(block=BottleneckBlock, num_blocks=[3,4,23,3], num_classes=10)

# def ResNet152():
#     return ResNet(block=XXX, num_blocks=XXX, num_classes=XXX)
def ResNet152():
    return ResNet(block=BottleneckBlock, num_blocks=[3,8,36,3], num_classes=10)
########################################################################
def test_output_shape():
    net = ResNet18()
    x = torch.randn(batch_size,3,32,32)  # 模拟输入
    print(x.shape)
    y = net(x)
    print(net)  # 查看网络结构
    print("")
    print(y.shape)  # 查看输出形状
test_output_shape()
