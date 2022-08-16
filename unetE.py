import torch
from torch import nn
from typing import Tuple
from toolkit import bilinear_kernel

# For nested 3 channels are required

class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch)->None:
        super(DoubleConv, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


class Zoom(nn.Module):
    """
    Twice the width & height, channels stay the same.
    """
    def __init__(self, num_chans:int) -> None:
        super().__init__()
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.convT=nn.ConvTranspose2d(num_chans,num_chans,kernel_size=4,stride=2,bias=False,padding=1)
        # self.convT.weight.data.copy_(bilinear_kernel(num_chans,num_chans,kernel_size=4))
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x=self.Up(x)
        return x
    
# Nested Unet
class UnetE(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch:int, out_ch:int):
        super(UnetE, self).__init__()

        n1 = 64
        FILTERS = (n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16,)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up0_1=Zoom(FILTERS[1])
        
        self.up1_1=Zoom(FILTERS[2])
        self.up0_2=Zoom(FILTERS[1])

        self.up2_1=Zoom(FILTERS[3])
        self.up1_2=Zoom(FILTERS[2])
        self.up0_3=Zoom(FILTERS[1])

        self.up3_1=Zoom(FILTERS[4])
        self.up2_2=Zoom(FILTERS[3])
        self.up1_3=Zoom(FILTERS[2])
        self.up0_4=Zoom(FILTERS[1])       
        

        self.conv0_0 = DoubleConv(in_ch, FILTERS[0], FILTERS[0])
        self.conv1_0 = DoubleConv(FILTERS[0], FILTERS[1], FILTERS[1])
        self.conv2_0 = DoubleConv(FILTERS[1], FILTERS[2], FILTERS[2])
        self.conv3_0 = DoubleConv(FILTERS[2], FILTERS[3], FILTERS[3])
        self.conv4_0 = DoubleConv(FILTERS[3], FILTERS[4], FILTERS[4])

        self.conv0_1 = DoubleConv(FILTERS[0] + FILTERS[1], FILTERS[0], FILTERS[0])
        self.conv1_1 = DoubleConv(FILTERS[1] + FILTERS[2], FILTERS[1], FILTERS[1])
        self.conv2_1 = DoubleConv(FILTERS[2] + FILTERS[3], FILTERS[2], FILTERS[2])
        self.conv3_1 = DoubleConv(FILTERS[3] + FILTERS[4], FILTERS[3], FILTERS[3])

        self.conv0_2 = DoubleConv(FILTERS[0] + FILTERS[1], FILTERS[0], FILTERS[0])
        self.conv1_2 = DoubleConv(FILTERS[1] + FILTERS[2], FILTERS[1], FILTERS[1])
        self.conv2_2 = DoubleConv(FILTERS[2] + FILTERS[3], FILTERS[2], FILTERS[2])

        self.conv0_3 = DoubleConv(FILTERS[0] + FILTERS[1], FILTERS[0], FILTERS[0])
        self.conv1_3 = DoubleConv(FILTERS[1] + FILTERS[2], FILTERS[1], FILTERS[1])

        self.conv0_4 = DoubleConv(FILTERS[0] + FILTERS[1], FILTERS[0], FILTERS[0])

        self.final1 = nn.Sequential(
            nn.Conv2d(FILTERS[0], out_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(FILTERS[0], out_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.final3 = nn.Sequential(
            nn.Conv2d(FILTERS[0], out_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.final4 = nn.Sequential(
            nn.Conv2d(FILTERS[0], out_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode(self, x:torch.Tensor)->Tuple[torch.Tensor]:
        """
        return (x0_0,x1_0,x2_0,x3_0,x4_0,)
        """
        x0_0:torch.Tensor = self.conv0_0(x)
        x1_0:torch.Tensor = self.conv1_0(self.pool(x0_0))
        x2_0:torch.Tensor = self.conv2_0(self.pool(x1_0))
        x3_0:torch.Tensor = self.conv3_0(self.pool(x2_0))
        x4_0:torch.Tensor = self.conv4_0(self.pool(x3_0))
        if __name__=='__main__':
            print('x(0,0):',x0_0.shape)
            print('x(1,0):',x1_0.shape)
            print('x(2,0):',x2_0.shape)
            print('x(3,0):',x3_0.shape)
            print('x(4,0):',x4_0.shape)
        return (x0_0,x1_0,x2_0,x3_0,x4_0,)

    def decode(self, x0_0:torch.Tensor,x1_0:torch.Tensor, \
            x2_0:torch.Tensor, x3_0:torch.Tensor, \
            x4_0:torch.Tensor)->Tuple[torch.Tensor]:
        """
        数值0~1
        return (x0_1,x0_2,x0_3,x0_4,)
        """
        x0_1:torch.Tensor = self.conv0_1(torch.cat([x0_0, self.up0_1(x1_0)], dim=1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up1_1(x2_0)], dim=1))
        x0_2:torch.Tensor = self.conv0_2(torch.cat([x0_0, self.up0_2(x1_1)], dim=1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up2_1(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up1_2(x2_1)], dim=1))
        x0_3:torch.Tensor = self.conv0_3(torch.cat([x0_0, self.up0_3(x1_2)], dim=1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3_1(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up2_2(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up1_3(x2_2)], dim=1))
        x0_4:torch.Tensor = self.conv0_4(torch.cat([x0_0, self.up0_4(x1_3)], dim=1))

        if __name__ == '__main__':
            print('Without activation:')
            print('\tx(0,1):',x0_1.shape)
            print('\tx(0,2):',x0_2.shape)
            print('\tx(0,3):',x0_3.shape)
            print('\tx(0,4):',x0_4.shape)
        
        return (self.final1(x0_1), self.final2(x0_2), self.final3(x0_3), self.final4(x0_4),)
    
    def multi_forward(self, x:torch.Tensor)->Tuple[torch.Tensor]:
        """
        返回值: (x1_0,x2_0,x3_0,x4_0,| x0_1,x0_2,x0_3,x0_4,)
        """
        x0_0:torch.Tensor
        x1_0:torch.Tensor
        x2_0:torch.Tensor
        x3_0:torch.Tensor
        x4_0:torch.Tensor
        x0_0,x1_0,x2_0,x3_0,x4_0=self.encode(x)

        x0_1:torch.Tensor
        x0_2:torch.Tensor
        x0_3:torch.Tensor
        x0_4:torch.Tensor
        x0_1,x0_2,x0_3,x0_4=self.decode(x0_0,x1_0,x2_0,x3_0,x4_0)

        return (x1_0,x2_0,x3_0,x4_0,x0_1,x0_2,x0_3,x0_4,)

    
    def multi_predict(self, x:torch.Tensor)->Tuple[torch.Tensor]:
        """
        数值 0~1
        返回值: (x0_1,x0_2,x0_3,x0_4,)
        """
        return self.multi_forward(x)[4:]

if __name__ == '__main__':
    model=UnetE(1,1)
    model.train()
    model.zero_grad()

    x=torch.randn(size=(2,1,256,256))

    x0_1:torch.Tensor
    x0_2:torch.Tensor
    x0_3:torch.Tensor
    x0_4:torch.Tensor
    x0_1,x0_2,x0_3,x0_4 =model.multi_predict(x)
    label:torch.Tensor = torch.randint(0,2,size=(2, 1,256,256,))

    from dice_loss import binary_dice_loss

    outputs=(x0_1,x0_2,x0_3,x0_4,)

    # loss=binary_dice_loss(output,label)
    loss:torch.Tensor=0
    for x in outputs:
        loss+=binary_dice_loss(x,label)
    # loss=binary_dice_loss(x0_2,label)
    print('calculated loss:',loss.item())

    print('No gradient yet:',model.conv1_1.conv1.weight.grad)
    loss.backward()
    print('Calculated gradient:', model.conv1_1.conv1.weight.grad.shape,\
         model.conv1_1.conv1.weight.grad)

    print(x0_4.type(),loss.item())
    print(x0_4.size(),x0_4.dim())

    def contain01(x:torch.Tensor)->bool:
        count1=(x>1).sum().item()
        count2=(x<0).sum().item()
        if count1==0 and count2==0:
            return True
        else:
            return False
    print('check01:',contain01(x0_1),contain01(x0_2),contain01(x0_3),contain01(x0_4))