import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

ratio = 8
loc_time = time.strftime("%H%M%S", time.localtime()) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def squash(s):
    mag_sq = torch.sum(s**2, dim=2, keepdim=True)
    mag = torch.sqrt(mag_sq)
    s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
    return s

class FixCapsNet(nn.Module):
    def __init__(self,conv_inputs,conv_outputs,
                 num_primary_units,
                 primary_unit_size,output_unit_size,
                 num_classes=7,init_weights=False):
        super().__init__()
                
        self.Convolution = nn.Sequential(nn.Conv2d(conv_inputs, conv_outputs, 31,stride=2),
                                     nn.ReLU(inplace=True),    
                                     nn.FractionalMaxPool2d(2, output_size=(20))
                                    )
        
        self.CBAM = Conv_CBAM(conv_outputs,conv_outputs)
        
        self.primary = Primary_Caps(in_channels=conv_outputs,#128
                                    caps_units=num_primary_units,#8
                                    )

        self.digits = Digits_Caps(in_units=num_primary_units,#8
                                   in_channels=primary_unit_size,#16*6*6=576
                                   num_units=num_classes,#classification_num
                                   unit_size=output_unit_size,#16
                                   )
        self.relu = nn.ReLU(inplace=True)
        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        x = self.Convolution(x)
        x = self.CBAM(x)
        out = self.digits(self.primary(x))
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
           
    def loss(self, img_input, target, size_average=True):
        batch_size = img_input.size(0)
        # ||vc|| from the paper -- Dynamic Routing Between Capsules.
        # https://arxiv.org/pdf/1710.09829.pdf
        v_mag = torch.sqrt(torch.sum(img_input**2, dim=2, keepdim=True))

        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1)).to(device)
        m_plus, m_minus = 0.9, 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2
        loss_lambda = 0.5#0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = torch.sum(L_c,1)

        if size_average:
            L_c = torch.mean(L_c)
            
        return L_c
    
class Primary_Caps(nn.Module):
    def __init__(self, in_channels, caps_units):
        super(Primary_Caps, self).__init__()

        self.in_channels = in_channels
        self.caps_units = caps_units
        
        def create_conv_unit(unit_idx):
            unit = ConvUnit(in_channels=in_channels)
            self.add_module("Caps_" + str(unit_idx), unit)
            return unit
        self.units = [create_conv_unit(i) for i in range(self.caps_units)]
   
    #no_routing
    def forward(self, x):
        # Get output for each unit.
        # Each will be (batch, channels, height, width).#[batch,16,6,6]
        u = [self.units[i](x) for i in range(self.caps_units)]
        # Stack all unit outputs (batch, unit, channels, height, width).
        u = torch.stack(u, dim=1)
        # Flatten to (batch, unit, output).
        u = u.view(x.size(0), self.caps_units, -1) # [batch,8,576]
        # Return squashed outputs.
        return squash(u)
    
class Digits_Caps(nn.Module):
    def __init__(self, in_units, in_channels, num_units, unit_size):
        super(Digits_Caps, self).__init__()

        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.W = nn.Parameter(torch.randn(1, in_channels, self.num_units, unit_size, in_units))
        # self.w = [1,576,7,16,8]
        
    #routing
    def forward(self, x):
        batch_size = x.size(0)    
        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2)  #[batch,576,8]       
        # (batch, features, in_units) -> (batch, features, num_units, in_units, 1)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4) #[batch,576,7, 8, 1]         
        # (batch, features, in_units, unit_size, num_units)
        W = torch.cat([self.W] * batch_size, dim=0)
        # Transform inputs by weight matrix.
        # (batch_size, features, num_units, unit_size, 1)
        u_hat = torch.matmul(W, x) #[batch,576,7,16,1]
        # Initialize routing logits to zero.
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1)).to(device)

        num_iterations = 3
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            c_ij = b_ij.softmax(dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            # [batch, 1, 7, 16, 1]
            s_j = torch.sum(c_ij * u_hat, dim=1, keepdim=True)

            # (batch_size, 1, num_units, unit_size, 1)
            v_j = squash(s_j)#CapsuleLayer.squash

            # (batch_size, features, num_units, unit_size, 1)
            # [batch, 576, 7, 16, 1]
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update b_ij (routing)
            b_ij = b_ij + u_vj1
            
        return v_j.squeeze(1)

class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()
        
        Caps_out = in_channels // ratio # 16
        self.Cpas = nn.Sequential(nn.Conv2d(in_channels,Caps_out,9,
                                  stride=2,groups=Caps_out),
                                 )
    def forward(self, x):
        out = self.Cpas(x)
        return out

class Conv_CBAM(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv_CBAM, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.ca = ChannelAttention(c2, reduction=1)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x
    
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size,padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        me_c = channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1) # GMP
        self.fc1   = nn.Conv2d(channels, me_c, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(me_c, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
 