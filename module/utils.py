from torch import nn
import torch
import torch.nn.functional as F
from torch import nn
from timm.layers import trunc_normal_


class EfficientAgentAttention(nn.Module):
    def __init__(self, dim=256, num_heads=8, dropout=0., window=16, qkv_bias=True, agent_num=49):
        super(EfficientAgentAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.window = window

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)

        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)

        pool_size = int(agent_num ** 0.5)
        self.v_pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.k_pool = nn.AdaptiveMaxPool2d(output_size=(pool_size, pool_size))
    
    def forward(self, x, x_lang):
        x_res = x.clone()
        
        b, n, c = x.shape
        h = w = int(n ** 0.5)

        q = self.q(x)
        k_agent = self.k(x_lang)
        v_agent = self.v(x_lang)
        k_agent = self.k_pool(k_agent.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        v_agent = self.v_pool(v_agent.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        
        q = q.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_agent = k_agent.reshape(b, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_agent = v_agent.reshape(b, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        agent_bias1 = F.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ k_agent.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ v_agent

        x = x.transpose(1, 2).reshape(b, n, c)
        q = q.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.dwc(q).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)

        out = x_res * x

        return out

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def conv_layer(in_dim, out_dim, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_dim+2, out_dim, 1, 1),
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=in_dim, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))

class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 ):
        super().__init__()
        # self.conv1 = conv_layer(in_channels, out_channels, kernel_size, stride,
        #                         padding)
        self.conv1 = nn.Conv2d(in_channels + 2, out_channels, 1, 1)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x

class MSDC(nn.Module):
    '''
    Multi-scale Depth-wise Convolution
    '''
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                CoordConv(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2),
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        # You can return outputs based on what you intend to do with them
        return outputs
    
class CrossModalAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(CrossModalAttention, self).__init__()

        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads

        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        self.project_k = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.vis_coord = CoordConv(self.v_in_channels, self.key_channels)
        self.project_q = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.project_v = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )
        

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, C_v)
        # l input shape: (B, C_l, T)
        # l_mask shape: (B, T, 1)

        B, HW = x.size(0), x.size(1)
        H = W = int(HW ** 0.5)
        n_l = l.size(-1)

        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, T, 1) -> (B, 1, T)

        query = self.vis_coord(x.reshape(B, -1, H, W))
        query = self.project_q(query.flatten(2))  # (B, key_channels, H*W)
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.project_k(l)  # (B, key_channels, T)
        value = self.project_v(l) # (B, value_channels, T)
        key = key * l_mask  # (B, key_channels, T)
        value = value * l_mask  # (B, value_channels, T)

        query = query.reshape(B, HW, self.num_heads, self.key_channels // self.num_heads).permute(0, 2, 1, 3)
        # (B, num_heads, H*W, key_channels//num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels // self.num_heads, n_l)
        # (B, num_heads, key_channels//num_heads, T)
        value = value.reshape(B, self.num_heads, self.value_channels // self.num_heads, n_l)
        # # (B, num_heads, value_channels//num_heads, T)
        l_mask = l_mask.unsqueeze(1)  # (B, 1, 1, T)

        # attention score
        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, T)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4 * l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, T)

        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = self.W(out.permute(0, 2, 1)).permute(0, 2, 1)  # (B, out_channels, HW)

        return out
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x