
import torch.nn as nn
import torch.nn.functional as F
import torch, math

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, input):
        mean_ = input.mean(-1, keepdim=True)
        std_ = input.std(-1, keepdim=True)
        return self.a_2*(input-mean_)/(std_+self.eps) + self.b_2 #???



# AttFlat, LayerNorm, AttFlat_nofc
class AttFlat(nn.Module):
    def __init__(self, hidden_size=512, flat_mlp_size=512, flat_out_size=1024, flat_glimpses=1, dropout_r=0.1):
        super(AttFlat, self).__init__()
        self.flat_glimpses = flat_glimpses

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_out_size,
            dropout_r=dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            hidden_size * flat_glimpses,
            flat_out_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x) #att: [bs,seq_len,1024]
        x_mask = (x_mask == 0).bool() #[bs, seq_len]
        att = att.masked_fill(
            x_mask.unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1) #[bs, obj_num, 1024]

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1) #torch.sum(att[:, :, i: i + 1], dim=1)=1
            )

        x_atted = torch.cat(att_list, dim=1) #[bs, 512]
        x_atted = self.linear_merge(x_atted)

        return x_atted

class AttFlat_nofc(nn.Module):
    def __init__(self, hidden_size=512, flat_mlp_size=512, flat_out_size=1024, flat_glimpses=1, dropout_r=0.1):
        super(AttFlat_nofc, self).__init__()
        self.flat_glimpses = flat_glimpses

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_out_size,
            dropout_r=dropout_r,
            use_relu=True
        )

        # self.linear_merge = nn.Linear(
        #     hidden_size * flat_glimpses,
        #     flat_out_size
        # )

    def forward(self, x, x_mask):
        att = self.mlp(x) #att: [bs,seq_len,1024]
        x_mask = (x_mask == 0).bool() #[bs, seq_len]
        att = att.masked_fill(
            x_mask.unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1) #[bs, obj_num, 1024]

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1) #torch.sum(att[:, :, i: i + 1], dim=1)=1
            )

        x_atted = torch.cat(att_list, dim=1) #[bs, 512]
        # x_atted = self.linear_merge(x_atted)

        return x_atted











