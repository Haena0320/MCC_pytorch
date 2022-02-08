
import torch.nn as nn
import torch.nn.functional as F
import torch, math

class MLP(nn.Module):
    def __init__(self, in_size, out_size, dropout_r, use_relu):
        super(MLP, self).__init__()
        self.mlp = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(dropout_r) if dropout_r > 0 else None
        self.relu = nn.ReLU() if use_relu else None

    def forward(self, input):
        output = self.mlp(input)
        if self.relu is not None:
            output = self.relu(output)
        if self.dropout is not None:
            output = self.dropout(output)
        return output

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
    def __init__(self, in_size=512, out_size=1024, dropout_r=0.1):
        super(AttFlat, self).__init__()
        self.mlp = MLP(in_size=in_size,
                       out_size=in_size,
                       dropout_r=dropout_r,
                       use_relu=True)
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, input, input_mask): # input : [bs, seq_len, 512]
        att = self.mlp(input) # [bs, seq_len, 1024]
        input_mask = (input_mask == 0).bool()
        att = att.masked_fill(input_mask.unsqueeze(2), -1e9)
        att = F.softmax(att, dim=-1) # [bs, seq_len, 1]
        output = torch.sum(att*input, dim=1)  #[bs, 512]
        output = self.linear(output) # [bs, 1024]
        return output

class AttFlat_nofc(nn.Module):
    def __init__(self, in_size=512, out_size=1024,dropout_r=0.1):
        super(AttFlat_nofc, self).__init__()
        self.mlp = MLP(in_size=in_size,
                       out_size=out_size,
                       dropout_r=dropout_r,
                       use_relu=True)

    def forward(self, input, input_mask): # input : [bs, seq_len, 512]
        att = self.mlp(input) # [bs, seq_len, 1024]
        input_mask = (input_mask == 0).bool()
        att = att.masked_fill_(input_mask.unsqueeze(2), -1e9).cuda()
        att = F.softmax(att, dim=-1) # [bs, seq_len, 1]
        output = torch.sum(att*input, dim=1)  #[bs, 512]
        return output












