import torch
import torch.nn as nn
from torch.nn.utils.rnn import packed_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, h_src, h_t_tgt, mask=None):
        # |h_t_tgt| = (bs, 1, hs) / 디코더의 현재 타임스탭의 hidden state
        # |h_src| = (bs, length, hs) / 인코더의 모든 타임스탭의 hidden state
        # |mask| = (bs, length)
        # 쿼리와 인코더의 모든 타임 스텝의 은닉 상태 간의 유사도 계산

        # (bs, 1, hs)
        query = self.linear(h_t_tgt)

        # (bs, 1, hs) * (bs, hs, length) = (bs, 1, length)
        # 디코더 배치사이즈의 각 샘플별 현재 타임 스탭의 인코더의 모든 hidden state
        weight = torch.bmm(query, h_src.transpose(1,2))

        if mask is not None:
            weight.masked_fill(mask.unsqueeze(1), -float('-inf'))
        
        weight = self.softmax(weight)

        # (bs, 1, length) * (bs, length, hs) = (bs, 1, hs)
        context_vector = torch.bmm(weight, h_src)

        return context_vector
    
