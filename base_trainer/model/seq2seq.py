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
    
class Encoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Encoder, self).__init__()

        self.rnn = nn.LSTM(
            word_vec_size,
            int(hidden_size / 2),
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, emb):
        # |emb| = (batch_size, length, word_vec_size)

        if isinstance(emb, tuple):
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True)

        else:
            x = emb

        y, h = self.rnn(x)
        # |y| = (batch_size, length, hidden_size)
        # |h[0]| = (num_layers * 2, batch_size, hidden_size / 2)

        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)

        return y, h

class Decoder(nn.Module):
    
    def __init__(self, word_vec_size, hidden_size, n_layers=4, droupout_p=.2):
        super(Decoder, self).__init__()

        self.rnn = nn.LSTM(
            word_vec_size+hidden_size,
            hidden_size,
            num_layers=n_layers,
            dropout=droupout_p,
            bidirectional=False,
            batch_first=True,
        )
    
    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        # |emb_t| : 현재 시점의 embedding vector / (bs, 1, ws)
        # |h_t_1_tilde| : 이전 시점의 출력값 / (bs, 1, hs)
        # |h_t_1[0]| : 이전 시점의 hidden state / (n_layers, bs, hs)

        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)

        if h_t_1_tilde is None:
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()
        
        # input feeding
        x = torch.cat([h_t_1_tilde, emb_t], dim=-1)

        y, h = self.rnn(x, h_t_1)

        return y,h

class Generator(nn.Module):
    
    def __init__(self, hidden_size, output_size):

        self.linear = nn.Linear(hidden_size, output_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # |x| = (bs, length, hs)
        # train : length, inference : 1
        # |y| = (bs, length, hs)
        y = self.softmax(self.linear(x))

        return y
    
class Seq2Seq(nn.Module):

    # input_size = src voc size
    # output_size = tgt voc size
    # word_vec_size = word embedding size

    def __init__(
        self,
        input_size,
        word_vec_size,
        hidden_size,
        output_size,
        n_layers=4,
        dropout_p=.2
    ):
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super(Seq2Seq, self).__init__()

        self.emb_src = nn.Embedding(input_size,word_vec_size)
        self.emb_dec = nn.Embedding(output_size, word_vec_size)

        self.encoder = Encoder(
            word_vec_size, hidden_size,
            n_layers=n_layers, dropout_p=dropout_p,
        )
        self.decoder = Decoder(
            word_vec_size, hidden_size,
            n_layers=n_layers, droupout_p=dropout_p,
        )
        self.attn = Attention(hidden_size)

        # after teacher forcing
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator()
    
    def generate_mask(self, x, length):
        mask = []

        max_length = max(length)
        for lengx in length:
            if max_length - lengx > 0 :
                mask += [torch.cat(x.new_ones(1, lengx).zero_(), 
                                x.new_ones(1, max_length-1))]
            else:
                mask += [x.new_ones(1, lengx).zero_()]
        
        mask = torch.cat(mask, dim=0).bool()

        return mask
    
    def fast_merge_encoder_hiddens(self, encoder_hiddens):
        h_0_tgt, c_0_tgt = encoder_hiddens
        batch_size = h_0_tgt.size(1)

        # (layers*2 * bs * hs/2) to (layers * bs * hs)
        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size, -1, self.hidden_size)
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size, -1, self.hidden_size)

        return h_0_tgt, c_0_tgt
    
    def forward(self, src, tgt):
        batch_size = tgt.size(0)

        mask = None
        x_length = None

        #패딩되지 않은 부분의 길이
        if isinstance(src, tuple):
            x, x_length = src
            
            mask = self.generate_mask(x, x_length)

        else:
            x = src

        if isinstance(tgt, tuple):
            tgt = tgt[0]
        
        emb_src = self.emb_src(x)
        
        # y, h
        h_src, h_0_tgt = self.encoder(emb_src, x_length)

        h_0_tgt = self.fast_merge_encoder_hiddens(h_0_tgt)
        emb_tgt = self.emb_dec(tgt)

        h_tilde = []

        h_t_tilde = None
        decoder_hidden = h_0_tgt

        for t in range(tgt.size(1)):
            
            emb_t = emb_tgt[:,t,:].unsqueeze(1)

            decoder_output, decoder_hidden = self.decoder(emb_t,h_t_tilde, decoder_hidden)

            # 인코더의 전체 hs, 
            context_vector = self.attn(h_src, decoder_output, mask)

            h_t_tilde = self.tanh(self.concat(torch.cat(decoder_output, context_vector), dim=-1))

            h_tilde += h_t_tilde
        
        h_tilde = torch.cat(h_tilde, dim=1)

        y_hat = self.generator(h_tilde)

        return y_hat
    