import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import utils

log = utils.get_logger()

class SeqContext(nn.Module):

    def __init__(self, input_seze, intermediate_size, args):
        super(SeqContext, self).__init__()
        self.rnn_type = args.rnn
        self.input_size = input_seze
        self.intermediate_size = intermediate_size

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(self.input_size, self.intermediate_size, dropout=args.drop_rate,
                               bidirectional=False, num_layers=2, batch_first=True)
        elif self.rnn_type == "bi_lstm":
            self.rnn = nn.LSTM(self.input_size, self.intermediate_size, dropout=args.drop_rate,
                               bidirectional=True, num_layers=2, batch_first=True)

    def forward(self, text_tensor, text_len_tensor):

        text_tensor_packed = pad_sequence(text_tensor, batch_first=True)
        packed = pack_padded_sequence(
            text_tensor_packed,
            text_len_tensor,
            batch_first=True,
            enforce_sorted=False)
        rnn_out, (h, _) = self.rnn(packed, None)
        
        if self.rnn_type == "lstm":
            encoded_utt_context, _ = pad_packed_sequence(rnn_out, batch_first=True)
            encoded_conv_context = h[1]
        elif self.rnn_type == "bi_lstm":
            encoded_utt_context, _ = pad_packed_sequence(rnn_out, batch_first=True)
            encoded_conv_context = torch.cat((h[2], h[3]), 1)
        return encoded_utt_context, encoded_conv_context
