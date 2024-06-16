import torch, json, math
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from argparse import ArgumentParser
from torch.optim import Adam, SGD, Adadelta
import seq_context
import linear_regression
import utils
import numpy as np

log = utils.get_logger()

class MainModel(nn.Module):
    def __init__(self, args):
        super(MainModel, self).__init__()
        # パラメータの設定
        self.modal = args.modal
        self.mse_loss = nn.MSELoss()
        # # 層数の指定
        input_size = {"A": 88, "V": 238}
        intermediate_size = 128
        trait_size = 0
        # モデルの指定
        self.seq_context_encoder_A = seq_context.SeqContext(input_size["A"], intermediate_size, args)
        self.linear_regression_A = linear_regression.LinearRegression(intermediate_size, intermediate_size, intermediate_size, args)
        self.seq_context_encoder_V = seq_context.SeqContext(input_size["V"], intermediate_size, args)
        self.linear_regression_V = linear_regression.LinearRegression(intermediate_size, intermediate_size, intermediate_size, args)
        if self.modal in ["A", "V"]:
            self.predictor = linear_regression.LinearRegression(intermediate_size+trait_size, intermediate_size, 1, args)
        elif "early" in self.modal:
            self.seq_context_encoder_early = seq_context.SeqContext(input_size[self.modal[0]]+input_size[self.modal[1]], intermediate_size, args)
            self.predictor = linear_regression.LinearRegression(intermediate_size+trait_size, intermediate_size, 1, args)
        elif "hieral" in self.modal:
            self.seq_context_encoder_MM = seq_context.SeqContext(intermediate_size*2, intermediate_size, args)
            self.predictor = linear_regression.LinearRegression(intermediate_size+trait_size, intermediate_size, 1, args)

    def get_rep(self, data):

        if "early" in self.modal:
            mm = torch.cat((data["AU_perceiver"], data["eGeMAPS_perceiver"]), dim=2)
            _, lstm_conv = self.seq_context_encoder_early(mm, data["utt_num_list_perceiver"])
        else:
            if "A" in self.modal:
                A_lstm_utt, lstm_conv = self.seq_context_encoder_A(data["eGeMAPS_perceiver"], data["utt_num_list_perceiver"])
            if "V" in self.modal:
                V_lstm_utt, lstm_conv = self.seq_context_encoder_V(data["AU_perceiver"], data["utt_num_list_perceiver"])
            elif "hieral" in self.modal:
                V_mlp = self.linear_regression_V(V_lstm_utt)
                A_mlp = self.linear_regression_A(A_lstm_utt)
                MM_mlp = torch.cat((V_mlp, A_mlp), dim=2)
                _, lstm_conv = self.seq_context_encoder_MM(MM_mlp, data["utt_num_list_perceiver"])

        return lstm_conv

    def forward(self, data):
        context_rep_perceiver= self.get_rep(data)
        preds = self.predictor(context_rep_perceiver)
        if len(preds.detach().to("cpu").tolist()) != 1:
            loss = self.mse_loss(torch.squeeze(preds), data["golds"])
        elif len(preds.detach().to("cpu").tolist()) == 1:
            loss = self.mse_loss(preds[0], data["golds"])
            
        return loss, preds
