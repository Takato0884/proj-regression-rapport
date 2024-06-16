import torch, json, math
import torch.nn as nn
from torch.nn import functional as F
import utils

log = utils.get_logger()

class LinearRegression(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size, args):
        super(LinearRegression, self).__init__()
        self.drop = nn.Dropout(args.drop_rate)
        self.lin1 = nn.Linear(input_size, intermediate_size)
        self.lin2 = nn.Linear(intermediate_size, intermediate_size)
        self.lin3 = nn.Linear(intermediate_size, output_size)

    def forward(self, h):

        hidden1 = self.drop(F.relu(self.lin1(h)))
        hidden2 = self.drop(F.relu(self.lin2(hidden1)))
        output = self.lin3(hidden2)

        return output
