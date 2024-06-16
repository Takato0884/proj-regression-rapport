import json
import torch, os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import utils
import pickle
import math
import numpy as np
import random

log = utils.get_logger()

def collator(minibatch_data):
    batch_size = len(minibatch_data)
    input_size_AU = len(minibatch_data[0]["AU_perceiver"][0])
    input_size_eGeMAPS = len(minibatch_data[0]["eGeMAPS_perceiver"][0])
    input_size_verbal = len(minibatch_data[0]["verbal_perceiver"][0])
    utt_num_list_perceiver = []
    for sample in minibatch_data:
        utt_num_list_perceiver.append(len(sample["AU_perceiver"]))
    utt_num_list = utt_num_list_perceiver

    temp_AU_perceiver = torch.zeros((batch_size, max(utt_num_list), input_size_AU))
    temp_eGeMAPS_perceiver = torch.zeros((batch_size, max(utt_num_list), input_size_eGeMAPS))
    temp_verbal_perceiver = torch.zeros((batch_size, max(utt_num_list), input_size_verbal))

    gold = torch.tensor([s["gold"] for s in minibatch_data]).to(torch.float32)
    trait_perceiver = torch.tensor([s["trait_perceiver"] for s in minibatch_data]).to(torch.float32)
    trait_target = torch.tensor([s["trait_target"] for s in minibatch_data]).to(torch.float32)

    perceiver_Ext = torch.tensor([s["perceiver_Ext"] for s in minibatch_data]).to(torch.float32)
    perceiver_Neu = torch.tensor([s["perceiver_Neu"] for s in minibatch_data]).to(torch.float32)
    perceiver_Ope = torch.tensor([s["perceiver_Ope"] for s in minibatch_data]).to(torch.float32)
    perceiver_Con = torch.tensor([s["perceiver_Con"] for s in minibatch_data]).to(torch.float32)
    perceiver_Agr = torch.tensor([s["perceiver_Agr"] for s in minibatch_data]).to(torch.float32)

    target_Ext = torch.tensor([s["target_Ext"] for s in minibatch_data]).to(torch.float32)
    target_Neu = torch.tensor([s["target_Neu"] for s in minibatch_data]).to(torch.float32)
    target_Ope = torch.tensor([s["target_Ope"] for s in minibatch_data]).to(torch.float32)
    target_Con = torch.tensor([s["target_Con"] for s in minibatch_data]).to(torch.float32)
    target_Agr = torch.tensor([s["target_Agr"] for s in minibatch_data]).to(torch.float32)


    conv_list = []
    for idx, data in enumerate(minibatch_data):
        conv_list.append((data["id"]))
        utt_len_perceiver = len(data["AU_perceiver"])

        tensor_AU_perceiver = torch.tensor(np.array(data["AU_perceiver"])).to(torch.float32)
        tensor_eGeMAPS_perceiver = torch.tensor(np.array(data["eGeMAPS_perceiver"])).to(torch.float32)
        tensor_verbal_perceiver = torch.tensor(np.array(data["verbal_perceiver"])).to(torch.float32)

        temp_AU_perceiver[idx, :utt_len_perceiver, :] = tensor_AU_perceiver
        temp_eGeMAPS_perceiver[idx, :utt_len_perceiver, :] = tensor_eGeMAPS_perceiver
        temp_verbal_perceiver[idx, :utt_len_perceiver, :] = tensor_verbal_perceiver

    data_to_return = {"AU_perceiver": temp_AU_perceiver, "eGeMAPS_perceiver": temp_eGeMAPS_perceiver, "verbal_perceiver": temp_verbal_perceiver,\
                      "golds": gold, "conv_list": conv_list, "utt_num_list_perceiver": utt_num_list_perceiver, "trait_perceiver": trait_perceiver, \
                      "trait_target": trait_target, \
                      "perceiver_Ext": perceiver_Ext, "perceiver_Neu": perceiver_Neu, "perceiver_Ope": perceiver_Ope, \
                      "perceiver_Con": perceiver_Con, "perceiver_Agr": perceiver_Agr, \
                        "target_Ext": target_Ext, "target_Neu": target_Neu, "target_Ope": target_Ope, \
                      "target_Con": target_Con, "target_Agr": target_Agr}

    return data_to_return

class ConversationRelDataModule():

    def __init__(self, train_dataset, test_dataset, batch_size, collator, modal, features_dict, label):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.collator = collator
        self.modal = modal
        self.features_dict = features_dict
        self.label = label
        assert self.collator is not None, "Must specify batch data collator"

    def setup(self, stage):

        if stage == "fit" or stage is None:
            self.train_data = ConversationRelDataset(self.train_dataset, self.features_dict, self.label)

        if stage == "test" or stage is None:
            self.test_data = ConversationRelDataset(self.test_dataset, self.features_dict, self.label)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collator)

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=self.collator)

class ConversationRelDataset(Dataset):
    def __init__(self, samples, features_dict, label):
            
        self.dataset = []
        for s in samples:
            k = s
            kr = k[4:8] + k[:4] + k[-1]
            self.dataset.append(
                {"id": k, "gold": features_dict[k][label], "verbal_perceiver": features_dict[k]["BERT"], \
                 "AU_perceiver": features_dict[k]["norm_AU"], "eGeMAPS_perceiver": features_dict[k]["norm_eGeMAPS"], \
                 "trait_perceiver": features_dict[k]["norm_BigFive"], "trait_target": features_dict[kr]["norm_BigFive"], \
                 "perceiver_Ext": features_dict[k]["norm_Ext"], "perceiver_Neu": features_dict[k]["norm_Neu"], \
                 "perceiver_Ope": features_dict[k]["norm_Ope"], "perceiver_Con": features_dict[k]["norm_Con"], \
                 "perceiver_Agr": features_dict[k]["norm_Agr"], \
                 "target_Ext": features_dict[kr]["norm_Ext"], "target_Neu": features_dict[kr]["norm_Neu"], \
                 "target_Ope": features_dict[kr]["norm_Ope"], "target_Con": features_dict[kr]["norm_Con"], \
                 "target_Agr": features_dict[kr]["norm_Agr"]})
        log.info("finished loading {} examples".format(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]