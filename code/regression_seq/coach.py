import copy
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics
import utils
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

log = utils.get_logger()

class Coach:

    def __init__(self, trainset, testset, model, opt, args):
        self.trainset = trainset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args

    def train(self):
        log.info("lstm:{}, lr:{}, drop: {}, modal: {}, epoch: {}: label: {}".format(self.args.rnn, self.args.learning_rate, self.args.drop_rate, \
                                                                                     self.args.modal, self.args.epochs, self.args.label))
        best_test_rmse = None
        train_golds_list = []
        train_preds_list = []
        train_conv_list = []
        test_golds_list = []
        test_preds_list = []
        test_conv_list = []
        
        # Train
        for epoch in range(1, self.args.epochs + 1):
            train_golds, train_preds, train_conv = self.train_epoch(epoch)
            loss_test, test_golds, test_preds, test_conv, pr_test = self.evaluate()
            log.info("[Test set] [Loss {:.4f}] [PeasonR: {:.4f}]".format(loss_test, pr_test))
            if best_test_rmse is None or loss_test < best_test_rmse:
                best_test_rmse = loss_test
                log.info("best loss model.")

            train_golds_list.append(train_golds)
            train_preds_list.append(train_preds)
            train_conv_list.append(train_conv)
            test_golds_list.append(test_golds)
            test_preds_list.append(test_preds)
            test_conv_list.append(test_conv)

        log.info("-----------------------------------------------")
        return {"test_golds_list": test_golds_list, "test_preds_list": test_preds_list, "test_conv_list": test_conv_list, \
                "train_preds_list": train_preds_list, "train_golds_list": train_golds_list, "train_conv_list": train_conv_list}

    def train_epoch(self, epoch):
        dataset = self.trainset
        start_time = time.time()
        self.model.train()
        golds_list = []
        preds_list = []
        conv_list = []

        # ここからバッチごとの処理
        loss_epoch = 0
        for step, batch in enumerate(dataset):
            golds_list.extend(batch["golds"].tolist())
            self.model.zero_grad()
            for k, v in batch.items():
                if k not in ["conv_list", "utt_num_list_perceiver", "utt_num_list_target"]:
                    batch[k] = v.to("cuda:0")
            loss, preds  = self.model(batch)
            loss_epoch += np.sqrt(loss.detach().to("cpu").tolist())
            if len(preds.detach().to("cpu").tolist()) != 1:
                preds_list.extend(np.squeeze(preds.detach().to("cpu").tolist()))
            elif len(preds.detach().to("cpu").tolist()) == 1:
                preds_list.append(preds.detach().to("cpu").tolist()[0][0])
            conv_list.extend(batch["conv_list"])
            loss.backward()
            self.opt.step()

        pr = pearsonr(preds_list, golds_list)[0]
        end_time = time.time()
        log.info("[Epoch %d] [Loss: %f][PeasonR: %f][Time: %f]" %
                 (epoch, loss_epoch, pr, end_time - start_time))

        return golds_list, preds_list, conv_list

    def evaluate(self):
        dataset = self.testset
        self.model.eval()
        with torch.no_grad():
            golds_list = []
            preds_list = []
            conv_list = []
            loss_epoch = 0
            for step, batch in enumerate(dataset):
                golds_list.extend(batch["golds"].tolist())
                for k, v in batch.items():
                    if k not in ["conv_list", "utt_num_list_perceiver", "utt_num_list_target"]:
                        batch[k] = v.to("cuda:0")
                loss, preds  = self.model(batch)
                loss_epoch += np.sqrt(loss.detach().to("cpu").tolist())
                if len(preds.detach().to("cpu").tolist()) != 1:
                    preds_list.extend(np.squeeze(preds.detach().to("cpu").tolist()))
                elif len(preds.detach().to("cpu").tolist()) == 1:
                    preds_list.append(preds.detach().to("cpu").tolist()[0][0])
                conv_list.extend(batch["conv_list"])
        pr = pearsonr(preds_list, golds_list)[0]

        return loss_epoch, golds_list, preds_list, conv_list, pr
