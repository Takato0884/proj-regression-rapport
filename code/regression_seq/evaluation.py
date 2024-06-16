import utils
from sklearn.metrics import accuracy_score
from scipy import stats
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

log = utils.get_logger()

def cal_pearson_std(features, return_dict, epoch, print_bool, train_test):
    if train_test == "train":
        preds_key = "train_preds_list"
        golds_key = "train_golds_list"
    elif train_test == "test":
        preds_key = "test_preds_list"
        golds_key = "test_golds_list"

    # 全ラベルを含むリストを作成
    label_list = []
    label = "rapport"
    for k, v in features.items():
        label_list.append(v[label])

    all_pr_list = []
    all_mse_list = []
    for k, v in return_dict.items():
        golds_list = []
        preds_list = []
        for i in v[preds_key][epoch]:
            preds_list.append((i * (np.max(label_list) - np.min(label_list))) + np.min(label_list))
        for i in v[golds_key][epoch]:
            golds_list.append((i * (np.max(label_list) - np.min(label_list))) + np.min(label_list))
        all_pr_list.append(pearsonr(golds_list, preds_list)[0])
        all_mse_list.append(mean_squared_error(golds_list, preds_list))
        if print_bool == True:
            print("{}: {}: {}".format(k, v["best_lr"], pearsonr(v[golds_key][epoch], v[preds_key][epoch])[0]))
    if print_bool == True:
        print("all: {}: {}".format(np.mean(all_pr_list), np.mean(all_mse_list)))
    return np.mean(all_pr_list), np.mean(all_mse_list)

def cal_pearson_std_fold(features, return_dict, epoch, print_bool, train_test):
    if train_test == "test":
        preds_key = "test_preds_list"
        golds_key = "test_golds_list"

    # 全ラベルを含むリストを作成
    label_list = []
    label = "rapport"
    for k, v in features.items():
        label_list.append(v[label])

    all_pr_list = []
    all_mse_list = []
    for k, v in return_dict.items():
        golds_list = []
        preds_list = []
        for i in v[preds_key][epoch]:
            preds_list.append((i * (np.max(label_list) - np.min(label_list))) + np.min(label_list))
        for i in v[golds_key][epoch]:
            golds_list.append((i * (np.max(label_list) - np.min(label_list))) + np.min(label_list))
        all_pr_list.append(pearsonr(golds_list, preds_list)[0])
        all_mse_list.append(mean_squared_error(golds_list, preds_list))
    return all_pr_list, all_mse_list

def create_id_dict_per_person(id_list):
    ranklist_dict = {}
    u_id_set = set()
    for id in id_list:
        u_id_set.add(id[:3])
    u_id_list = sorted(list(u_id_set))
    for u in u_id_list:
        ranklist_dict[u] = {"ids": []}
        for id in id_list:
            if u == id[:3]:
                ranklist_dict[u]["ids"].append(id)
    return ranklist_dict

def person_test(return_dict, epoch, print_bool):
    conv_list = []
    preds_list = []
    golds_list = []
    for k, v in return_dict.items():
        conv_list.extend(v["test_conv_list"][epoch])
        preds_list.extend(v["test_preds_list"][epoch])
        golds_list.extend(v["test_golds_list"][epoch])
    ranklist_dict = create_id_dict_per_person(conv_list)

    tau, pearson = cal_person(ranklist_dict, golds_list, preds_list, conv_list, print_bool)
    
    return tau, pearson

def person_train(return_dict, epoch, print_bool, group):
    conv_list = []
    preds_list = []
    golds_list = []
    conv_list.extend(return_dict[group]["train_conv_list"][epoch])
    preds_list.extend(return_dict[group]["train_preds_list"][epoch])
    golds_list.extend(return_dict[group]["train_golds_list"][epoch])
    ranklist_dict = create_id_dict_per_person(conv_list)

    tau, pearson = cal_person(ranklist_dict, golds_list, preds_list, conv_list, print_bool)
    
    return tau, pearson

def cal_person(ranklist_dict, golds_list, preds_list, conv_list, print_bool):
    for k, v in ranklist_dict.items():
        v["golds"] = [golds_list[conv_list.index(id)] for id in v["ids"]]
        v["preds"] = [preds_list[conv_list.index(id)] for id in v["ids"]]
        
    tau_list = []
    pearson_list = []
    for k, v in ranklist_dict.items():
        tau, p_value = stats.kendalltau(v["preds"], v["golds"])
        pearson = pearsonr(v["preds"], v["golds"])[0]
        tau_list.append(tau)
        pearson_list.append(pearson)

    if print_bool == True:
        print("tau: {}, len: {}".format(np.mean(tau_list), len(tau_list)))
        print("pearson: {}, len: {}".format(np.mean(pearson_list), len(pearson_list)))
    
    return np.mean(tau_list), np.mean(pearson_list)

def cal_effect_pred_std(features_dict, return_dict, df):
        # 全ユーザーidが入ったリストを作成
    u_id_set = set()
    for k in features_dict.keys():
        u_id_set.add(k[:3])
    u_id_list = sorted(list(u_id_set))

    # 各ユーザーごとに付けた点数と付けられた点数を辞書に集約
    person_rapport_dict = {}
    for k in u_id_list:
        person_rapport_dict[k] = {"perceiver_list": [], "target_list": []}


    for k, v in features_dict.items():
        person_rapport_dict[k[:3]]["perceiver_list"].append(v["rapport"])
        person_rapport_dict[k[4:7]]["target_list"].append(v["rapport"])
    for k, v in person_rapport_dict.items():
        person_rapport_dict[k]["mean_perceiver"] = np.mean(person_rapport_dict[k]["perceiver_list"])
        person_rapport_dict[k]["mean_target"] = np.mean(person_rapport_dict[k]["target_list"])

    # グループ平均を辞書に集約
    group_rapport_dict = {}
    for k in return_dict.keys():
        group_rapport_dict[k] = {"ids": [], "list": []}

    for k, v in features_dict.items():
        group_rapport_dict[k[0]]["ids"].append(k)
        group_rapport_dict[k[0]]["list"].append(v["rapport"])
    for k, v in group_rapport_dict.items():
        group_rapport_dict[k]["mean"] = np.mean(group_rapport_dict[k]["list"])

    for k, v in person_rapport_dict.items():
        person_rapport_dict[k]["perceiver_effect"] = (((df-1)**2)/(df*(df-2)))*v["mean_perceiver"] + ((df-1)/(df*(df-2)))*v["mean_target"] - \
                                                            ((df-1)/(df-2))*group_rapport_dict[k[0]]["mean"]
        person_rapport_dict[k]["target_effect"] = (((df-1)**2)/(df*(df-2)))*v["mean_target"] + ((df-1)/(df*(df-2)))*v["mean_perceiver"] - \
                                                            ((df-1)/(df-2))*group_rapport_dict[k[0]]["mean"]
    
    relationship_effect = {}
    for k, v in features_dict.items():
        relationship_effect[k[:7]] = v["rapport"] - person_rapport_dict[k[:3]]["perceiver_effect"] - \
                                    person_rapport_dict[k[4:7]]["target_effect"] - group_rapport_dict[k[0]]["mean"]
        
    return person_rapport_dict, relationship_effect

def SRM_effect_pred_std_test(features, return_dict, df, epoch):

    # 全ラベルを含むリストを作成
    label_list = []
    label = "rapport"
    for k, v in features.items():
        label_list.append(v[label])

    # 予測出力のスケールを真値に揃える
    # 予測出力の辞書を作成する
    features_dict = {}
    for k in return_dict.keys():
        preds_list = []
        for i2 in return_dict[k]["test_preds_list"][epoch]:
            preds_list.append((i2 * (np.max(label_list) - np.min(label_list))) + np.min(label_list))
        for c, v in zip(return_dict[k]["test_conv_list"][epoch], preds_list):
            features_dict[c] = {}
            features_dict[c]["rapport"] = v

    person_rapport_dict, relationship_effect = cal_effect_pred_std(features_dict, return_dict, df)

    return person_rapport_dict, relationship_effect

def SRM_effect_pred_std_train(features, return_dict, df, epoch, group):

    # 全ラベルを含むリストを作成
    label_list = []
    label = "rapport"
    for k, v in features.items():
        label_list.append(v[label])
        
    # 予測出力のスケールを真値に揃える
    # 予測出力の辞書を作成する
    features_dict = {}
    preds_list = []
    for i2 in return_dict[group]["train_preds_list"][epoch]:
        preds_list.append((i2 * (np.max(label_list) - np.min(label_list))) + np.min(label_list))
    for c, v in zip(return_dict[group]["train_conv_list"][epoch], preds_list):
        features_dict[c] = {}
        features_dict[c]["rapport"] = v
    person_rapport_dict, relationship_effect = cal_effect_pred_std(features_dict, return_dict, df)

    return person_rapport_dict, relationship_effect

def SRM_eval(person_rapport_dict_gold, relationship_effect_gold, person_rapport_dict_pred, relationship_effect_pred, print_bool):
    g_id_set = set()
    for k in person_rapport_dict_gold.keys():
        g_id_set.add(k[0])
    g_id_list = sorted(list(g_id_set))

    main_dict = {}
    sub_dict = {}
    for id in g_id_list:
        main_dict[id] = {"perceiver_golds": [], "perceiver_preds": [], "target_golds": [], "target_preds": []}
        sub_dict[id] = {"golds": [], "preds": []}
        for k, v in person_rapport_dict_gold.items():
            if id == k[0]:
                main_dict[id]["perceiver_golds"].append(v["perceiver_effect"])
                main_dict[id]["target_golds"].append(v["target_effect"])
        for k, v in person_rapport_dict_pred.items():
            if id == k[0]:
                main_dict[id]["perceiver_preds"].append(v["perceiver_effect"])
                main_dict[id]["target_preds"].append(v["target_effect"])
        for k, v in relationship_effect_gold.items():
            if id == k[0]:
                sub_dict[id]["golds"].append(v)
        for k, v in relationship_effect_pred.items():
            if id == k[0]:
                sub_dict[id]["preds"].append(v)
    
    p_peason_list = []
    p_mse_list = []
    for k, v in main_dict.items():
        p_peason_list.append(pearsonr(v["perceiver_preds"], v["perceiver_golds"])[0])
        p_mse_list.append(mean_squared_error(v["perceiver_preds"], v["perceiver_golds"]))
    if print_bool == True:
        print("--------Perceiver--------")
        print("pearsonR: {}".format(np.mean(p_peason_list)))
        print("MSE: {}".format(sum(p_mse_list)))

    t_peason_list = []
    t_mse_list = []
    for k, v in main_dict.items():
        t_peason_list.append(pearsonr(v["target_preds"], v["target_golds"])[0])
        t_mse_list.append(mean_squared_error(v["target_preds"], v["target_golds"]))
    if print_bool == True:
        print("--------Target--------")
        print("pearsonR: {}".format(np.mean(t_peason_list)))
        print("MSE: {}".format(sum(t_mse_list)))

    r_peason_list = []
    r_mse_list = []
    for k, v in sub_dict.items():
        r_peason_list.append(pearsonr(v["preds"], v["golds"])[0])
        r_mse_list.append(mean_squared_error(v["preds"], v["golds"]))
    if print_bool == True:
        print("--------Relationship--------")
        print("pearsonR: {}".format(np.mean(r_peason_list)))
        print("MSE: {}".format(sum(r_mse_list)))

    return np.mean(p_peason_list), sum(p_mse_list), np.mean(t_peason_list), sum(t_mse_list), np.mean(r_peason_list), sum(r_mse_list)

def cal_effect_gold_std(features_dict, return_dict, df):
    # 全ユーザーidが入ったリストを作成
    u_id_set = set()
    for k in features_dict.keys():
        u_id_set.add(k[:3])
    u_id_list = sorted(list(u_id_set))

    # 各ユーザーごとに付けた点数と付けられた点数を辞書に集約
    person_rapport_dict = {}
    for k in u_id_list:
        person_rapport_dict[k] = {"perceiver_list": [], "target_list": []}

    for k, v in features_dict.items():
        person_rapport_dict[k[:3]]["perceiver_list"].append(v["rapport"])
        person_rapport_dict[k[4:7]]["target_list"].append(v["rapport"])
    for k, v in person_rapport_dict.items():
        person_rapport_dict[k]["mean_perceiver"] = np.mean(person_rapport_dict[k]["perceiver_list"])
        person_rapport_dict[k]["mean_target"] = np.mean(person_rapport_dict[k]["target_list"])

    # グループ平均を辞書に集約
    group_rapport_dict = {}
    for k in return_dict.keys():
        group_rapport_dict[k] = {"ids": [], "list": []}

    for k, v in features_dict.items():
        group_rapport_dict[k[0]]["ids"].append(k)
        group_rapport_dict[k[0]]["list"].append(v["rapport"])
    for k, v in group_rapport_dict.items():
        group_rapport_dict[k]["mean"] = np.mean(group_rapport_dict[k]["list"])

    for k, v in person_rapport_dict.items():
        person_rapport_dict[k]["perceiver_effect"] = (((df-1)**2)/(df*(df-2)))*v["mean_perceiver"] + ((df-1)/(df*(df-2)))*v["mean_target"] - \
                                                            ((df-1)/(df-2))*group_rapport_dict[k[0]]["mean"]
        person_rapport_dict[k]["target_effect"] = (((df-1)**2)/(df*(df-2)))*v["mean_target"] + ((df-1)/(df*(df-2)))*v["mean_perceiver"] - \
                                                            ((df-1)/(df-2))*group_rapport_dict[k[0]]["mean"]
    
    relationship_effect = {}
    for k, v in features_dict.items():
        relationship_effect[k[:7]] = v["rapport"] - person_rapport_dict[k[:3]]["perceiver_effect"] - \
                                    person_rapport_dict[k[4:7]]["target_effect"] - group_rapport_dict[k[0]]["mean"]
        
    return person_rapport_dict, relationship_effect

def SRM_effect_gold_test(features, return_dict, df):

    features_dict = {}
    for k, v in features.items():
        if k[-1] == "1":
            features_dict[k] = {}
            features_dict[k]["rapport"] = v["rapport"]

    person_rapport_dict, relationship_effect = cal_effect_gold_std(features_dict, return_dict, df)
        
    return person_rapport_dict, relationship_effect

def SRM_effect_gold_train(features, return_dict, df, epoch, group):

    features_dict = {}
    for c in return_dict[group]["train_conv_list"][epoch]:
        features_dict[c] = {}
        features_dict[c]["rapport"] = features[c]["rapport"]
    person_rapport_dict, relationship_effect = cal_effect_pred_std(features_dict, return_dict, df)

    return person_rapport_dict, relationship_effect

def cal_variance(features_dict, return_dict, df, print_bool):
    # 全ユーザーidが入ったリストを作成
    u_id_set = set()
    for k in features_dict.keys():
        u_id_set.add(k[:3])
    u_id_list = sorted(list(u_id_set))

    g_id_set = set()
    for k in features_dict.keys():
        g_id_set.add(k[0])
    g_id_list = sorted(list(g_id_set))

    # 各ユーザーごとに付けた点数と付けられた点数を辞書に集約
    person_rapport_dict = {}
    for k in u_id_list:
        person_rapport_dict[k] = {"perceiver_list": [], "target_list": []}

    for k, v in features_dict.items():
        person_rapport_dict[k[:3]]["perceiver_list"].append(v["rapport"])
        person_rapport_dict[k[4:7]]["target_list"].append(v["rapport"])
    for k, v in person_rapport_dict.items():
        person_rapport_dict[k]["mean_perceiver"] = np.mean(person_rapport_dict[k]["perceiver_list"])
        person_rapport_dict[k]["mean_target"] = np.mean(person_rapport_dict[k]["target_list"])

    # グループ平均を辞書に集約
    group_rapport_dict = {}
    for k in g_id_list:
        group_rapport_dict[k] = {"ids": [], "list": []}

    for k, v in features_dict.items():
        group_rapport_dict[k[0]]["ids"].append(k)
        group_rapport_dict[k[0]]["list"].append(v["rapport"])
    for k, v in group_rapport_dict.items():
        group_rapport_dict[k]["mean"] = np.mean(group_rapport_dict[k]["list"])

    for k, v in person_rapport_dict.items():
        person_rapport_dict[k]["perceiver_effect"] = (((df-1)**2)/(df*(df-2)))*v["mean_perceiver"] + ((df-1)/(df*(df-2)))*v["mean_target"] - \
                                                            ((df-1)/(df-2))*group_rapport_dict[k[0]]["mean"]
        person_rapport_dict[k]["target_effect"] = (((df-1)**2)/(df*(df-2)))*v["mean_target"] + ((df-1)/(df*(df-2)))*v["mean_perceiver"] - \
                                                            ((df-1)/(df-2))*group_rapport_dict[k[0]]["mean"]
    
    relationship_effect = {}
    for k, v in features_dict.items():
        relationship_effect[k[:7]] = v["rapport"] - person_rapport_dict[k[:3]]["perceiver_effect"] - \
                                    person_rapport_dict[k[4:7]]["target_effect"] - group_rapport_dict[k[0]]["mean"]
        
    for k1, v1 in group_rapport_dict.items():
        ms_b_list = []
        ms_w_list = []
        for k2, v2 in relationship_effect.items():
            if k1 == k2[0]:
                rev_key = k2[4:8] + "_" + k2[:3]
                ms_b_list.append(((v2+relationship_effect[rev_key])**2))
                ms_w_list.append(((v2-relationship_effect[rev_key])**2))
        v1["ms_b"] = sum(ms_b_list)/(2*((df-1)*(df-2)-1))
        v1["ms_w"] = sum(ms_w_list)/(2*(df-1)*(df-2))
        v1["relationship_var"] = (v1["ms_b"] + v1["ms_w"])/2

    for k1, v1 in group_rapport_dict.items():
        perceiver_effect_list = []
        target_effect_list = []
        cor = (v1["ms_b"]/(2*(df-2))) + (v1["ms_w"]/(2*df))
        for k2, v2 in person_rapport_dict.items():
            if k1 == k2[0]:
                perceiver_effect_list.append(v2["perceiver_effect"])
                target_effect_list.append(v2["target_effect"])
        v1["perceiver_var"] = sum([i**2 for i in perceiver_effect_list])/(df-1) - cor
        v1["target_var"] = sum([i**2 for i in target_effect_list])/(df-1) - cor

    p_list, t_list, r_list = [], [], []
    for k, v in group_rapport_dict.items():
        p_var = v["perceiver_var"] if v["perceiver_var"] >= 0 else 0
        t_var = v["target_var"] if v["target_var"] >= 0 else 0
        r_var = v["relationship_var"]
        total_var = p_var + t_var + r_var
        p_list.append(p_var/total_var)
        t_list.append(t_var/total_var)
        r_list.append(r_var/total_var)
        if print_bool == True:
            print("{}: {}: {}: {}".format(k, p_var/total_var, t_var/total_var, r_var/total_var))
    if print_bool == True:    
        print("{}: {}: {}".format(np.mean(p_list), np.mean(t_list), np.mean(r_list)))

    return np.mean(p_list), np.mean(t_list), np.mean(r_list)

def SRM_variance_test(features, return_dict, df, epoch, print_bool):
    # 全ラベルを含むリストを作成
    label_list = []
    label = "rapport"
    for k, v in features.items():
        label_list.append(v[label])

    # 予測出力のスケールを真値に揃える
    # 予測出力の辞書を作成する
    features_dict = {}
    for k in return_dict.keys():
        preds_list = []
        for i2 in return_dict[k]["test_preds_list"][epoch]:
            preds_list.append((i2 * (np.max(label_list) - np.min(label_list))) + np.min(label_list))
        for c, v in zip(return_dict[k]["test_conv_list"][epoch], preds_list):
            features_dict[c] = {}
            features_dict[c]["rapport"] = v

    p_var, t_var, r_var = cal_variance(features_dict, return_dict, df, print_bool)

    return p_var, t_var, r_var

def SRM_variance_train(features, return_dict, df, epoch, print_bool, group):
    # 全ラベルを含むリストを作成
    label_list = []
    label = "rapport"
    for k, v in features.items():
        label_list.append(v[label])
        
    # 予測出力のスケールを真値に揃える
    # 予測出力の辞書を作成する
    features_dict = {}
    preds_list = []
    for i2 in return_dict[group]["train_preds_list"][epoch]:
        preds_list.append((i2 * (np.max(label_list) - np.min(label_list))) + np.min(label_list))
    for c, v in zip(return_dict[group]["train_conv_list"][epoch], preds_list):
        features_dict[c] = {}
        features_dict[c]["rapport"] = v

    p_var, t_var, r_var = cal_variance(features_dict, return_dict, df, print_bool)

    return p_var, t_var, r_var

def plot_learning_curve(features, return_dict, epoch_num, print_bool, train_test):
    pr_list = []
    mse_list = []
    for epoch in range(epoch_num):
        pr, mse = cal_pearson_std(features, return_dict, epoch, print_bool, train_test)
        pr_list.append(pr)
        mse_list.append(mse)

    epoch_list = [i for i in range(epoch_num)]
    fig, ax1 = plt.subplots(figsize=(4,3))
    ax2 = ax1.twinx()
    ax1.plot(epoch_list, pr_list, marker=',', label='PCC')
    ax2.plot(epoch_list, mse_list, marker=',', color = "#FF7F0E", label='MSE')
    ax1.set_ylim([0.0, 1.0])
    ax2.set_ylim([0, 600])
    # fig.legend(fontsize=6)
    plt.show()

def plot_pearson(features, return_dict, df, epoch_num, print_bool, train_test):
    pr_list = []
    p_pr_list = []
    t_pr_list = []
    r_pr_list = []
    if train_test == "test":
        person_rapport_dict_gold, relationship_effect_gold = SRM_effect_gold_test(features, return_dict, df)
        for epoch in range(epoch_num):
            pr, mse = cal_pearson_std(features, return_dict, epoch, print_bool, train_test)
            person_rapport_dict_pred, relationship_effect_pred = SRM_effect_pred_std_test(features, return_dict, df, epoch)
            p_pr, p_mse, t_pr, t_mse, r_pr, r_mse = SRM_eval(person_rapport_dict_gold, relationship_effect_gold, person_rapport_dict_pred, relationship_effect_pred, print_bool)
            pr_list.append(pr)
            p_pr_list.append(p_pr)
            t_pr_list.append(t_pr)
            r_pr_list.append(r_pr)
    elif train_test == "train":
        for epoch in range(epoch_num):
            p_pr_list_epoch = []
            t_pr_list_epoch = []
            r_pr_list_epoch = []
            pr, mse = cal_pearson_std(features, return_dict, epoch, print_bool, train_test)
            for fold in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                person_rapport_dict_pred, relationship_effect_pred = SRM_effect_pred_std_train(features, return_dict, df, epoch, fold)
                person_rapport_dict_gold, relationship_effect_gold = SRM_effect_gold_train(features, return_dict, df, epoch, fold)
                p_pr, p_mse, t_pr, t_mse, r_pr, r_mse = SRM_eval(person_rapport_dict_gold, relationship_effect_gold, person_rapport_dict_pred, relationship_effect_pred, print_bool)
                p_pr_list_epoch.append(p_pr)
                t_pr_list_epoch.append(t_pr)
                r_pr_list_epoch.append(r_pr)
            pr_list.append(pr)
            p_pr_list.append(np.mean(p_pr_list_epoch))
            t_pr_list.append(np.mean(t_pr_list_epoch))
            r_pr_list.append(np.mean(r_pr_list_epoch))

    epoch_list = [i for i in range(epoch_num)]
    fig, ax1 = plt.subplots(figsize=(4,3))
    ax1.plot(epoch_list, pr_list, marker=',', label='All')
    ax1.plot(epoch_list, p_pr_list, marker=',', label='Perceiver')
    ax1.plot(epoch_list, t_pr_list, marker=',', label='Target')
    ax1.plot(epoch_list, r_pr_list, marker=',', label='Relationship')
    if train_test == "train":
        ax1.set_ylim([0.0, 1.0])
    elif train_test == "test":
        ax1.set_ylim([-0.2, 0.8])
    fig.legend(fontsize=6)
    plt.show()

def plot_mse(features, return_dict, df, epoch_num, print_bool, train_test):
    mse_list = []
    p_mse_list = []
    t_mse_list = []
    r_mse_list = []
    if train_test == "test":
        person_rapport_dict_gold, relationship_effect_gold = SRM_effect_gold_test(features, return_dict, df)
        for epoch in range(epoch_num):
            pr, mse = cal_pearson_std(features, return_dict, epoch, print_bool, train_test)
            person_rapport_dict_pred, relationship_effect_pred = SRM_effect_pred_std_test(features, return_dict, df, epoch)
            p_pr, p_mse, t_pr, t_mse, r_pr, r_mse = SRM_eval(person_rapport_dict_gold, relationship_effect_gold, person_rapport_dict_pred, relationship_effect_pred, print_bool)
            mse_list.append(mse)
            p_mse_list.append(p_mse)
            t_mse_list.append(t_mse)
            r_mse_list.append(r_mse)
    elif train_test == "train":
        for epoch in range(epoch_num):
            p_mse_list_epoch = []
            t_mse_list_epoch = []
            r_mse_list_epoch = []
            pr, mse = cal_pearson_std(features, return_dict, epoch, print_bool, train_test)
            for fold in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                person_rapport_dict_pred, relationship_effect_pred = SRM_effect_pred_std_train(features, return_dict, df, epoch, fold)
                person_rapport_dict_gold, relationship_effect_gold = SRM_effect_gold_train(features, return_dict, df, epoch, fold)
                p_pr, p_mse, t_pr, t_mse, r_pr, r_mse = SRM_eval(person_rapport_dict_gold, relationship_effect_gold, person_rapport_dict_pred, relationship_effect_pred, print_bool)
                p_mse_list_epoch.append(p_mse)
                t_mse_list_epoch.append(t_mse)
                r_mse_list_epoch.append(r_mse)
            mse_list.append(mse)
            p_mse_list.append(np.mean(p_mse_list_epoch))
            t_mse_list.append(np.mean(t_mse_list_epoch))
            r_mse_list.append(np.mean(r_mse_list_epoch))

    epoch_list = [i for i in range(epoch_num)]
    fig, ax1 = plt.subplots(figsize=(4,3))
    ax1.plot(epoch_list, mse_list, marker=',', label='All')
    ax1.plot(epoch_list, p_mse_list, marker=',', label='Perceiver')
    ax1.plot(epoch_list, t_mse_list, marker=',', label='Target')
    ax1.plot(epoch_list, r_mse_list, marker=',', label='Relationship')
    ax1.set_ylim([0, 1500])
    fig.legend(fontsize=6)
    plt.show()

def plot_var(features, return_dict, df, epoch_num, print_bool, train_test):
    p_var_list = []
    t_var_list = []
    r_var_list = []
    if train_test == "test":
        for epoch in range(epoch_num):
            p_var, t_var, r_var = SRM_variance_test(features, return_dict, df, epoch, print_bool)
            p_var_list.append(p_var)
            t_var_list.append(t_var)
            r_var_list.append(r_var)
    elif train_test == "train":
        for epoch in range(epoch_num):
            p_var_list_epoch = []
            t_var_list_epoch = []
            r_var_list_epoch = []
            for fold in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                p_var, t_var, r_var = SRM_variance_train(features, return_dict, df, epoch, print_bool, fold)
                p_var_list_epoch.append(p_var)
                t_var_list_epoch.append(t_var)
                r_var_list_epoch.append(r_var)
            p_var_list.append(np.mean(p_var_list_epoch))
            t_var_list.append(np.mean(t_var_list_epoch))
            r_var_list.append(np.mean(r_var_list_epoch))

    epoch_list = [i for i in range(epoch_num)]
    fig, ax1 = plt.subplots(figsize=(4,3))
    ax1.plot(epoch_list, p_var_list, marker=',', color = "#FF7F0E", label='Perceiver')
    ax1.plot(epoch_list, t_var_list, marker=',', color = "#2CA02C", label='Target')
    ax1.plot(epoch_list, r_var_list, marker=',', color = "#D62728", label='Relationship')
    ax1.set_ylim([0, 1])
    fig.legend(fontsize=6)
    plt.show()

def plot_person(return_dict, epoch_num, print_bool, train_test):
    tau_list = []
    pearson_list = []
    if train_test == "test":
        for epoch in range(epoch_num):
            tau, pearson = person_test(return_dict, epoch, print_bool)
            tau_list.append(tau)
            pearson_list.append(pearson)
    elif train_test == "train":
        for epoch in range(epoch_num):
            tau_list_epoch = []
            pearson_list_list_epoch = []
            for fold in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                tau, pearson = person_train(return_dict, epoch, print_bool, fold)
                tau_list_epoch.append(tau)
                pearson_list_list_epoch.append(pearson)
            tau_list.append(np.mean(tau_list_epoch))
            pearson_list.append(np.mean(pearson_list_list_epoch))
    
    epoch_list = [i for i in range(epoch_num)]
    fig, ax1 = plt.subplots(figsize=(4,3))
    ax1.plot(epoch_list, tau_list, marker=',', label='KTCC')
    ax1.plot(epoch_list, pearson_list, marker=',', label='PCC')
    if train_test == "train":
        ax1.set_ylim([0.0, 1.0])
    elif train_test == "test":
        ax1.set_ylim([-0.2, 0.4])
    fig.legend(fontsize=6)
    plt.show()