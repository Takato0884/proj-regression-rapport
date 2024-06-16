# Requirements
<img src="https://img.shields.io/badge/-Python-3776AB.svg?logo=python&style=flat&logoColor=white"> <img src="https://img.shields.io/badge/-PyTorch-EE4C2C.svg?logo=pytorch&style=flat&logoColor=white">

# Description
This project demonstrates the use of a regression model tp predict rapport scores. It includes training and evaluating the model using various parameters and settings.

# Usage Instructions
1. Change path<br>

`train/20240603/V.ipynb: In [2]`
```train/20240603/rank_sample_seq.ipynb: In [2]
with open('C:\\Users\\hayas\\Dataset_Rapport\\proj-personal-features\\202407\\output\\features_fr.pkl', 'rb') as p:
    features = pickle.load(p)
with open('C:\\Users\\hayas\\Dataset_Rapport\\proj-personal-features\\202402\\output\\cv_fr_topic1_test.pkl', 'rb') as p:
    cv_test = pickle.load(p)
with open('C:\\Users\\hayas\\Dataset_Rapport\\proj-personal-features\\202402\\output\\cv_fr_topic1_val.pkl', 'rb') as p:
    cv_val = pickle.load(p)
```

`train/20240603/V.ipynb:In [4]`
```train/20240603/rank_sample_seq.ipynb:In [4]
with open("C:\\Users\\hayas\\proj-regression-general\\git\\output\\ret\\20240603\\{}\\{}.pickle".format(modal, file_name), mode="wb") as f:
    pickle.dump(return_dict, f)
```

`train/20240603/V.ipynb:In [5]`
```train/20240603/rank_sample_seq.ipynb:In [5]
with open("C:\\Users\\hayas\\proj-regression-general\\git\\output\\ret\\20240603\\V\\exp0.pickle", 'rb') as p:
    return_dict = pickle.load(p)
```

`train/20240603/V.ipynb:In [6]`
```train/20240603/rank_sample_seq.ipynb:In [6]
with open("C:\\Users\\hayas\\proj-regression-general\\git\\output\\ret\\20240603\\V\\{}.pickle".format(file_name), 'rb') as p:
    return_dict = pickle.load(p)
```

`code/regression_seq/utils.py: line 6`
```code/ranknet_seq/utils.py: line 6
logging.basicConfig(filename="C:\\Users\\hayas\\proj-regression-general\\git\\output\\log\\20240603\\test.log", level=logging.INFO)
```

2. Run training notebook<br>

`train/20240603/V.ipynb`

# Dataset
The dataset is stored in dictionary format.<br>
`features_fr.pkl`: dataset for friend conversations.<br>
`cv_fr_topic1_test.pkl`: keys to separate train set and test set. this keys contain only the first topic.<br>
Keys of these dictionary is `[participant_id]_[stimuli_id]`. <br>
Values of these dictionary is `{"utterance": [list], "start": [list], "end_list": [list], "BERT": [list], "eGeMAPS": [list], "AU": [list],"rapport": [float], "speaker_id": [string], "BigFive": [list], "Ext": [list], "Neu": [list], "Ope": [list], "Con": [list], "Agr": [list]}`.

# Training notebook
Training file is Jupyter notebook format.<br>
`train/20240603/V.ipynb`: Training notebook using visual features.<br>

If you want to use other modality, change `modal` parameter.<br>
V: visual features
A: audio features
AV_early: cross-modal features using early fusion method ()
AV_hieral: cross-modal features using hierarchical fusion method (please refere to Section V-E in [this paper](https://link.springer.com/chapter/10.1007/978-3-031-61312-8_2))