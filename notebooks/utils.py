import os
import pickle
import datetime
from pathlib import Path
from tqdm.notebook import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import sweetviz as sv
import lightgbm as lgb
import mne

# 元データのラベルをAASMによる分類IDに変更
RANK_LABEL2ID = {
    'Movement time': -1,
    'Sleep stage ?': -1,
    'Sleep stage W': 4,
    'Sleep stage R': 3,
    'Sleep stage 1': 2,
    'Sleep stage 2': 1,
    'Sleep stage 3': 0,
    'Sleep stage 4': 0,
}

# submissionで使用するラベル
LABEL2ID = {
    'Movement time': -1,
    'Sleep stage ?': -1,
    'Sleep stage W': 4,
    'Sleep stage R': 3,
    'Sleep stage 1': 2,
    'Sleep stage 2': 1,
    'Sleep stage 3/4': 0,
}

ID2LABEL = { v: k for k, v in LABEL2ID.items() }

DATA_DIR = Path("../data")
EDF_DIR = DATA_DIR / "edf_data"
SUBMISSION_DIR = Path("../submission")

def load_epoch(epoch_name: str):
    if epoch_name not in ["train", "test"]:
        ValueError("Epoch name must be 'train' or 'test'")
        
    with open(DATA_DIR / f'{epoch_name}_epochs.pickle', mode='rb') as f:
        epochs = pickle.load(f)
    return epochs
