# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 2025

@author: 黄若杰、宋奕德、朱江岳、郝守彬、梅宁

下面的函数主要用于MEG searchlight temporal decoding任务中
"""
import os
import numpy as np
from utils import load_data,decode_within_sphere,get_neighbors_idx
from glob import glob
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit


if __name__ == '__main__':
    # searchlight temporal decoding使用例子
    subj = 9 # 选择一个被试
    subject = 'subject' + str(subj)
    radius = 0.04 # in meter
    working_dir = os.path.join('../data/MEG', subject)
    working_data = np.sort(glob(os.path.join(working_dir, "cleaned", "unconscious-session*fif")))
    working_beha = np.sort(glob(os.path.join(working_dir, "cleaned", "unconscious-session*csv")))

    label_map = {
        "Nonliving_Things": 1,
        "Living_Things"   : 0,
    }

    epochs_unconscious, epochs_conscious, df_unconscious, df_conscious = load_data(working_data, working_beha)
    labels_unconscious = df_unconscious['category'].map(label_map).values
    labels_conscious = df_conscious['category'].map(label_map).values
    maga_for_loop_mag, maga_for_loop_grad = get_neighbors_idx(epochs_unconscious,radius,)
    cv = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=12345)

    scores = Parallel(n_jobs=2, verbose=1, )(
        delayed(decode_within_sphere)(**{'epoch_data': epochs_unconscious.get_data()[:, sphere, time_point],
                                         'labels': labels_unconscious,
                                         'cv': cv, }) for sphere, time_point in maga_for_loop_mag)
    results_dir = os.path.join('../results/', 'MEG', subject)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    np.save(os.path.join(results_dir, 'searchlight_scores.npy'), scores)

