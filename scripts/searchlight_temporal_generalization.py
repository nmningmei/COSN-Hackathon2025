# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 2025

@author: 黄若杰、宋奕德、朱江岳、郝守彬、梅宁

下面的函数主要用于MEG searchlight temporal decoding任务中

"""
from glob import glob

import numpy as np
import mne
import itertools
import os
import re
import pandas as pd
from matplotlib import pyplot as plt
from mne.decoding import GeneralizingEstimator
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectPercentile, mutual_info_classif, VarianceThreshold
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from joblib import Parallel, delayed
import utils
from typing import Any






if __name__ == '__main__':
    subj = 9
    radius = 0.04  # in meters
    subject = 'subject' + str(subj)
    model_name = 'None + Linear-SVM'
    mag_idx_idx, grad_idx_idx = utils.get_chanpo_fif(subj, radius)
    maga_for_loop, epochs_unconscious, epochs_conscious, df_unconscious, df_conscious = utils.load_data(9, .04)
    labels_unconscious = df_unconscious['category'].map(utils.label_map).values
    labels_conscious = df_conscious['category'].map(utils.label_map).values

    scores=[]
    for sphere in mag_idx_idx :
        pipeline = utils.build_model_dictionary(model_name=model_name)
        time_gen = GeneralizingEstimator(pipeline, scoring="roc_auc", n_jobs=6, verbose=True)
        time_gen.fit(epochs_unconscious.get_data()[:, sphere, :], labels_unconscious)
        scores_within_sphere = time_gen.score(epochs_unconscious.get_data()[:, sphere, :], labels_unconscious)
        scores.append(scores_within_sphere)

        fig, ax = plt.subplots(layout="constrained")
        im = ax.matshow(
            scores_within_sphere,
            vmin=0,
            vmax=1.0,
            cmap="RdBu_r",
            origin="lower",
            extent=epochs_unconscious.times[[0, -1, 0, -1]],
        )
        ax.axhline(0.0, color="k")
        ax.axvline(0.0, color="k")
        ax.xaxis.set_ticks_position("bottom")
        ax.set_xlabel(
            'Testing Time (s)',
        )
        ax.set_ylabel('Training Time (s)')
        ax.set_title("Generalization across time and condition", fontweight="bold")
        fig.colorbar(im, ax=ax, label="Performance (ROC AUC)")
        plt.show()



    scores = np.array(scores)
    results_dir = os.path.join('../results/', 'MEG', subject)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    np.save(os.path.join(results_dir, 'searchlight_tg_scores.npy'), scores)
    # cv = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=12345)
    #
    # scores = Parallel(n_jobs=2, verbose=1, )(
    #     delayed(utils.decode_within_sphere)(**{'epoch_data': epochs_unconscious.get_data()[:, sphere, time_point],
    #                                      'labels': labels_unconscious,
    #                                      'cv': cv, }) for sphere, time_point in maga_for_loop)
    # results_dir = os.path.join('../results/', 'MEG', subject)
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)
    # np.save(os.path.join(results_dir, 'searchlight_scores.npy'), scores)

