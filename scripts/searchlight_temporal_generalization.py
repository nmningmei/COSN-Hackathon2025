# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 2025

@author: 黄若杰、宋奕德、朱江岳、郝守彬、梅宁

下面的函数主要用于MEG searchlight temporal generalization任务中

"""
import os
import numpy as np
from utils import load_data,get_channel_position,build_model_dictionary
from glob import glob
from mne.decoding import GeneralizingEstimator
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # searchlight temporal decoding使用例子
    subj = 9 # 选择一个被试
    subject = 'subject' + str(subj)
    radius = 0.04 # in meter
    model_name = 'None + LinearSVC'
    visualize_during_fitting = False #
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
    mag_idx_idx, grad_idx_idx = get_channel_position(epochs_unconscious, radius)

    scores=[]
    for sphere in mag_idx_idx :
        pipeline = build_model_dictionary(model_name=model_name)
        time_gen = GeneralizingEstimator(pipeline, scoring="roc_auc", n_jobs=6, verbose=True)
        time_gen.fit(epochs_unconscious.get_data()[:, sphere, :], labels_unconscious)
        scores_within_sphere = time_gen.score(epochs_unconscious.get_data()[:, sphere, :], labels_unconscious)
        scores.append(scores_within_sphere)
        if visualize_during_fitting:
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
    

