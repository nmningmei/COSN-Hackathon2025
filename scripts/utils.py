# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 2025

@author: 黄若杰、宋奕德、朱江岳、郝守彬、梅宁

下面的函数主要用于MEG searchlight temporal decoding任务中

"""

import numpy as np
import mne
import itertools
import re
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectPercentile, mutual_info_classif, VarianceThreshold
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


from typing import Any


def get_neighbors_within_radius(pos, radius: float, ) -> list[
    int | list[int] | list[list[int]] | list[list[list[Any]]]]:

    """
    获取通道位置后输入该函数，设定sphere半径
    计算在sphere中的channel的索引并返回

    Parameters
    ----------
    pos: 所有通道的三维坐标
    radius: searchlight sphere半径

    Returns
    -------

    neighbors: 每个sphere内的mag索引组成的嵌套列表
    """

    dist_mat = cdist(pos, pos)
    neighbors = [np.where(dist_mat[i] <= radius)[0].tolist() for i in range(len(pos))]
    return neighbors


def get_channel_position(example_epochs, radius: float = 0.04):
    """
    通过被试的epoch信息，提取以每个channel为中心的sphere内mag、grad索引

    Parameters
    ----------
    example_epochs: 被试的epochs数据
    radius: searchlight sphere半径，默认值0.04

    Returns
    -------
    mag_idx_idx: list, 每个sphere内的mag索引组成的嵌套列表
    grad_idx_idx: list, 每个sphere内的grad索引组成的嵌套列表

    """

    # 提取磁强计索引
    mag_idx = mne.pick_types(example_epochs.info, meg='mag')

    # 提取梯度计索引
    grad_idx = mne.pick_types(example_epochs.info, meg='grad')

    # 获取磁强计的位置
    mag_pos = [example_epochs.info['chs'][i]['loc'][:3] for i in mag_idx]

    # 获取梯度计的位置
    grad_pos = [example_epochs.info['chs'][i]['loc'][:3] for i in grad_idx]



    mag_idx_idx = get_neighbors_within_radius(mag_pos, radius)
    # 判断sphere中是否只有中心点
    for idx in range( len( mag_idx_idx ) ):
        if len( mag_idx_idx[idx] )  <2:
            raise ValueError("radius值过小")


    grad_idx_idx = np.arange(len(grad_pos)).reshape(-1, 2)
    grad_idx_idx = [grad_idx_idx[item].flatten() for item in mag_idx_idx]

    # 判断sphere中是否只有中心点
    for idx in range( len( grad_idx_idx ) ) :
        if len( grad_idx_idx[idx] ) <4:
            raise ValueError("radius值过小")

    return mag_idx_idx, grad_idx_idx


def str2int(x):
    if type(x) is str:
        return int(re.findall(r'\d+', x)[0])
    else:
        return x


def build_model_dictionary(model_name: str = 'None + Linear-SVM',
                           class_weight: str = 'balanced',
                           remove_invariant: bool = True,
                           scaler=None,
                           l1: bool = True,
                           C: float = 1,
                           tol: float = 1e-3,
                           ):
    """
    Parameters
    ----------
    model_name : str
        DESCRIPTION. The default is 'None + Linear-SVM', which means no feature extraction and linear SVM as decoder.
    class_weight : str, optional
        DESCRIPTION. The default is 'balanced'.
    remove_invariant : bool, optional
        DESCRIPTION. The default is True.
    scaler : TYPE, optional
        Defult scaler is always Standard Scaler if it is set to None. The default is None.
    l1 : bool, optional
        DESCRIPTION. The default is True.
    C : float, optional
        DESCRIPTION. The default is 1.
    tol : float, optional
        DESCRIPTION. The default is 1e-2.

    Returns
    -------
    sklearn.pipeline
        DESCRIPTION.

    """

    np.random.seed(12345)

    xgb = RandomForestClassifier(random_state=12345)
    RF = SelectFromModel(xgb,
                         prefit=False,
                         threshold='median'  # induce sparsity
                         )
    uni = SelectPercentile(score_func=mutual_info_classif,
                           percentile=50,
                           )  # so annoying that I cannot control the random state

    pipeline = []

    if remove_invariant:
        pipeline.append(VarianceThreshold())

    if scaler == None:
        scaler = StandardScaler()
    pipeline.append(scaler)

    feature_extractor, decoder = model_name.split(' + ')

    if feature_extractor == 'PCA':
        pipeline.append(PCA(n_components=.90,
                            random_state=12345))
        # l1 = False
    elif feature_extractor == 'Mutual':
        pipeline.append(uni)
        # l1 = False
    elif feature_extractor == 'RandomForest':
        pipeline.append(RF)
        # l1 = False

    if l1:
        svm = LinearSVC(penalty='l1',  # not default
                        dual=False,  # not default
                        tol=tol,  # not default
                        random_state=12345,  # not default
                        max_iter=int(1e4),  # default
                        class_weight=class_weight,  # not default
                        C=C,
                        )
    else:
        svm = LinearSVC(penalty='l2',  # default
                        dual=True,  # default
                        tol=tol,  # not default
                        random_state=12345,  # not default
                        max_iter=int(1e4),  # default
                        class_weight=class_weight,  # not default
                        C=C,
                        )
    svm = CalibratedClassifierCV(estimator=svm,
                                 method='sigmoid',
                                 cv=8)

    bagging = BaggingClassifier(estimator=svm,
                                n_estimators=30,  # not default
                                max_features=0.9,  # not default
                                max_samples=0.9,  # not default
                                bootstrap=True,  # default
                                bootstrap_features=True,  # default
                                random_state=12345,  # not default
                                )
    knn = KNeighborsClassifier()
    tree = DecisionTreeClassifier(random_state=12345,
                                  class_weight=class_weight)
    dummy = DummyClassifier(strategy='uniform', random_state=12345, )

    if decoder == 'Linear-SVM':
        pipeline.append(svm)
    elif decoder == 'Dummy':
        pipeline.append(dummy)
    elif decoder == 'KNN':
        pipeline.append(knn)
    elif decoder == 'Tree':
        pipeline.append(tree)
    elif decoder == 'Bagging':
        pipeline.append(bagging)

    return make_pipeline(*pipeline)

def get_neighbors_idx(epochs, radius:float):
    """
    Compute channel neighborhood indices × time points for searchlight temporal decoding.

    In searchlight temporal decoding, we want to evaluate decoding accuracy
    across both spatial neighborhoods (channels) and time points. Instead of
    writing two nested loops (over channels and time), we use
    `itertools.product(mag_idx_idx, time)` to directly create all possible
    (channel_neighbors, time_point) pairs.

    Parameters
    ----------
    epochs : mne.Epochs
        MEG epochs object containing sensor information and time points.
    radius : float
        Radius (in meters) used to define the spatial neighborhood of each sensor.

    Returns
    -------
    maga_for_loop_mag : list of tuple
        List of (magnetometer_neighbors, time_point) pairs.
        - magnetometer_neighbors : list[int], indices of neighboring magnetometer channels
        - time_point : int, index of a time sample
    maga_for_loop_grad : list of tuple
        List of (gradiometer_neighbors, time_point) pairs.
        - gradiometer_neighbors : list[int], indices of neighboring gradiometer channels
        - time_point : int, index of a time sample
    """
    mag_idx_idx, grad_idx_idx = get_channel_position(epochs, radius)
    time = np.arange(epochs.times.shape[0])
    maga_for_loop_mag = list(itertools.product(mag_idx_idx, time))
    maga_for_loop_grad = list(itertools.product(grad_idx_idx, time))

    return maga_for_loop_mag, maga_for_loop_grad

def load_data(working_data, working_beha, radius: float = .04):
    """
    Load MEG epochs and corresponding behavioral data, and prepare searchlight indices.

    Parameters
    ----------
    working_data :
        File paths to the MEG epochs data (e.g., .fif files).
    working_beha :
        File paths to the behavioral data (CSV files), aligned with `working_data`.
    radius : float, optional
        Radius (in meters) for defining sensor neighborhoods in the MEG searchlight.
        Default is 0.04.

    Returns
    -------
    maga_for_loop : list of tuple
        List of (sensor_neighbors, time_point) pairs to be used for searchlight decoding.
        - sensor_neighbors : list[int], indices of neighboring magnetometer sensors
        - time_point : int, index of a time sample
    epochs_unconscious : mne.Epochs
        Concatenated MEG epochs for unconscious trials (where `visible.keys_raw == 1`).
    epochs_conscious : mne.Epochs
        Concatenated MEG epochs for conscious trials (where `visible.keys_raw == 3`).
    df_unconscious : pandas.DataFrame
        Behavioral data rows corresponding to unconscious trials.
    df_conscious : pandas.DataFrame
        Behavioral data rows corresponding to conscious trials.
    """

    epochs = []
    df = []
    for epoch_file, beha_file in zip(working_data, working_beha):
        epochs.append(mne.read_epochs(epoch_file, preload=True, verbose=False))
        df_temp = pd.read_csv(beha_file)
        df.append(df_temp)
    epochs = mne.concatenate_epochs(epochs)

    df = pd.concat(df)
    df["visible.keys_raw"] = df["visible.keys_raw"].apply(str2int)
    idx_unconscious = df['visible.keys_raw'] == 1
    idx_conscious = df['visible.keys_raw'] == 3
    epochs_unconscious = epochs[idx_unconscious]
    df_unconscious = df[idx_unconscious]
    epochs_conscious = epochs[idx_conscious]
    df_conscious = df[idx_conscious]

    
    return epochs_unconscious, epochs_conscious, df_unconscious, df_conscious


def decode_within_sphere(epoch_data, labels, cv, model_name='None + Linear-SVM'):
    """
        Perform decoding within a sphere using a specified machine learning model.

        Parameters
        ----------
        epoch_data : ndarray
            The MEG data to be used for decoding. Shape is typically (n_samples, n_features).
        labels : ndarray
            The labels corresponding to the data samples.
        cv : sklearn.model_selection._BaseKFold
            Cross-validation splitting strategy.
        model_name : str, optional
            The name of the model to use for decoding. Default is 'None + Linear-SVM'.
            The format is 'FeatureExtractor + Decoder', where the feature extractor and decoder
            are specified as strings.

        Returns
        -------
        test_scores : ndarray
            The cross-validated test scores for the decoding task.
        """
    pipeline = build_model_dictionary(model_name=model_name)
    res = cross_validate(pipeline, epoch_data, labels, cv=cv, scoring='roc_auc', n_jobs=1)
    return res['test_score']
