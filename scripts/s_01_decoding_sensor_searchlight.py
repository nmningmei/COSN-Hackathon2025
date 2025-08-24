import numpy as np
import mne
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.spatial.distance import cdist
import os
from tqdm import tqdm
import experiment_settings as ES
from settings import (
    subs,
    epochs_types,
    meg_types,
    n_perm,
    n_jobs,
    get_sub_working_dir,
    get_sub_subjects_dir,
    get_sub_trans_dir,
    get_processing_data_sub_dir,
    get_sorted_data_dir,
)
from joblib import Parallel, delayed


def sensor_searchlight_decoding(sub, epochs_type, radius=0.04):
    """
    对单被试sensor空间进行searchlight解码
    """
    # 路径设置，参考s10
    working_dir = get_sub_working_dir(sub)
    subjects_dir = get_sub_subjects_dir(sub)
    trans_dir = get_sub_trans_dir(sub)
    processing_data_dir = get_processing_data_sub_dir(sub)
    if not os.path.exists(trans_dir):
        os.mkdir(trans_dir)
    fname_epochs = os.path.join(processing_data_dir, f"{epochs_type}_{sub}-epo.fif")
    epochs = mne.read_epochs(fname_epochs, preload=True)
    # 只取MEG通道
    picks = mne.pick_types(epochs.info, meg=True, eeg=False)
    ch_names = [epochs.ch_names[p] for p in picks]
    n_trials = len(epochs)
    n_chans = len(picks)
    n_times = len(epochs.times)
    data = epochs.get_data()[:, picks, :]  # (n_trials, n_chans, n_times)
    # 获取通道三维坐标
    pos = np.array([epochs.info["chs"][p]["loc"][:3] for p in picks])
    dist_mat = cdist(pos, pos)  # (n_chans, n_chans)
    # 构造标签
    y = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        event_code = epochs.events[i, 2]
        if event_code in [11, 12, 13, 19]:
            y[i] = 1
        elif event_code in [21, 22, 23, 29]:
            y[i] = 2
        else:
            raise ValueError(f"未知的event_code: {event_code}")
    n_splits = ES.n_splits if hasattr(ES, "n_splits") else 20
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=12345)
    print(
        f"Sensor searchlight decoding: {sub}, radius={radius}, n_chans={n_chans}, n_times={n_times}"
    )
    # 主循环
    scores = np.full((n_splits, n_chans, n_times), np.nan)
    for c in tqdm(range(n_chans)):
        idx = np.where(dist_mat[c] <= radius)[0]
        if len(idx) < 2:
            continue
        for t in range(n_times):
            X = data[:, idx, t]  # (n_trials, n_features)
            pipeline = make_pipeline(
                StandardScaler(),
                LinearSVC(
                    penalty="l1",
                    C=1,
                    dual=False,
                    max_iter=1e4,
                    tol=1e-4,
                    random_state=12345,
                    class_weight="balanced",
                ),
            )
            try:
                res = cross_validate(
                    pipeline,
                    X,
                    y,
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=1,
                    verbose=0,
                )
                scores[:, c, t] = res["test_score"]
            except Exception as e:
                print(f"Chan {c}, time {t} decoding failed: {e}")
    # 保存
    sorted_data_dir = get_sorted_data_dir()
    res_dir = os.path.join(sorted_data_dir, "within_subject_sensor_level_decoding", sub)
    os.makedirs(res_dir, exist_ok=True)
    fname_base = os.path.join(res_dir, f"sensor_searchlight_{epochs_type}_{sub}")
    np.save(fname_base + "_scores.npy", scores)
    np.save(fname_base + "_ch_names.npy", ch_names)
    np.save(fname_base + "_pos.npy", pos)
    print(f"Sensor searchlight decoding results saved to {fname_base}_*.npy")
    print(f"scores shape: {scores.shape}")


def load_results(sub, epochs_type, radius=0.04):
    sorted_data_dir = get_sorted_data_dir()
    res_dir = os.path.join(sorted_data_dir, "within_subject_sensor_level_decoding", sub)
    fname_base = os.path.join(res_dir, f"sensor_searchlight_{epochs_type}_{sub}")

    scores = np.load(fname_base + "_scores.npy")
    ch_names = np.load(fname_base + "_ch_names.npy")
    pos = np.load(fname_base + "_pos.npy")

    return {"scores": scores, "ch_names": ch_names, "pos": pos}


def get_neighbors_within_radius(pos, radius):
    """
    对于每个通道，返回其半径范围内所有通道的下标
    pos: (n_chans, 3) 通道坐标
    radius: 距离阈值
    返回: list，list[i]为第i个通道半径范围内的通道下标
    """
    dist_mat = cdist(pos, pos)
    neighbors = [np.where(dist_mat[i] <= radius)[0].tolist() for i in range(len(pos))]
    return neighbors


def count_neighbors_within_radius(pos, radius):
    """
    统计每个通道在指定半径内有多少个通道（包括自身）
    pos: (n_chans, 3) 通道坐标
    radius: 距离阈值
    返回: list，list[i]为第i个通道半径范围内的通道数量
    """
    dist_mat = cdist(pos, pos)
    counts = [np.sum(dist_mat[i] <= radius) for i in range(len(pos))]
    return counts


def sensor_searchlight_decoding_by_meg_type(
    sub, epochs_type, radius=0.04, n_jobs=8, meg_type="mag"
):
    """
    对指定类型的MEG通道（mag或grad）进行searchlight解码，并行处理每个通道
    """
    processing_data_dir = get_processing_data_sub_dir(sub)

    fname_epochs = os.path.join(processing_data_dir, f"{epochs_type}_{sub}-epo.fif")
    epochs = mne.read_epochs(fname_epochs, preload=True)
    picks = mne.pick_types(epochs.info, meg=meg_type, eeg=False)
    ch_names = [epochs.ch_names[p] for p in picks]
    n_trials = len(epochs)
    n_chans = len(picks)
    n_times = len(epochs.times)
    data = epochs.get_data()[:, picks, :]
    pos = np.array([epochs.info["chs"][p]["loc"][:3] for p in picks])
    neighbors = get_neighbors_within_radius(pos, radius)
    y = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        event_code = epochs.events[i, 2]
        if event_code in [11, 12, 13, 19]:
            y[i] = 1
        elif event_code in [21, 22, 23, 29]:
            y[i] = 2
        else:
            raise ValueError(f"未知的event_code: {event_code}")
    n_splits = ES.n_splits if hasattr(ES, "n_splits") else 20
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=12345)
    print(
        f"{meg_type} searchlight decoding: {sub}, radius={radius}, n_chans={n_chans}, n_times={n_times}"
    )

    def decode_one_channel(c):
        idx = neighbors[c]
        if len(idx) < 2:
            return np.full((n_splits, n_times), np.nan)
        scores_c = np.full((n_splits, n_times), np.nan)
        for t in range(n_times):
            X = data[:, idx, t]
            pipeline = make_pipeline(
                StandardScaler(),
                LinearSVC(
                    penalty="l1",
                    C=1,
                    dual=False,
                    max_iter=int(1e4),
                    tol=1e-4,
                    random_state=12345,
                    class_weight="balanced",
                ),
            )
            try:
                res = cross_validate(
                    pipeline,
                    X,
                    y,
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=1,  # 只在最外层并行
                    verbose=0,
                )
                scores_c[:, t] = res["test_score"]
            except Exception as e:
                print(f"Chan {c}, time {t} decoding failed: {e}")
        return scores_c

    results = Parallel(n_jobs=n_jobs)(
        delayed(decode_one_channel)(c) for c in tqdm(range(n_chans))
    )
    scores = np.stack(results, axis=1)  # (n_splits, n_chans, n_times)

    # 保存
    sorted_data_dir = get_sorted_data_dir()
    res_dir = os.path.join(sorted_data_dir, "within_subject_sensor_level_decoding", sub)
    os.makedirs(res_dir, exist_ok=True)

    fname_base = os.path.join(
        res_dir, f"sensor_searchlight_{meg_type}_{epochs_type}_{sub}"
    )
    np.save(fname_base + "_scores.npy", scores)
    np.save(fname_base + "_ch_names.npy", ch_names)
    np.save(fname_base + "_pos.npy", pos)
    print(f"{meg_type} searchlight decoding results saved to {fname_base}_*.npy")
    print(f"scores shape: {scores.shape}")


def sensor_searchlight_decoding_by_meg_type_sig_test(
    sub, epochs_type, meg_type, n_perm=1000, n_jobs=8, baseline=0.5
):
    """
    对sensor searchlight decoding的scores做显著性检测，流程参考s05中的scores_sig_test
    使用mne.stats.spatio_temporal_cluster_1samp_test进行空间-时间聚类置换检验。
    该函数对每个通道（空间）和每个时间点的AUC分数进行单样本t检验，并通过置换法对聚类进行多重比较校正。
    输入:
        X: (n_splits, n_chans, n_times) 例如交叉验证分数
        n_permutations: 置换次数
        tail: 1表示单尾检验（均值>0）
        n_jobs: 并行数
        baseline: 检验的基线值（如0.5）
    输出:
        T_obs: t统计量 (n_chans, n_times)
        clusters: 聚类掩码列表，每个聚类是布尔数组 (n_chans, n_times)
        cluster_p_values: 每个聚类的p值
        H0: 置换分布的最大聚类统计量
    """
    from mne.stats import spatio_temporal_cluster_1samp_test

    print(
        f"Running significance test for {sub} {epochs_type} {meg_type} with baseline={baseline}"
    )

    # 读取scores
    sorted_data_dir = get_sorted_data_dir()
    res_dir = os.path.join(sorted_data_dir, "within_subject_sensor_level_decoding", sub)
    fname_base = os.path.join(
        res_dir, f"sensor_searchlight_{meg_type}_{epochs_type}_{sub}"
    )
    scores_path = fname_base + "_scores.npy"
    if not os.path.exists(scores_path):
        print(f"未找到文件: {scores_path}")
        return
    scores = np.load(scores_path)  # (n_splits, n_chans, n_times)
    # 对每个通道每个时间点，检验AUC是否显著大于baseline（单尾）
    X = scores - baseline  # shape: (n_splits, n_chans, n_times)
    # 使用spatio_temporal_cluster_1samp_test进行空间-时间聚类置换检验
    T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
        X,
        n_permutations=n_perm,
        tail=1,
        out_type="mask",
        n_jobs=n_jobs,
        seed=42,
    )
    # 保存结果
    out_dir = os.path.join(res_dir, "sig_test")
    os.makedirs(out_dir, exist_ok=True)
    fname_out_base = os.path.join(
        out_dir, f"sensor_searchlight_{meg_type}_{epochs_type}_{sub}"
    )
    np.save(fname_out_base + f"_baseline_{baseline}_scores_sig_test_T_obs.npy", T_obs)
    np.save(
        fname_out_base + f"_baseline_{baseline}_scores_sig_test_clusters.npy",
        clusters,
        allow_pickle=True,
    )
    np.save(
        fname_out_base + f"_baseline_{baseline}_scores_sig_test_clusters_p.npy",
        cluster_p_values,
    )
    np.save(fname_out_base + f"_baseline_{baseline}_scores_sig_test_H0.npy", H0)
    print(f"Significance test finished for {sub} {epochs_type} {meg_type}.")
    print(f"clusters_p < 0.05 数量: {np.sum(cluster_p_values < 0.05)}")


def check_mag_grad_positions(sub, epochs_type):
    """
    检查mag通道(102)和grad通道(204)的位置是否一致
    """
    working_dir = f"../../data/MEG/{sub}/cleaned"
    fname_epochs = os.path.join(working_dir, f"{epochs_type}_{sub}-epo.fif")
    epochs = mne.read_epochs(fname_epochs, preload=True)
    picks_mag = mne.pick_types(epochs.info, meg="mag", eeg=False)
    picks_grad = mne.pick_types(epochs.info, meg="grad", eeg=False)
    pos_mag = np.array([epochs.info["chs"][p]["loc"][:3] for p in picks_mag])
    pos_grad = np.array([epochs.info["chs"][p]["loc"][:3] for p in picks_grad])
    # 对于每个mag通道，检查在grad通道中有多少个位置完全一致
    match_counts = []
    for i, pm in enumerate(pos_mag):
        matches = np.sum(np.all(np.isclose(pos_grad, pm, atol=1e-8), axis=1))
        match_counts.append(matches)
    print(f"每个mag通道在grad通道中找到的相同位置数量: {match_counts}")
    if all(c == 2 for c in match_counts):
        print("每个mag通道都能在grad通道中找到2个相同位置的grad通道。")
    else:
        print("注意：有mag通道无法在grad通道中找到2个对应位置！")
    return match_counts


# 主程序
# 循环subs进行解码
if __name__ == "__main__":
    # from settings import sub, epochs_type, meg_type, n_perm

    # sensor_searchlight_decoding_by_type_sig_test(
    #     sub=sub,
    #     epochs_type=epochs_type,
    #     meg_type=meg_type,
    # )
    # from settings import auc_baseline

    def process_one_train(sub, epochs_type, meg_type):
        sensor_searchlight_decoding_by_meg_type(
            sub=sub,
            epochs_type=epochs_type,
            meg_type=meg_type,
            n_jobs=1,  # 只用单进程，避免嵌套并行
        )

    train_tasks = [
        (sub, epochs_type, meg_type)
        for sub in subs
        for epochs_type in epochs_types
        for meg_type in meg_types
    ]
    Parallel(n_jobs=4)(
        delayed(process_one_train)(sub, epochs_type, meg_type)
        for sub, epochs_type, meg_type in train_tasks
    )

    def process_one_sig_test(sub, epochs_type, meg_type, auc_baseline):
        sensor_searchlight_decoding_by_meg_type_sig_test(
            sub=sub,
            epochs_type=epochs_type,
            meg_type=meg_type,
            n_perm=n_perm,
            n_jobs=1,  # 只用单进程，避免嵌套并行
            baseline=auc_baseline,
        )

    sig_test_tasks = [
        (sub, epochs_type, meg_type, auc_baseline)
        for sub in subs
        for epochs_type in epochs_types
        for meg_type in meg_types
        for auc_baseline in [0.5, 0.51, 0.52, 0.53]
    ]
    Parallel(n_jobs=4)(
        delayed(process_one_sig_test)(sub, epochs_type, meg_type, auc_baseline)
        for sub, epochs_type, meg_type, auc_baseline in sig_test_tasks
    )

    # pass

    # n_jobs = 8
    # for sub in subs:
    #     for epochs_type in epochs_types:
    #         print(f"Processing {sub} with epochs type {epochs_type}")
    #         print(
    #             f"Running sensor searchlight == mag == decoding for {sub} with epochs type {epochs_type}"
    #         )
    #         sensor_searchlight_decoding_by_type(
    #             sub, epochs_type=epochs_type, radius=0.04, n_jobs=n_jobs, meg_type="mag"
    #         )
    #         print(
    #             f"Running sensor searchlight == grad == decoding for {sub} with epochs type {epochs_type}"
    #         )
    #         sensor_searchlight_decoding_by_type(
    #             sub,
    #             epochs_type=epochs_type,
    #             radius=0.04,
    #             n_jobs=n_jobs,
    #             meg_type="grad",
    #         )
