import mne
import numpy as np
import os

# 判断当前系统是win还是linux
if os.name == "nt":
    root_dir = r"d:\Desktop\unconfeat\tmp_ws"
else:
    root_dir = "/home/MeiNing_03/workspace/unconfeat/unconfeatMEG"

# print(f"root_dir: {root_dir}")


# root_dir = "/home/MeiNing_03/workspace/unconfeat/unconfeatMEG"
# root_dir = r"d:\Desktop\unconfeat\tmp_ws"
scripts_dir = os.path.join(root_dir, "scripts/00_main_exp_analysis")
all_data_dir = os.path.join(root_dir, "data")
meg_data_dir = os.path.join(all_data_dir, "MEG")
mri_data_dir = os.path.join(all_data_dir, "MRI")
sorted_data_dir = os.path.join(scripts_dir, "z_sorted_data")
sorted_res_dir = os.path.join(scripts_dir, "z_sorted_results")


def get_sorted_res_dir():
    return sorted_res_dir


def get_sub_working_dir(sub):
    working_dir = os.path.abspath(os.path.join(meg_data_dir, sub, "cleaned"))
    return working_dir


def get_sub_subjects_dir(sub):
    subjects_dir = os.path.abspath(os.path.join(mri_data_dir, sub))
    return subjects_dir


def get_sub_trans_dir(sub):
    subjects_dir = get_sub_subjects_dir(sub)
    trans_dir = os.path.abspath(os.path.join(subjects_dir, "trans"))
    return trans_dir


def get_sorted_data_dir():
    return sorted_data_dir


def get_processing_data_sub_dir(sub):
    processing_data_sub_dir = os.path.join(sorted_data_dir, "processing_data", sub)
    return processing_data_sub_dir


def get_personal_invert_to_mni152_utils_dir(sub):
    personal_invert_to_mni152_utils_dir = os.path.join(
        sorted_data_dir, "personal_invert_to_mni152_utils", sub
    )
    return personal_invert_to_mni152_utils_dir


def get_fs_vol_2_src_path():
    sorted_data_dir = get_sorted_data_dir()
    out_dir = os.path.join(sorted_data_dir, "processing_data", "fsaverage")
    src_path = os.path.join(out_dir, "fsaverage-vol-2-src.fif")
    return src_path


def get_fs_vol_pos_src_path(pos):
    sorted_data_dir = get_sorted_data_dir()
    out_dir = os.path.join(sorted_data_dir, "processing_data", "fsaverage")
    src_path = os.path.join(out_dir, f"fsaverage-vol-{int(pos)}-src.fif")
    return src_path


def get_fs_surf_src_path(spacing="ico5"):
    out_dir = os.path.join(get_sorted_data_dir(), "processing_data", "fsaverage")
    src_path = os.path.join(out_dir, f"fsaverage-surf-{spacing}-src.fif")
    return src_path


def get_fs_subjects_dir():
    return get_sub_subjects_dir(subs[0])


epochs_type = "epochs_conscious"
annot_type = "aparc.a2009s"
n_perm = 10000
n_jobs = -1  # 并行处理的线程数
is_refit = True  # 是否重新计算
sub = "subject9"
meg_type = "mag"  # 可选 "mag" 或 "grad"
auc_baseline = 0.52  # AUC基准值，用于计算显著性阈值

tmin = -0.1
tmax = 0.5


# spacing参数常用选项及点数（以fsaverage为例）:
# 'oct5' 约10242个点/半球
# 'oct6' 约4098个点/半球
# 'oct7' 约16384个点/半球
# 'ico2' 约162个点/半球
# 'ico3' 约642个点/半球
# 'ico4' 约2562个点/半球
# 'ico5' 约10242个点/半球
# 'ico6' 约40962个点/半球
src_spacings = {"verylow": "ico3", "low": "ico4", "high": "oct6"}  # 可选更低分辨率

# 需要循环的epochs_type列表
epochs_types = [
    "epochs_unconscious",
    "epochs_conscious",
    # "epochs_glimpse",
]

annot_types = [
    "aparc.a2009s",
]

subs = [
    "subject9",
    "subject12",
    "subject13",
    "subject14",
    "subject15",
    "subject16",
    "subject17",
]

meg_types = [
    "mag",
    "grad",
]

baselines = [0.50, 0.51, 0.52, 0.53]

corss_sub_sig_baseline = 0.5
cross_sub_auc_max_limit = 0.90
cross_sub_auc_min_limit = 0.20


within_sub_sig_baseline = 0.52


# 排除的组合条件
# exclude_combinations = [
#     {
#         "sub": "subject9",
#         "epochs_type": "epochs_unconscious",
#     }
# ]
def get_masked_epochs(sub, epochs_type, meg_type, baseline):
    # 读取需要处理的epochs
    processing_data_dir = get_processing_data_sub_dir(sub)
    fname_epochs = os.path.join(
        processing_data_dir,
        f"auc_masked_baseline_{baseline}_{epochs_type}_{sub}_picked_{meg_type}-epo.fif",
    )
    epochs = mne.read_epochs(fname_epochs, preload=True)

    return epochs


def get_masked_epochs_stcs(
    sub, epochs_type, meg_type, baseline, is_vol=True, src_spacing_dec="high"
):
    # 读取需要处理的epochs
    processing_data_dir = get_processing_data_sub_dir(sub)
    fname_epochs = os.path.join(
        processing_data_dir,
        f"auc_masked_baseline_{baseline}_{epochs_type}_{sub}_picked_{meg_type}-epo.fif",
    )
    epochs = mne.read_epochs(fname_epochs, preload=True)

    # 读取filters
    fname_surf_filter = os.path.join(
        processing_data_dir,
        f"LCMV_filter_spacing_{meg_type}_{src_spacing_dec}_{sub}-lcmv.h5",
    )
    fname_vol_filter = os.path.join(
        processing_data_dir,
        f"LCMV_volume_filter_spacing_{meg_type}_{src_spacing_dec}_{sub}-lcmv.h5",
    )
    if is_vol:
        fname_filter = fname_vol_filter
    else:
        fname_filter = fname_surf_filter

    filter = mne.beamformer.read_beamformer(fname_filter)

    stcs = mne.beamformer.apply_lcmv_epochs(epochs, filter)

    return stcs


def get_average_masked_epochs_stcs(
    sub, epochs_type, meg_type, baseline, is_vol=True, src_spacing_dec="high"
):
    # 读取需要处理的epochs
    processing_data_dir = get_processing_data_sub_dir(sub)
    fname_epochs = os.path.join(
        processing_data_dir,
        f"auc_masked_baseline_{baseline}_{epochs_type}_{sub}_picked_{meg_type}-epo.fif",
    )
    epochs = mne.read_epochs(fname_epochs, preload=True)
    epochs = epochs.average()
    # 读取filters
    fname_surf_filter = os.path.join(
        processing_data_dir,
        f"LCMV_filter_spacing_{meg_type}_{src_spacing_dec}_{sub}-lcmv.h5",
    )
    fname_vol_filter = os.path.join(
        processing_data_dir,
        f"LCMV_volume_filter_spacing_{meg_type}_{src_spacing_dec}_{sub}-lcmv.h5",
    )
    if is_vol:
        fname_filter = fname_vol_filter
    else:
        fname_filter = fname_surf_filter

    filter = mne.beamformer.read_beamformer(fname_filter)

    stcs = mne.beamformer.apply_lcmv_epochs(epochs, filter)

    return stcs


def get_cross_sub_all_masked_auc_epochs(epochs_type, meg_type):
    data_dir = get_sorted_data_dir()
    save_epochs_dir = os.path.join(
        data_dir,
        "cross_subject_sensor_level_decoding_masked_auc_ac_epochs",
        f"{meg_type}_{epochs_type}",
        "all_subs_as_one",
    )

    fname_epochs = os.path.join(
        save_epochs_dir, f"all_subs_as_one_{epochs_type}_{meg_type}_epochs-epo.fif"
    )

    # fname_evoked = os.path.join(
    #     save_epochs_dir, f"all_subs_as_one_{epochs_type}_{meg_type}_evoked-epo.fif"
    # )

    epochs = mne.read_epochs(fname=fname_epochs, preload=True)
    # evoked = mne.read_evokeds(fname_evoked, preload=True)

    return epochs


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def get_decoding_pipeline():
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

    return pipeline


def get_cross_sub_searchlight_sig_test_mask(epochs_type, meg_type):
    save_dir = os.path.join(
        get_sorted_data_dir(),
        "cross_subject_sensor_level_decoding_sig_test",
    )
    fname_base = os.path.join(save_dir, f"sig_test_{meg_type}_{epochs_type}")
    fname_sig_mask = fname_base + f"_sig_mask.npy"
    sig_mask = np.load(fname_sig_mask)
    return sig_mask


def get_cross_sub_searchlight_sig_test_masked_auc_list(
    sub_test, subs, epochs_type, meg_type
):
    save_dir = os.path.join(
        get_sorted_data_dir(),
        "cross_subject_sensor_level_decoding_masked_auc",
        f"{epochs_type}_{meg_type}",
    )
    masked_auc_dir = os.path.join(save_dir, sub_test)

    masked_auc_list = []

    for sub_train in subs:
        if sub_train == sub_test:
            continue
        fname_masked_auc = os.path.join(
            masked_auc_dir,
            f"cross_sub_searchlight_{meg_type}_{epochs_type}_test-{sub_test}_train-{sub_train}_auc.npy",
        )
        masked_auc = np.load(fname_masked_auc)
        masked_auc_list.append(masked_auc)

    return masked_auc_list


def get_meg_type_picked_epochs(sub, epochs_type, meg_type):
    processing_data_dir = get_processing_data_sub_dir(sub)
    fname_picked_epochs = os.path.join(
        processing_data_dir,
        f"{epochs_type}_{sub}_picked_{meg_type}-epo.fif",
    )
    epochs = mne.read_epochs(fname_picked_epochs, preload=True)
    return epochs


def get_cross_sub_masked_epochs_stcs(
    sub_test, epochs_type, meg_type, is_vol=True, src_spacing_dec="high"
):
    # 读取需要处理的epochs
    processing_data_dir = get_processing_data_sub_dir(sub_test)
    data_dir = get_sorted_data_dir()
    save_epochs_dir = os.path.join(
        data_dir,
        "cross_subject_sensor_level_decoding_masked_auc_ac_epochs",
        f"{meg_type}_{epochs_type}",
        sub_test,
    )
    fname_epochs = os.path.join(
        save_epochs_dir,
        f"test-{sub_test}_{epochs_type}_{meg_type}-epo.fif",
    )
    epochs = mne.read_epochs(fname_epochs, preload=True)

    # 读取filters
    fname_filter = os.path.join(
        processing_data_dir,
        f"LCMV_filter_spacing_{meg_type}_{src_spacing_dec}_{sub_test}-lcmv.h5",
    )
    if is_vol:
        fname_filter = os.path.join(
            processing_data_dir,
            f"LCMV_volume_filter_spacing_{meg_type}_{src_spacing_dec}_{sub_test}-lcmv.h5",
        )

    filter = mne.beamformer.read_beamformer(fname_filter)

    stcs = mne.beamformer.apply_lcmv_epochs(epochs, filter)

    return stcs


def get_subject_src(sub, is_vol=True, src_spacing_dec="high"):
    out_dir = get_processing_data_sub_dir(sub)

    fname_src = os.path.join(
        out_dir, f"source_space_spacing_{src_spacing_dec}_{sub}-src.fif"
    )

    if is_vol:
        fname_src = os.path.join(
            out_dir,
            f"volume_source_space_spacing_{src_spacing_dec}_{sub}-src.fif",
        )

    src = mne.read_source_spaces(fname_src)
    return src


from pathlib import Path

# FSL 路径
FSLDIR = os.path.abspath("/opt/fox_cloud/share/app/imaging/fsl")
mni152_brain = os.path.join(FSLDIR, "data/standard/MNI152_T1_2mm_brain.nii.gz")
mni152_mask = os.path.join(FSLDIR, "data/standard/MNI152_T1_2mm_brain_mask.nii.gz")


def get_sig_time_ranges(epochs_type, meg_type, is_print=False):
    # 1. 读取 mask (n_chs, 121)
    sig_mask = get_cross_sub_searchlight_sig_test_mask(epochs_type, meg_type)

    n_times = sig_mask.shape[1]

    # 2. 定义时间轴
    times = np.arange(-100, 501, 5)  # 121 个时间点

    # 3. 找到连续 True 的片段
    roi_time_ranges = []
    start = None
    end = None

    for t in range(n_times):
        if sig_mask[:, t].any():
            if start is None:
                start = t
                end = t
            else:
                end = t
        else:
            if start is not None and end is not None:
                roi_time_ranges.append((start, end))
                start = None
                end = None

    if start is not None and end is not None:
        roi_time_ranges.append((start, end))

    if is_print:
        for start, end in roi_time_ranges:
            print(f"start: {times[start]}, end: {times[end]}")

    return roi_time_ranges


def get_all_conditions_epochs_data_max_min():
    mag_max = None
    mag_min = None
    grad_max = None
    grad_min = None
    for sub in subs:
        for epochs_type in epochs_types:
            for meg_type in meg_types:
                meg_type_picked_epochs = get_meg_type_picked_epochs(
                    sub,
                    epochs_type,
                    meg_type,
                )
                # 最大值
                if meg_type == "mag":
                    if mag_max is None:
                        mag_max = meg_type_picked_epochs.get_data().max()
                    else:
                        mag_max = max(mag_max, meg_type_picked_epochs.get_data().max())
                # 最小值
                if meg_type == "mag":
                    if mag_min is None:
                        mag_min = meg_type_picked_epochs.get_data().min()
                    else:
                        mag_min = min(mag_min, meg_type_picked_epochs.get_data().min())

                # 梯度最大值
                if meg_type == "grad":
                    if grad_max is None:
                        grad_max = meg_type_picked_epochs.get_data().max()
                    else:
                        grad_max = max(
                            grad_max, meg_type_picked_epochs.get_data().max()
                        )
                # 梯度最小值
                if meg_type == "grad":
                    if grad_min is None:
                        grad_min = meg_type_picked_epochs.get_data().min()
                    else:
                        grad_min = min(
                            grad_min, meg_type_picked_epochs.get_data().min()
                        )

    return {
        "mag_max": mag_max,
        "mag_min": mag_min,
        "grad_max": grad_max,
        "grad_min": grad_min,
    }
