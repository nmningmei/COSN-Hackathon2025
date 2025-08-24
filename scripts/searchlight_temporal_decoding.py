import os
from utils import *
from glob import glob
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit


if __name__ == '__main__':
    subj = 9
    subject = 'subject' + str(subj)
    working_dir = os.path.join('../data/MEG', subject)
    working_data = np.sort(glob(os.path.join(working_dir, "cleaned", "unconscious-session*fif")))
    working_beha = np.sort(glob(os.path.join(working_dir, "cleaned", "unconscious-session*csv")))

    label_map = {
        "Nonliving_Things": 1,
        "Living_Things"   : 0,
    }

    maga_for_loop, epochs_unconscious, epochs_conscious, df_unconscious, df_conscious = load_data(working_data, working_beha)
    labels_unconscious = df_unconscious['category'].map(label_map).values
    labels_conscious = df_conscious['category'].map(label_map).values

    cv = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=12345)

    scores = Parallel(n_jobs=2, verbose=1, )(
        delayed(decode_within_sphere)(**{'epoch_data': epochs_unconscious.get_data()[:, sphere, time_point],
                                         'labels': labels_unconscious,
                                         'cv': cv, }) for sphere, time_point in maga_for_loop)
    results_dir = os.path.join('../results/', 'MEG', subject)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    np.save(os.path.join(results_dir, 'searchlight_scores.npy'), scores)

