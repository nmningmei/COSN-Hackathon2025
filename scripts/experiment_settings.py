# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:31:51 2024

@author: zhaojiahui
"""

import numpy as np
np.bool = np.bool_
# subjects information
subject_info = {
    'subject9':{
        'raw_ref':'../data/MEG/subject9/day1/unconsciousmeg_localizer.fif',
        'day1':{
                'empty_room_data':{'../data/MEG/subject9/day1/empty_room.fif': ''},
                'MEG_data':{#'../data/MEG/subject9/day1/unconsciousmeg_1_1.fif': '../data/behavioral/subject9/day1/9_fixed-probe_post-mask_2024-05-06_12h09.10.661trials.csv',
                            #'../data/MEG/subject9/day1/unconsciousmeg_1_2.fif': '../data/behavioral/subject9/day1/9_fixed-probe_post-mask_2024-05-06_12h31.30.578trials.csv',
                            '../data/MEG/subject9/day1/unconsciousmeg_2_1.fif': '../data/behavioral/subject9/day1/9_fixed-probe_post-mask_2024-05-06_12h44.03.509trials.csv',
                            #'../data/MEG/subject9/day1/unconsciousmeg_2_2.fif': '../data/behavioral/subject9/day1/9_fixed-probe_post-mask_2024-05-06_13h02.03.664trials.csv',
                            '../data/MEG/subject9/day1/unconsciousmeg_localizer.fif': '../data/behavioral/subject9/day1/9_localizer_2024-05-06_11h55.18.014trials.csv',}
                },
        'day2':{
                'empty_room_data':{'../data/MEG/subject9/day2/empty_room.fif': ''},
                'MEG_data':{
                            #'../data/MEG/subject9/day2/unconsciousmeg_3_1.fif': '../data/behavioral/subject9/day2/9_fixed-probe_post-mask_2024-05-07_11h30.37.850trials.csv',
                            '../data/MEG/subject9/day2/unconsciousmeg_3_2.fif': '../data/behavioral/subject9/day2/9_fixed-probe_post-mask_2024-05-07_11h44.35.338trials.csv',
                            #'../data/MEG/subject9/day2/unconsciousmeg_4_1.fif': '../data/behavioral/subject9/day2/9_fixed-probe_post-mask_2024-05-07_12h01.56.352trials.csv',
                            '../data/MEG/subject9/day2/unconsciousmeg_4_2.fif': '../data/behavioral/subject9/day2/9_fixed-probe_post-mask_2024-05-07_12h20.13.952trials.csv',
                            #'../data/MEG/subject9/day2/unconsciousmeg_5_1.fif': '../data/behavioral/subject9/day2/9_fixed-probe_post-mask_2024-05-07_12h37.02.454trials.csv',
                            #'../data/MEG/subject9/day2/unconsciousmeg_5_2.fif': '../data/behavioral/subject9/day2/9_fixed-probe_post-mask_2024-05-07_12h50.27.919trials.csv',
                            '../data/MEG/subject9/day2/unconsciousmeg_motor_localizer.fif': '../data/behavioral/subject9/day2/9_motor_localizer_2024-05-07_11h24.54.705trials.csv',}
                },
        'day3':{
                'empty_room_data':{'../data/MEG/subject9/day3/empty_room.fif': ''},
                'MEG_data':{'../data/MEG/subject9/day3/unconsciousmeg_1_1.fif': '../data/behavioral/subject9/day3/9_fixed-probe_post-mask_2024-06-25_14h26.14.848trials.csv',
                            '../data/MEG/subject9/day3/unconsciousmeg_1_2.fif': '../data/behavioral/subject9/day3/9_fixed-probe_post-mask_2024-06-25_14h40.06.769trials.csv',
                            '../data/MEG/subject9/day3/unconsciousmeg_2_2.fif': '../data/behavioral/subject9/day3/9_fixed-probe_post-mask_2024-06-25_14h52.42.731trials.csv',
                            '../data/MEG/subject9/day3/unconsciousmeg_3_1.fif': '../data/behavioral/subject9/day3/9_fixed-probe_post-mask_2024-06-25_15h10.28.394trials.csv',
                            '../data/MEG/subject9/day3/unconsciousmeg_4_1.fif': '../data/behavioral/subject9/day3/9_fixed-probe_post-mask_2024-06-25_15h22.42.675trials.csv',
                            '../data/MEG/subject9/day3/unconsciousmeg_5_1.fif': '../data/behavioral/subject9/day3/9_fixed-probe_post-mask_2024-06-25_15h35.36.068trials.csv',
                            '../data/MEG/subject9/day3/unconsciousmeg_5_2.fif': '../data/behavioral/subject9/day3/9_fixed-probe_post-mask_2024-06-25_15h49.58.591trials.csv',}
                }
                },


    'subject12':{
        # 'day1':{
        #         'empty_room_data':{'../data/MEG/subject12/day1/empty_room.fif': ''},
        #         'MEG_data':{'../data/MEG/subject12/day1/unconsciousmeg_1_1.fif': '../data/behavioral/subject12/day1/12_fixed-probe_post-mask_2024-05-07_15h56.37.237trials.csv',
        #                     '../data/MEG/subject12/day1/unconsciousmeg_1_2.fif': '../data/behavioral/subject12/day1/12_fixed-probe_post-mask_2024-05-07_16h10.43.393trials.csv',
        #                     '../data/MEG/subject12/day1/unconsciousmeg_2_1.fif': '../data/behavioral/subject12/day1/12_fixed-probe_post-mask_2024-05-07_16h31.10.357trials.csv',
        #                     '../data/MEG/subject12/day1/unconsciousmeg_2_2.fif': '../data/behavioral/subject12/day1/12_fixed-probe_post-mask_2024-05-07_16h59.15.283trials.csv',
        #                     '../data/MEG/subject12/day1/unconsciousmeg_3_1.fif': '../data/behavioral/subject12/day1/12_fixed-probe_post-mask_2024-05-07_16h45.50.398trials.csv',
        #                     '../data/MEG/subject12/day1/unconsciousmeg_localizer.fif': '../data/behavioral/subject12/day1/12_localizer_2024-05-07_15h44.19.252trials.csv',}
        #         },
        'raw_ref':'../data/MEG/subject12/day1/unconsciousmeg_localizer.fif',
        'day2':{
                'empty_room_data':{'../data/MEG/subject12/day2/empty_room.fif': ''},
                'MEG_data':{'../data/MEG/subject12/day2/unconsciousmeg_3_2.fif': '../data/behavioral/subject12/day2/12_fixed-probe_post-mask_2024-05-09_13h58.49.232trials.csv',
                            '../data/MEG/subject12/day2/unconsciousmeg_4_1.fif': '../data/behavioral/subject12/day2/12_fixed-probe_post-mask_2024-05-09_14h14.46.664trials.csv',
                            '../data/MEG/subject12/day2/unconsciousmeg_4_2.fif': '../data/behavioral/subject12/day2/12_fixed-probe_post-mask_2024-05-09_14h27.22.768trials.csv',
                            '../data/MEG/subject12/day2/unconsciousmeg_5_1.fif': '../data/behavioral/subject12/day2/12_fixed-probe_post-mask_2024-05-09_14h51.37.719trials.csv',
                            '../data/MEG/subject12/day2/unconsciousmeg_5_2.fif': '../data/behavioral/subject12/day2/12_fixed-probe_post-mask_2024-05-09_15h04.00.785trials.csv',
                            '../data/MEG/subject12/day2/unconsciousmeg_motor_localizer.fif': '../data/behavioral/subject12/day2/12_motor_localizer_2024-05-09_13h52.27.682trials.csv',}
                },
        'day1':{
                'empty_room_data':{'../data/MEG/subject12/day1/empty_room.fif': ''},
                'MEG_data':{'../data/MEG/subject12/day1/unconsciousmeg_1_1.fif': '../data/behavioral/subject12/day1/12_fixed-probe_post-mask_2024-05-21_12h31.04.077trials.csv',
                            '../data/MEG/subject12/day1/unconsciousmeg_1_2.fif': '../data/behavioral/subject12/day1/12_fixed-probe_post-mask_2024-05-21_12h44.19.564trials.csv',
                            '../data/MEG/subject12/day1/unconsciousmeg_2_1.fif': '../data/behavioral/subject12/day1/12_fixed-probe_post-mask_2024-05-21_12h57.49.104trials.csv',
                            '../data/MEG/subject12/day1/unconsciousmeg_2_2.fif': '../data/behavioral/subject12/day1/12_fixed-probe_post-mask_2024-05-21_13h11.21.282trials.csv',
                            '../data/MEG/subject12/day1/unconsciousmeg_3_1.fif': '../data/behavioral/subject12/day1/12_fixed-probe_post-mask_2024-05-21_13h26.44.407trials.csv',
                            '../data/MEG/subject12/day1/unconsciousmeg_localizer.fif': '../data/behavioral/subject12/day1/12_localizer_2024-05-07_15h44.19.252trials.csv',}
                }
                },
    
    'subject13':{
        'raw_ref':'../data/MEG/subject13/day1/unconsciousmeg_localizer.fif',
        'day1':{
                'empty_room_data':{'../data/MEG/subject13/day1/empty_room.fif': ''},
                'MEG_data':{'../data/MEG/subject13/day1/unconsciousmeg_1_1.fif': '../data/behavioral/subject13/day1/13_fixed-probe_post-mask_2024-05-09_09h45.39.570trials.csv',
                            '../data/MEG/subject13/day1/unconsciousmeg_1_2.fif': '../data/behavioral/subject13/day1/13_fixed-probe_post-mask_2024-05-09_09h59.14.664trials.csv',
                            '../data/MEG/subject13/day1/unconsciousmeg_2_1.fif': '../data/behavioral/subject13/day1/13_fixed-probe_post-mask_2024-05-09_10h13.37.795trials.csv',
                            '../data/MEG/subject13/day1/unconsciousmeg_2_2.fif': '../data/behavioral/subject13/day1/13_fixed-probe_post-mask_2024-05-09_10h29.06.282trials.csv',
                            '../data/MEG/subject13/day1/unconsciousmeg_3_1.fif': '../data/behavioral/subject13/day1/13_fixed-probe_post-mask_2024-05-09_10h41.22.807trials.csv',
                            '../data/MEG/subject13/day1/unconsciousmeg_localizer.fif': '../data/behavioral/subject13/day1/13_localizer_2024-05-09_09h32.23.402trials.csv',}
                },
        'day2':{
                'empty_room_data':{'../data/MEG/subject13/day2/empty_room.fif': ''},
                'MEG_data':{'../data/MEG/subject13/day2/unconsciousmeg_3_2.fif': '../data/behavioral/subject13/day2/13_fixed-probe_post-mask_2024-05-10_09h39.53.732trials.csv',
                            '../data/MEG/subject13/day2/unconsciousmeg_4_1.fif': '../data/behavioral/subject13/day2/13_fixed-probe_post-mask_2024-05-10_09h52.54.600trials.csv',
                            '../data/MEG/subject13/day2/unconsciousmeg_4_2.fif': '../data/behavioral/subject13/day2/13_fixed-probe_post-mask_2024-05-10_10h05.21.847trials.csv',
                            '../data/MEG/subject13/day2/unconsciousmeg_5_1.fif': '../data/behavioral/subject13/day2/13_fixed-probe_post-mask_2024-05-10_10h21.05.072trials.csv',
                            '../data/MEG/subject13/day2/unconsciousmeg_5_2.fif': '../data/behavioral/subject13/day2/13_fixed-probe_post-mask_2024-05-10_10h33.39.678trials.csv',
                            '../data/MEG/subject13/day2/unconsciousmeg_motor_localizer.fif': '../data/behavioral/subject13/day2/13_motor_localizer_2024-05-10_09h33.32.440trials.csv',}
                }        
                },

    'subject14':{
        'raw_ref':'../data/MEG/subject14/day1/unconsciousmeg_localizer.fif',
        'day1':{'empty_room_data':{'../data/MEG/subject14/day1/empty_room.fif': ''},
                'MEG_data':{'../data/MEG/subject14/day1/unconsciousmeg_1_1.fif': '../data/behavioral/subject14/day1/14_fixed-probe_post-mask_2024-05-09_15h53.45.113trials.csv',
                            '../data/MEG/subject14/day1/unconsciousmeg_1_2.fif': '../data/behavioral/subject14/day1/14_fixed-probe_post-mask_2024-05-09_16h06.00.858trials.csv',
                            '../data/MEG/subject14/day1/unconsciousmeg_2_1.fif': '../data/behavioral/subject14/day1/14_fixed-probe_post-mask_2024-05-09_16h21.00.240trials.csv',
                            '../data/MEG/subject14/day1/unconsciousmeg_2_2.fif': '../data/behavioral/subject14/day1/14_fixed-probe_post-mask_2024-05-09_16h38.51.861trials.csv',
                            '../data/MEG/subject14/day1/unconsciousmeg_3_1.fif': '../data/behavioral/subject14/day1/14_fixed-probe_post-mask_2024-05-09_16h51.12.299trials.csv',
                            '../data/MEG/subject14/day1/unconsciousmeg_localizer.fif': '../data/behavioral/subject14/day1/14_localizer_2024-05-09_15h32.40.578trials.csv',}
                },
        'day2':{
                'empty_room_data':{'../data/MEG/subject14/day2/empty_room.fif': ''},
                'MEG_data':{'../data/MEG/subject14/day2/unconsciousmeg_3_2.fif': '../data/behavioral/subject14/day2/14_fixed-probe_post-mask_2024-05-10_13h21.14.178trials.csv',
                            '../data/MEG/subject14/day2/unconsciousmeg_4_1.fif': '../data/behavioral/subject14/day2/14_fixed-probe_post-mask_2024-05-10_13h33.14.955trials.csv',
                            '../data/MEG/subject14/day2/unconsciousmeg_4_2.fif': '../data/behavioral/subject14/day2/14_fixed-probe_post-mask_2024-05-10_13h45.39.392trials.csv',
                            '../data/MEG/subject14/day2/unconsciousmeg_5_1.fif': '../data/behavioral/subject14/day2/14_fixed-probe_post-mask_2024-05-10_14h03.31.454trials.csv',
                            '../data/MEG/subject14/day2/unconsciousmeg_5_2.fif': '../data/behavioral/subject14/day2/14_fixed-probe_post-mask_2024-05-10_14h17.25.888trials.csv',
                            '../data/MEG/subject14/day2/unconsciousmeg_motor_localizer.fif': '../data/behavioral/subject14/day2/14_motor_localizer_2024-05-10_13h14.52.560trials.csv',}
                }
                },

    'subject15':{
        'raw_ref':'../data/MEG/subject15/day1/unconsciousmeg_localizer.fif',
        'day1':{
                'empty_room_data':{'../data/MEG/subject15/day1/empty_room.fif': ''},
                'MEG_data':{'../data/MEG/subject15/day1/unconsciousmeg_1_1.fif': '../data/behavioral/subject15/day1/15_fixed-probe_post-mask_2024-05-09_11h33.01.269trials.csv',
                            '../data/MEG/subject15/day1/unconsciousmeg_1_2.fif': '../data/behavioral/subject15/day1/15_fixed-probe_post-mask_2024-05-09_11h44.46.994trials.csv',
                            '../data/MEG/subject15/day1/unconsciousmeg_2_1.fif': '../data/behavioral/subject15/day1/15_fixed-probe_post-mask_2024-05-09_11h56.57.320trials.csv',
                            '../data/MEG/subject15/day1/unconsciousmeg_2_2.fif': '../data/behavioral/subject15/day1/15_fixed-probe_post-mask_2024-05-09_12h14.26.562trials.csv',
                            '../data/MEG/subject15/day1/unconsciousmeg_3_1.fif': '../data/behavioral/subject15/day1/15_fixed-probe_post-mask_2024-05-09_12h28.25.844trials.csv',
                            '../data/MEG/subject15/day1/unconsciousmeg_localizer.fif': '../data/behavioral/subject15/day1/15_localizer_2024-05-09_11h19.42.611trials.csv',}
                },
        'day2':{
                'empty_room_data':{'../data/MEG/subject15/day2/empty_room.fif': ''},
                'MEG_data':{'../data/MEG/subject15/day2/unconsciousmeg_3_2.fif': '../data/behavioral/subject15/day2/15_fixed-probe_post-mask_2024-05-10_11h24.38.215trials.csv',
                            '../data/MEG/subject15/day2/unconsciousmeg_4_1.fif': '../data/behavioral/subject15/day2/15_fixed-probe_post-mask_2024-05-10_11h37.33.221trials.csv',
                            '../data/MEG/subject15/day2/unconsciousmeg_4_2.fif': '../data/behavioral/subject15/day2/15_fixed-probe_post-mask_2024-05-10_11h50.46.380trials.csv',
                            '../data/MEG/subject15/day2/unconsciousmeg_5_1.fif': '../data/behavioral/subject15/day2/15_fixed-probe_post-mask_2024-05-10_12h07.26.321trials.csv',
                            '../data/MEG/subject15/day2/unconsciousmeg_5_2.fif': '../data/behavioral/subject15/day2/15_fixed-probe_post-mask_2024-05-10_12h22.58.839trials.csv',
                            '../data/MEG/subject15/day2/unconsciousmeg_motor_localizer.fif': '../data/behavioral/subject15/day2/15_motor_localizer_2024-05-10_11h16.45.384trials.csv',}
                }
                },

    'subject16':{
        'raw_ref':'../data/MEG/subject16/day1/unconsciousmeg_localizer.fif',
        'day1':{
                'empty_room_data':{'../data/MEG/subject16/day1/empty_room.fif': ''},
                'MEG_data':{'../data/MEG/subject16/day1/unconsciousmeg_1_1.fif': '../data/behavioral/subject16/day1/16_fixed-probe_post-mask_2024-05-21_11h07.17.645trials.csv',
                            '../data/MEG/subject16/day1/unconsciousmeg_1_2.fif': '../data/behavioral/subject16/day1/16_fixed-probe_post-mask_2024-05-21_11h19.37.440trials.csv',
                            '../data/MEG/subject16/day1/unconsciousmeg_2_1.fif': '../data/behavioral/subject16/day1/16_fixed-probe_post-mask_2024-05-21_11h33.26.459trials.csv',
                            '../data/MEG/subject16/day1/unconsciousmeg_2_2.fif': '../data/behavioral/subject16/day1/16_fixed-probe_post-mask_2024-05-21_11h49.52.417trials.csv',
                            '../data/MEG/subject16/day1/unconsciousmeg_3_1.fif': '../data/behavioral/subject16/day1/16_fixed-probe_post-mask_2024-05-21_12h02.05.735trials.csv',
                            '../data/MEG/subject16/day1/unconsciousmeg_localizer.fif': '../data/behavioral/subject16/day1/16_localizer_2024-05-21_10h47.40.148trials.csv',}
                },
        'day2':{
                'empty_room_data':{'../data/MEG/subject16/day2/empty_room.fif': ''},
                'MEG_data':{'../data/MEG/subject16/day2/unconsciousmeg_3_2.fif': '../data/behavioral/subject16/day2/16_fixed-probe_post-mask_2024-05-22_10h48.21.583trials.csv',
                            '../data/MEG/subject16/day2/unconsciousmeg_4_1.fif': '../data/behavioral/subject16/day2/16_fixed-probe_post-mask_2024-05-22_11h01.22.898trials.csv',
                            '../data/MEG/subject16/day2/unconsciousmeg_4_2.fif': '../data/behavioral/subject16/day2/16_fixed-probe_post-mask_2024-05-22_11h23.11.085trials.csv',
                            '../data/MEG/subject16/day2/unconsciousmeg_5_1.fif': '../data/behavioral/subject16/day2/16_fixed-probe_post-mask_2024-05-22_11h38.46.141trials.csv',
                            '../data/MEG/subject16/day2/unconsciousmeg_5_2.fif': '../data/behavioral/subject16/day2/16_fixed-probe_post-mask_2024-05-22_11h52.01.057trials.csv',
                            '../data/MEG/subject16/day2/unconsciousmeg_motor_localizer.fif': '../data/behavioral/subject16/day2/16_motor_localizer_2024-05-22_10h41.43.521trials.csv',}
                }
                },

    'subject17':{
        'raw_ref':'../data/MEG/subject17/day1/unconsciousmeg_localizer.fif',
        'day1':{
                'empty_room_data':{'../data/MEG/subject17/day1/empty_room.fif': ''},
                'MEG_data':{'../data/MEG/subject17/day1/unconsciousmeg_1_1.fif': '../data/behavioral/subject17/day1/17_fixed-probe_post-mask_2024-05-22_16h47.10.076trials.csv',
                            '../data/MEG/subject17/day1/unconsciousmeg_1_2.fif': '../data/behavioral/subject17/day1/17_fixed-probe_post-mask_2024-05-22_16h59.20.475trials.csv',
                            '../data/MEG/subject17/day1/unconsciousmeg_localizer.fif': '../data/behavioral/subject17/day1/17_localizer_2024-05-22_16h34.12.910trials.csv',}
                },
        'day2':{
                'empty_room_data':{'../data/MEG/subject17/day2/empty_room.fif': ''},
                'MEG_data':{'../data/MEG/subject17/day2/unconsciousmeg_2_1.fif': '../data/behavioral/subject17/day2/17_fixed-probe_post-mask_2024-05-23_12h49.49.004trials.csv',
                            '../data/MEG/subject17/day2/unconsciousmeg_2_2.fif': '../data/behavioral/subject17/day2/17_fixed-probe_post-mask_2024-05-23_13h03.19.439trials.csv',
                            '../data/MEG/subject17/day2/unconsciousmeg_3_1.fif': '../data/behavioral/subject17/day2/17_fixed-probe_post-mask_2024-05-23_13h23.53.417trials.csv',
                            '../data/MEG/subject17/day2/unconsciousmeg_3_2.fif': '../data/behavioral/subject17/day2/17_fixed-probe_post-mask_2024-05-23_13h36.55.670trials.csv',
                            '../data/MEG/subject17/day2/unconsciousmeg_4_1.fif': '../data/behavioral/subject17/day2/17_fixed-probe_post-mask_2024-05-23_13h58.49.639trials.csv',
                            '../data/MEG/subject17/day2/unconsciousmeg_4_2.fif': '../data/behavioral/subject17/day2/17_fixed-probe_post-mask_2024-05-23_14h10.52.900trials.csv',
                            '../data/MEG/subject17/day2/unconsciousmeg_5_1.fif': '../data/behavioral/subject17/day2/17_fixed-probe_post-mask_2024-05-23_14h24.37.639trials.csv',
                            '../data/MEG/subject17/day2/unconsciousmeg_5_2.fif': '../data/behavioral/subject17/day2/17_fixed-probe_post-mask_2024-05-23_14h40.01.738trials.csv',
                            '../data/MEG/subject17/day2/unconsciousmeg_motor_localizer.fif': '../data/behavioral/subject17/day2/17_motor_localizer_2024-05-23_12h06.20.578trials.csv',}
                }
                },
}

# seeg subjects
## subject 1,4,5 has very skewed distribution of unconscious/conscious responses
## subject 6 has very skewed distribution of orientation responses
seeg_subjects = [2,3]

# preprocessing parameters
preMaxFiltered  = False
n_interpolates  = np.array([1, 4, 32])
consensus_percs = np.linspace(0, 1.0, 11)
notch_filter    = 50
l_freq = 1
h_freq = 40
tmin_preprocessing  = -1.0
baseline = (tmin_preprocessing,0)
tmax_preprocessing  = 1.0
dict_category       = {'Living_Things':1,'Nonliving_Things':2}

# epoch information
tmin                    = -0.1
tmax                    = 0.5
resample                = 200
fmin                    = l_freq
fmax                    = h_freq
freqs                   = np.arange(start = fmin,stop = fmax + 1,step = 3,) # assemble frequencies
freq_ranges             = list(zip(freqs[:-1],freqs[1:])) # make freq list of tuples
n_cycles                = freqs / 4.0
n_cycles[n_cycles > 10] = 10
times                   = np.arange(tmin,tmax + 1/resample,1/resample,)
# infer window spacing from the max freq and number of cycles to avoid gaps
window_spacing          = 10 / np.max(freqs) / 2.0
centered_w_times        = np.arange(tmin, tmax, window_spacing)[1:]
n_windows               = len(centered_w_times)

# folder names
folder_name_sensor_decoding     = 'temporal_decoding_sensor'
folder_name_time_freq_decoding  = 'decoding_per-time_per-frequence'
folder_name_source_decoding     = 'temporal_decoding_source'

folder_name_baseline_analysis   = 'baseline_analysis'

folder_name_decoding_cons       = 'decoding_consciousness_on_sensors'

# HPC settings
n_jobs  = 40 # core
verbose = True
node = 1
mem = 6

# linear model parameters
C = 10
n_splits = 20
cv_cali = 5
param_grid = {'calibratedclassifiercv__estimator__C':np.logspace(0,3,4),}

# statistical parameters
n_permutations = int(3e4)
threshold_tfce = dict(start=0, step=0.2)
min_samples = 3
time_bins = [[0.0,0.1],
             [0.1,0.2],
             [0.2,0.3],
             [0.3,0.4],
             [0.4,0.5],
             ]
p_sign = 0.001
log_p_sign = -np.log(p_sign)

# decoding_kwargs
pipeline_kwargs = dict(penalty = 'l1',
                       C = C,
                       dual = False,
                       max_iter = int(1e4),
                       tol = 1e-4,
                       cv_cali = cv_cali,
                       steps = ['vectorize','preprocessing'],
                       grid_search = None
                       )
decoding_kwargs = dict(C = C,
                       n_jobs = n_jobs,
                       verbose = verbose,
                       threshold_tfce = threshold_tfce,
                       n_permutations = n_permutations,
                       param_grid = param_grid,
                       )

# cross subejct decoding settings
cross_decoding_folder_name = 'cross_subject_decoding_time_frequence'
subject_pairs = []
for sub_source in subject_info.keys():
    for sub_target in subject_info.keys():
        if sub_source != sub_target:
            subject_pairs.append([sub_source,sub_target])

# resample cross subject decoding settings
resample_cross_decoding_folder_name = 'resample_cross_decoding_time_frequence'
n_samples = int(500)

# compute inverse solution
inverse_method = 'dSPM' # could choose MNE, sLORETA, or eLORETA
signal_to_noise_ratio = 3.0
if __name__ == "__main__":
    pass






