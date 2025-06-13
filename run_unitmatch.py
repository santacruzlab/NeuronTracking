# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 18:04:24 2025

@author: coleb
"""


run_GUI = False

root = r'C:\Users\coleb\Desktop\Santacruz Lab\Neuron Tracking\UnitMatch'

# Choose one of the following pairs
# sess1 = r'Data\Mouse1\AL032\2019-11-21\Probe0\1'
# sess2 = r'Data\Mouse1\AL032\2019-11-22\Probe0\1'

# sess1 = r'Data\Mouse2\JF067\2022-02-14\Probe0\4'
# sess2 = r'Data\Mouse2\JF067\2022-02-15\Probe0\5'

# sess1 = r'Data\Mouse3\AV008\2022-03-12\Probe0\IMRO_1'
# sess2 = r'Data\Mouse3\AV008\2022-03-13\Probe0\IMRO_1'

# sess1 = r'Data\Mouse4\EB019\2022-07-21\Probe0\1'
# sess2 = r'Data\Mouse4\EB019\2022-07-22\Probe0\1'

sess1 = r'Data\Mouse5\CB016\2021-09-28\Probe0\1'
sess2 = r'Data\Mouse5\CB016\2021-09-29\Probe0\1'






#%%
import sys
import os
sys.path.insert(1, os.path.join(root,r'UnitMatchPy'))
import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.utils as util
import UnitMatchPy.overlord as ov
import numpy as np
import matplotlib.pyplot as plt
import UnitMatchPy.save_utils as su
import UnitMatchPy.GUI as gui
import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy.default_params as default_params

#%%


# Get default parameters, can add your own before or after!
param = default_params.get_default_param()

# Give the paths to the KS directories for each session
# If you don't have a dir with channel_positions.npy etc look at the detailed example for supplying paths separately 
KS_dirs = [os.path.join(root,sess1),os.path.join(root,sess2)]


param['KS_dirs'] = KS_dirs
wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(KS_dirs)
param = util.get_probe_geometry(channel_pos[0], param)

#%%

# STEP 0 -- data preparation
# Read in data and select the good units and exact metadata
waveform, session_id, session_switch, within_session, good_units, param = util.load_good_waveforms(wave_paths, unit_label_paths, param, good_units_only = True) 

# param['peak_loc'] = #may need to set as a value if the peak location is NOT ~ half the spike width

# Create clus_info, contains all unit id/session related info
clus_info = {'good_units' : good_units, 'session_switch' : session_switch, 'session_id' : session_id, 
            'original_ids' : np.concatenate(good_units) }

# STEP 1
# Extract parameters from waveform
extracted_wave_properties = ov.extract_parameters(waveform, channel_pos, clus_info, param)

# STEP 2, 3, 4
# Extract metric scores
total_score, candidate_pairs, scores_to_include, predictors  = ov.extract_metric_scores(extracted_wave_properties, session_switch, within_session, param, niter  = 2)

# STEP 5
# Probability analysis
# Get prior probability of being a match
prior_match = 1 - (param['n_expected_matches'] / param['n_units']**2 ) # freedom of choose in prior prob
priors = np.array((prior_match, 1-prior_match))

# Construct distributions (kernels) for Naive Bayes Classifier
labels = candidate_pairs.astype(int)
cond = np.unique(labels)
score_vector = param['score_vector']
parameter_kernels = np.full((len(score_vector), len(scores_to_include), len(cond)), np.nan)

parameter_kernels = bf.get_parameter_kernels(scores_to_include, labels, cond, param, add_one = 1)

# Get probability of each pair of being a match
probability = bf.apply_naive_bayes(parameter_kernels, priors, predictors, param, cond)

output_prob_matrix = probability[:,1].reshape(param['n_units'],param['n_units'])

num_units = param['n_units']

print(f"Number of units: {num_units}")


# Evaluate match probabilities
util.evaluate_output(output_prob_matrix, param, within_session, session_switch, match_threshold = 0.75)

match_threshold = param['match_threshold']
# match_threshold = try different values here!

output_threshold = np.zeros_like(output_prob_matrix)
output_threshold[output_prob_matrix > match_threshold] = 1

#%% Graph

fig,ax=plt.subplots()
ax.imshow(total_score, cmap = 'Greys')
ax.hlines(num_units//2,0,num_units-1,'k')
ax.vlines(num_units//2,0,num_units-1,'k')
ax.set_ylabel(f'Second half\nDay 2{" "*24}Day 1')
ax.set_xlabel(f'First half\nDay 1{" "*24}Day 2')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Total Similarity Score')

fig,ax=plt.subplots()
ax.imshow(output_threshold, cmap = 'Greys')
ax.hlines(num_units//2,0,num_units-1,'k')
ax.vlines(num_units//2,0,num_units-1,'k')
ax.set_ylabel(f'Second half\nDay 2{" "*24}Day 1')
ax.set_xlabel(f'First half\nDay 1{" "*24}Day 2')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Matches')

#%% STEP 6
if run_GUI:
    
    # Format data for GUI
    amplitude = extracted_wave_properties['amplitude']
    spatial_decay = extracted_wave_properties['spatial_decay']
    avg_centroid = extracted_wave_properties['avg_centroid']
    avg_waveform = extracted_wave_properties['avg_waveform']
    avg_waveform_per_tp = extracted_wave_properties['avg_waveform_per_tp']
    wave_idx = extracted_wave_properties['good_wave_idxs']
    max_site = extracted_wave_properties['max_site']
    max_site_mean = extracted_wave_properties['max_site_mean']
    gui.process_info_for_GUI(output_prob_matrix, match_threshold, scores_to_include, total_score, amplitude, spatial_decay,
                             avg_centroid, avg_waveform, avg_waveform_per_tp, wave_idx, max_site, max_site_mean, 
                             waveform, within_session, channel_pos, clus_info, param)
    # Run GUI

    is_match, not_match, matches_GUI = gui.run_GUI()

    #this function has 2 mode 'And' 'Or', which returns a matches if they appear in both or one cv pair
    #then it will add all the matches selected as IsMaatch, then remove all matches in NotMatch
    # matches_curated = util.curate_matches(matches_GUI, is_match, not_match, mode = 'And')
    
    
    matches = np.argwhere(output_threshold == 1)
    UIDs = aid.assign_unique_id(output_prob_matrix, param, clus_info)
#%%
# save_dir = r'Path/to/save/directory'
# #NOTE - change to matches to matches_curated if done manual curation with the GUI
# su.save_to_output(save_dir, scores_to_include, matches # matches_curated
#                   , output_prob_matrix, avg_centroid, avg_waveform, avg_waveform_per_tp, max_site,
#                    total_score, output_threshold, clus_info, param, UIDs = UIDs, matches_curated = None, save_match_table = True)

