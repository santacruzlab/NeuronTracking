# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:06:11 2025

@author: coleb
"""

import os
import re
import mat73
import numpy as np
import pandas as pd


def get_sessions(subj: str):
    #for sorting filenames
    
    if subj == 'Mouse1':
        sessions = [r"Mouse1\AL032\2019-11-21\Probe0\1",
              r"Mouse1\AL032\2019-11-22\Probe0\1"]
        
    if subj == 'Mouse2':
        sessions = [r"Mouse2\JF067\2022-02-14\Probe0\4",
              r"Mouse2\JF067\2022-02-15\Probe0\5"]
        
    if subj == 'Mouse3':
        sessions = [r"Mouse3\AV008\2022-03-12\Probe0\IMRO_1",
              r"Mouse3\AV008\2022-03-13\Probe0\IMRO_1"]
        
    if subj == 'Mouse4':
        sessions = [r"Mouse4\EB019\2022-07-21\Probe0\1",
              r"Mouse4\EB019\2022-07-22\Probe0\1"]
        
    if subj == 'Mouse5':
        sessions = [r"Mouse5\CB016\2021-09-28\Probe0\1",
              r"Mouse5\CB016\2021-09-29\Probe0\1"]
        
    return sessions


def parse_unit(full_unit_path: str):
    """
    Parse the unit path  >> "root\\Mouse1\\AL032\\2019-11-21\\Probe0\\1\\RawWaveforms\\Unit0_RawSpikes.npy"
        date >> 2019-11-21 >> 20191121
        unit_code >> 0
    """
    
    # Get unit code from base file name
    fname = os.path.basename(full_unit_path)
    pattern = r'Unit([0-9]*)_RawSpikes.npy'
    m = re.match(pattern, fname)
    unit_code = m.group(1)

    pattern = r'(\d{4})-(\d{2})-(\d{2})'
    m=re.search(pattern, fname)
    date=m[0].replace('-','')
    
    return date, unit_code



# def find_good_units(session_path):
#     
#     data = mat73.loadmat(os.path.join(session_path,'PreparedData.mat'))
#     
#     good_clusters_idxs = data['clusinfo']['Good_ID'].astype(bool) #get idxs of good clusters

#     good_clusters = data['clusinfo']['cluster_id'][good_clusters_idxs] #get good clusters
#     
#     good_clusters_chs = data['clusinfo']['ch'][good_clusters_idxs] #get the good clusters' chs
#     
# #     print(len(good_clusters_idxs))
# #     print(len(data['clusinfo']['ch']))
# #     print(good_clusters_chs)

#     return good_clusters, good_clusters_chs.astype(int)



def find_good_units(session_path):
    
    df = pd.read_csv(os.path.join(session_path,'cluster_group.tsv'), sep='\t', skiprows = 0)
    unit_label = df.values
    data = mat73.loadmat(os.path.join(session_path,'PreparedData.mat'))

    
    tmp_idx = np.argwhere(unit_label[:,1] == 'good')
    
    good_clusters_chs = data['clusinfo']['ch'][tmp_idx] #get the good clusters' chs

    good_clusters = unit_label[tmp_idx, 0]

    return good_clusters, good_clusters_chs.astype(int)
    

# def find_maximal_wf(wfs):
#     
#     #shape of wfs is (time x recording sites x halves of recording session)
#     
#     #first, avg together the two halves of session
#     wfs = np.mean(wfs,axis=-1)
#     
#     wf_max = np.max(abs(wfs),axis=0) #get max voltage value along time axis for each waveform
#     max_ch = np.argmax(wf_max)
#     max_wf = wfs[:,max_ch] #get wf from site which has the largest max voltage
#     
#     return max_wf,max_ch


def make_metadata(unit_codes, dates, chs):    
    
    result = pd.DataFrame(columns=['date','unit_code','channel'])
    
    assert len(unit_codes) == len(chs) == len(dates)
            
    for i in range(len(unit_codes)):
        
        result.at[i, 'date'] =  dates[i]
        result.at[i, 'unit_code'] =  unit_codes[i]
        result.at[i, 'channel'] =  chs[i]
    
    return result


def save_data(subj, metadata: pd.DataFrame, waveforms: np.ndarray) -> None:
   
    if not os.path.exists(os.path.join(save_out_folder, subj)):
         os.makedirs(os.path.join(save_out_folder, subj))
        
    metadata.to_csv(os.path.join(save_out_folder, subj, 'waveforms_metadata.csv'), index=False)
    np.save(os.path.join(save_out_folder, subj, 'waveforms.npy'), waveforms)
    
    return 


def read_data(subj):

    waveforms = []
    unit_codes = []
    dates = []
    chs_ = []
    
    for session in range(num_sessions):
        
        session_path = os.path.join(root,get_sessions(subj)[session]) #get path to folder which has all waveform files for all units
        
        wf_path = os.path.join(session_path,'RawWaveforms')
        units,chs = find_good_units(session_path)
        
        for session_half in range(2):
            
            for unit,ch in zip(units,chs):
    
                full_unit_path = os.path.join(wf_path,f'Unit{unit[0]}_RawSpikes.npy') #get path to current good unit waveform file
                wf=np.load(full_unit_path)
                waveforms.append(wf[:,ch,session_half])
                
                date, unit_code = parse_unit(full_unit_path)
                dates.append(date + f'-{session_half}') #append date with session half labeling
                chs_.append(ch)
                unit_codes.append(unit_code)
            
    return waveforms, unit_codes, dates, chs_


def main(subjects):
    
    for subj in subjects:
        
        waveforms, unit_codes, dates, chs = read_data(subj)
        metadata = make_metadata(unit_codes, dates, chs)
        save_data(subj, metadata, waveforms)














subjects = ["Mouse1","Mouse2","Mouse3","Mouse4","Mouse5"]
num_sessions = 2

root = r"C:\Users\coleb\Desktop\Santacruz Lab\Neuron Tracking\UnitMatch\Data"

save_out_folder = r"C:\Users\coleb\Desktop\Santacruz Lab\Neuron Tracking\Cole Processed Data"

main(subjects)

        


    


xx
#%% To check waveforms
import pandas as pd
import numpy as np
metadata = pd.read_csv(r"C:\Users\coleb\Desktop\Santacruz Lab\Neuron Tracking\Cole Processed Data\Mouse1\waveforms_metadata.csv")
a=np.load(r"C:\Users\coleb\Desktop\Santacruz Lab\Neuron Tracking\Cole Processed Data\Mouse1\waveforms.npy")

num_plots = 25

from matplotlib import pyplot as plt
for i in range(num_plots):
    fig,ax=plt.subplots()
    ax.plot(a[i,:])
    ax.set_xlabel('samples')
    ax.set_ylabel('voltage')
#     ax.set_title('asdlkfj')
    ax.set_title(f'unit_code: {str(metadata.iloc[i]["unit_code"])}, ch: {str(metadata.iloc[i]["channel"])}')


# to get a specific unit, i think its metadata["unit_code"]].loc[unitcode]