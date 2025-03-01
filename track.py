#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:39:59 2025

@author: hungyunlu
"""

import os 
import numpy as np
import pandas as pd


SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_FOLDER, 'data')
RESULT_FOLDER = os.path.join(SCRIPT_FOLDER, 'result')

class Tracking:
    
    def __init__(self, subject: str): 
        """
        The tracking algorithm reads the data for the input subject in the data folder.
        
        The folder structure should be like this.
        Consult preprocess.py or customize the waveform data into this format.
        --------------------------------
        track.py
        preprocess.py
        data
        |__ subject_A
            |__ waveforms.npy
            |__ waveforms_metadata.csv
        |__ subject_B
            |__ waveforms.npy
            |__ waveforms_metadata.csv
        |__ ....
        --------------------------------
        
        :param subject: the name of the subject, should be the same as the folder name.       
        """
        
        self.subject = subject
        self.data_dir = os.path.join(DATA_FOLDER, self.subject)
        
        if not os.path.exists(self.data_dir):
            raise ValueError('Data for this subject not existed.')
            
        self.waveforms = np.load(os.path.join(self.data_dir, 'waveforms.npy'))
        self.metadata = pd.read_csv(os.path.join(self.data_dir, 'waveforms_metadata.csv'))
        
        if len(self.metadata) != self.waveforms.shape[0]: # Last sanity check
            raise ValueError('Dimensions of waveforms and metadata do not agree.')
        
        # Extract channels
        self.channels = None
        
        
    def find_channels(self):
        qualified_channels = self.metadata['channel'].value_counts()
