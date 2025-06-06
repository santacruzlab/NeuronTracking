#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:39:59 2025

@author: hungyunlu
"""

import os 
import re
import random
import numpy as np
import pandas as pd
import datetime
from collections import Counter, defaultdict


SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(SCRIPT_FOLDER)
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
RESULT_FOLDER = os.path.join(PROJECT_FOLDER, 'result')
DUMMY_NUMBER = 1e7 
USEFUL_N_UNIT = 3

pd.options.mode.chained_assignment = None  # Suppresses the warning


def parse_unit(unit: str):
    """
    Parse the unit code into session, date, channel, and unit code.
    
    :param unit: The unit code in the format of 'date_channel_unit_code'.
    :return: A tuple containing date, channel, and unit code.
    """
    pattern = r'([0-9]{8})_(([0-9]*)[a-d])' 

    date = re.search(pattern, unit).group(1) # Like 20230101
    unit_code = re.search(pattern, unit).group(2) # Like 22a
    channel = re.search(pattern, unit).group(3) # Like 22
    
    return date, channel, unit_code


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
        
        # Combine both data
        self.metadata['waveform'] = [self.waveforms[i] for i in range(len(self.metadata))]

        # Extract channels 
        self.channels = None
        self.find_channels()

        # Construct useful dataframes
        self.useful_df: pd.DataFrame = None
        self.useful_df = self.metadata[self.metadata['channel'].isin(self.channels)]
        self.useful_df.at[:, 'unit'] = self.useful_df['date'].astype(str) + "_" + self.useful_df['unit_code'].astype(str)

        # Extract dates
        self.dates = self.useful_df.date.unique()

        # Calculate similarity
        print(f'[{self.subject}] Calculating similarities')
        self.sim_df = None
        self.calc_similarity()
        
        # Calculate similarity thresholds
        print(f'[{self.subject}] Calculating similarity thresholds')
        self.threshold = None
        self.threshold_fussy = None
        self.calc_similarity_threshold()

        # Obtain cluster information        
        print(f'[{self.subject}] Obtaining cluster information')
        self.matched_units: list[str] = None
        self.clusters: pd.DataFrame = None
        self.useful_clusters: pd.DataFrame = None
        self.tuning_params: dict = dict()
        # self.calc_matched_units()
        # self.get_clusters()


    def find_channels(self):
        qualified_channels = self.metadata['channel'].value_counts()
        self.channels = qualified_channels[qualified_channels.values>=USEFUL_N_UNIT].index


    @staticmethod
    def similarity(a, b, metric: str):
        """
        Transformed similarity metrics
        """
        same_array = np.all(a==b)
        match metric:
            case 'euclidean':
                if same_array:
                    return 0
                else:
                    return np.log(np.sqrt(np.sum((a-b)**2)))
            case 'correlation':
                if same_array:
                    return DUMMY_NUMBER
                else:
                    return np.arctanh(np.corrcoef(a, b)[0,1])
                

    @staticmethod
    def rescale(array, metric: str):
        """
        Rescaled similarity metrics
        """
        maxx = np.max(array[array!=DUMMY_NUMBER])
        minn = np.min(array[array!=DUMMY_NUMBER])
        
        match metric:
            case 'correlation':
                result = (array - minn) / (maxx - minn)
                
            case 'euclidean':
                # For euclidean distance, we would like the smallest number to be 1,
                # And the greatest number to be 0.
                result = (maxx - array) / (maxx - minn)
        
        result[result > 1] = 1
        result[result < 0] = 0
        result[np.isnan(result)] = 0
        return result
    
    
    def calc_similarity(self):
        """
        Calculate the similarity score for each pair of units within the same channels across all sessions.
        Here we use pearson correlation and Euclidean distance to estimate total similarity.
        The similarity data are stored in self.sim_df: pd.DataFrame.
        
        Each row will be the similarity results from one useful channel.
        There will be 4 columns: 'correlation', 'euclidean', 'unit', and 'total'.
        
        The 'unit' column for each row is a list of units in that channel.
        The other columns store matrices of similarity whose shapes are (m, m),
        where m is the number of units across all sessions in that channel. 
        """
        
        # [[sim_df]] - stores the total similarity matrix for each channel
        self.sim_df = pd.DataFrame(columns=['correlation', 'euclidean', 'unit'], index=self.channels)
        
        for ch in self.channels:
            temp_dict = dict() # stores dataframes for each metric 
            # TODO
            ch_df = self.useful_df[self.useful_df['channel']==ch].sort_values(by='date')

            temp_dict['unit'] = ch_df['unit'].values

            for metric in ['correlation', 'euclidean']:
                sim_temp = np.zeros((len(ch_df), len(ch_df))) # stores similarity
                    
                # Compute for each pair of waveforms
                for i in range(len(ch_df)):
                    for j in range(len(ch_df)):
                        sim_temp[i,j] = self.similarity(
                            ch_df['waveform'].iloc[i],
                            ch_df['waveform'].iloc[j],
                            metric
                        )
                sim_temp = self.rescale(sim_temp, metric=metric)
                temp_dict[metric] = sim_temp
                
            self.sim_df.loc[ch] = temp_dict
    
        # total similarity is the avg of correlation and euclidean.
        self.sim_df['total'] = (
            self.sim_df['correlation'] + 
            self.sim_df['euclidean'] )/2
        

    def calc_similarity_threshold(self, predetermine: bool = True):
        """
        Threshold for determining whether units were from same neurons.
        Refer to the get_threshold() function.
        Since the function takes a long time to run, we do not rerun it everytime.
        
        If the total similarity between unit A and B > self.threshold,
        then A and B are considered the same units.
        
        self.threshold_fussy is the std of the distribution.
        It's left here as a less strict criterion (see calc_matched_units() method).
        """
        
        if predetermine:
            match self.subject:
                case 'airp':
                    self.threshold = 0.542
                    self.threshold_fussy = 0.006
                case 'braz':
                    self.threshold = 0.562
                    self.threshold_fussy = 0.008
                case 'Mouse1':
                    self.threshold = 0.573
                    self.threshold_fussy = 0.002
                case 'Mouse2':
                    self.threshold = 0.636
                    self.threshold_fussy = 0.004
                case 'Mouse3':
                    self.threshold = 0.585
                    self.threshold_fussy = 0.002
                case 'Mouse4':
                    self.threshold = 0.602
                    self.threshold_fussy = 0.002
                case 'Mouse5':
                    self.threshold = 0.596
                    self.threshold_fussy = 0.004
        else:
            self.get_threshold(95, False)
            print(self.threshold)
            print(self.threshold_fussy)
     
                
    @staticmethod
    def find_cluster(T, label):
        """
        Helper function to implement the depth-first search algorithm.
        This DFS algorithm includes units where one of the similarity scores 
        with existing units is greater than the threshold.
        
        For example, if unit [A, B, C] are considered the same units, and we are testing unit D.
        if total similarity T(A,D) < threshold and T(B,D) < threshold, 
        but T(C,D) > threshold, then D will be included in the same cluster.
        
        This algorithm allows the waveforms to be gradually change over time.
        """
        visited = set() # Set to track visited nodes
        cluster_name = [] # List to hold each cluster
        cluster_index = [] # The indices of each unit
    
        for i in range(len(T)):
            if i not in visited:
                stack = [i] # Stack for DFS
                current_cluster_name = [] # List to collect nodes in the current cluster
                current_cluster_index = []
                
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        current_cluster_name.append(label[node])
                        current_cluster_index.append(node)
                        # Add neighbors connected to 'node' by a 'True' entry in T
                        stack.extend([j for j in range(len(T)) 
                                          if T[node, j] and j not in visited])
                cluster_name.append(sorted(current_cluster_name))
                cluster_index.append(sorted(current_cluster_index))
        return cluster_name, cluster_index
        

    def calc_matched_units(self, fussy: bool = False, verify: bool = True):
        """
        Obtain matched units using the total similarity matrices.
        """
        
        
        def find_repeated_indices(lst):
            """
            This function finds the indices that have repeated values. 
            For example, given [1,2,3,4,4,4,4], return [[3,4,5,6]];
            given [1,1,2,3,4,4,4] return [[0,1], [4,5,6]].
            
            The goal is to eliminate some false positives from the calc_matched_units function.
            Since the function is agnostic to the session number,
            it could group the units from the same channel in the same session together.
            For example, unit 22a, 22b, and 22c from the same session can be grouped together.
            
            The function finds the indices of these false positives.
            """
            index_dict = defaultdict(list)
            for i, val in enumerate(lst):
                index_dict[val].append(i)
                
            repeated_indices = [indices for indices in index_dict.values() if len(indices) > 1]
            return repeated_indices
        
        
        matched_units = []
        for u in range(len(self.sim_df)):
            similarity = self.sim_df.total.iloc[u]
            label = self.sim_df.unit.iloc[u]
            if fussy: # Higher tolerance
                matched = similarity > self.threshold - self.threshold_fussy
            else:
                matched = similarity > self.threshold
            potential_match_name, potential_match_ind = self.find_cluster(matched, label)
            
            # For these potential matches, we will check whether there are conflicts.
            # Meaning different units in the same channel and session are grouped together.
            # These are false positives that should be removed.
            for k in range(len(potential_match_name)):
                name = potential_match_name[k] # Shorthand
                
                # Grab the dates for all labels 
                counts = Counter([parse_unit(n)[0] for n in name], subject=self.subject) 
                bug_dates = [key for key, value in counts.items() if value > 1]
                
                if len(bug_dates) > 0: # Then we will deal with this case
                
                    # Find where in that potential match (var: name) has repetition
                    ind_for_potential_match = find_repeated_indices([parse_unit(n)[0] for n in name], subject=self.subject) 
                    
                    # Use a list to collect the indices to be removed
                    ind_to_remove = []
                    for j in range(len(ind_for_potential_match)): # There may be multiple repetition
                        ind_match = ind_for_potential_match[j] # Shorthand 
                        ind_similarity = np.array(potential_match_ind[k])[ind_match]
                        
                        # Keep the one that has the greatest avg similarity score 
                        # among the potential matched units.
                        keep = np.argmax(similarity[ind_similarity].mean(1))
                        ind_match.pop(keep)
                        
                        # The remaining will be removed.
                        ind_to_remove.extend(ind_match)
                        
                    # Collect them all and remove them at once
                    for remove in sorted(ind_to_remove, reverse=True):
                        name.pop(remove)
                        
                matched_units.append(potential_match_name[k])
            
        self.matched_units = matched_units
        
        if verify:
            for i in range(len(self.matched_units)):
                unit = self.matched_units[i]
                unique_dates = np.unique([parse_unit(u)[0] for u in unit], subject=self.subject) 
                if len(unit) > len(unique_dates):
                    print(f'Matched units {i} has conflict units.')
                    print(f'Use self.matched_units[{i}] to debug.')
                
    
    def get_clusters(self):
        """
        Generate a dataframe (self.clusters) where each row is a group of matched units.
        """
        
        clusters = [Matched(i) for i in self.matched_units]
        n_units = [cluster.n_unit for cluster in clusters]
        duration = [cluster.duration for cluster in clusters]
        
        self.clusters = pd.DataFrame({
            'neuron': clusters,
            'n_unit': n_units,
            'duration': duration,
            })
        
        self.clusters['cluster_ID'] = np.arange(len(self.clusters)) # Assign an ID
        for c in range(len(self.clusters)):
            cluster_ID = self.clusters['cluster_ID'].iloc[c]
            cluster_df = self.clusters['neuron'].iloc[c].df
            
            # Assign the ID to useful_df so that we can grab rotation angles and direct/indirect units.
            for u in range(len(cluster_df)):
                unit = cluster_df['unit'].iloc[u]
                row = np.where(self.useful_df.unit==unit)[0][0]             
                self.useful_df.at[row,'cluster_ID'] = cluster_ID 
                
        self.useful_clusters = self.clusters[self.clusters['n_unit'] >= USEFUL_N_UNIT].reset_index(drop=True)
                

    def get_threshold(self, pct: float, PLOT: bool):
        
        """
        Calculate the similarity threshold for determining matched units.

        This function generates a null distribution of similarity scores by randomly pairing 
        waveforms from different channels across sessions. The similarity metrics are 
        aggregated, rescaled, and used to compute a total similarity score for each pair. 
        A threshold is then determined based on the specified percentile of the null distribution.

        Parameters:
            pct (float): The percentile value (e.g., 95 for 95th percentile) used to 
                        define the similarity threshold.

        Returns:
            None: The function updates the threshold attribute of the Tracking object.
        """


        # Obtain the amount of units for each channel in each session.
        # channels is used to ensure there are at least USEFUL_N_UNIT wavefroms
        subj_ch_session = np.zeros((len(self.channels), len(self.dates)))
        for i, ch in enumerate(self.channels):
            for j, date in enumerate(self.dates):
                subj_ch_session[i,j] = len(self.useful_df[(self.useful_df.date==date) & 
                                                        (self.useful_df.channel==ch)])
        
        # Store the channels and dates indicies.
        loc = np.zeros((len(self.dates), 2))
        loc[:,1] = np.arange(len(self.dates))
        
        threshold = []
        
        for it in range(200):
            
            print(f'{it} iteration')
            run = 0
            dist = []
            
            while len(dist) < 5000:
                
                run += 1        
                loc[:,0] = random.sample(range(len(self.channels)), len(self.dates))
                
                used_pair = [] # The pairs of channels and dates to obtain the null.
                for l in loc:
                    if subj_ch_session[int(l[0]), int(l[1])] != 0:
                        used_pair.append((int(self.channels[int(l[0])]), self.dates[int(l[1])]))
                        
                # used_pair should be greater than 2 (meaning at least 3), otherwise comparison would be meaningless
                # If len(used_pair) == 1 -> Nothing to compare.
                # If len(used_pair) == 2 -> Cannot be rescaled.
                if len(used_pair) > 2:
                    sim_temp = np.zeros((2, len(used_pair), len(used_pair)))
                    for m, metric in enumerate(['correlation', 'euclidean']):
                        for i, (ch1, date1) in enumerate(used_pair):
                            for j, (ch2, date2) in enumerate(used_pair):
                                
                                # Obtain waveformS from designated channels and dates. 
                                wf_s1 = self.useful_df[(self.useful_df['channel']==ch1)&(self.useful_df['date']==date1)]['waveform']
                                wf_s2 = self.useful_df[(self.useful_df['channel']==ch2)&(self.useful_df['date']==date2)]['waveform']
                                
                                # Randomly pick one if > 1 unit in that channel.
                                wf1 = wf_s1.iloc[random.randint(0, len(wf_s1)-1) if len(wf_s1) > 1 else 0].flatten()
                                wf2 = wf_s2.iloc[random.randint(0, len(wf_s2)-1) if len(wf_s2) > 1 else 0].flatten()
                                
                                sim_temp[m,i,j] = self.similarity(wf1, wf2, metric)

                        # Rescale based on each metric. Note that if len(used_pair) < 3, then it cannot be rescaled properly.
                        sim_temp[m] = self.rescale(sim_temp[m], metric=metric)

                    # Obtain total similarity matrix for this batch.
                    total_temp = sim_temp.mean(axis=0)
                
                    # Obtain the lower triangle of the total similarity.
                    dist += [float(total_temp[i, j]) 
                            for i in range(len(total_temp)) 
                            for j in range(i + 1, len(total_temp))]
                
            thres = np.percentile(dist, pct)                
            threshold.append(thres)

        self.threshold = np.mean(threshold)
        self.threshold_fussy = np.std(threshold)
        
        
class Matched:
    
    def __init__(self, units: list[str]):
        """
        The class for putative matched units.
        The input is a list of the unit codes.  
        
        In general, matched units should be accessed by self.clusters in the Tracking class.
        The matched units information can be read by
        
            self.clusters.neuron.iloc[cluster_ID].df
            
        where the df is made here by make_df(self)
        """
        self.units: list[str] = units
        self.n_unit = len(self.units)

        self.df: pd.DataFrame = None
        self.make_df()
                
        self.start_date: datetime.date = None
        self.end_date: datetime.date = None
        self.all_dates: list[datetime.date] = None
        self.duration = None
        self.get_dates()
        
        
    def __repr__(self):
        return self.units[0]
        

    def make_df(self):  
        
        # The first part is more general information from the unit ID.
        date = []
        unit_code = []
        channel = []
        for i in range(self.n_unit):
            d, c, u = parse_unit(self.units[i])
            date.append(d)
            channel.append(c)
            unit_code.append(u)
            
        self.df = pd.DataFrame({
            'unit': self.units,
            'date': date,
            'channel': channel,
            'unit_code': unit_code,
            }).sort_values(by='date')
        
        # Will be set through get_clusters() in class Tracking.
        self.df['rotation'] = 0.0
        self.df['is_direct'] = False
        
    def get_dates(self):
        date_dt = [datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8])) for d in self.df.date]
        date_dt.sort()
        self.start_date = date_dt[0]
        self.end_date = date_dt[-1]
        self.all_dates = date_dt
        self.duration = abs(self.end_date - self.start_date).days + 1 # Count both ends



def main():
    print('Tracking algorithm is running.')
    a: Tracking = Tracking('airp')


if __name__ == '__main__':
    main()