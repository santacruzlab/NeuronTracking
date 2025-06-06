#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 15:51:57 2025

@author: hungyunlu
"""


from trackMEA import track

def load_example_data(subj: str):
    return track.Tracking(subj)


def get_waveforms_from_channel(tracking_data: track.tracking, channel: int):
    """
    Extracts waveforms from a specific channel.
    
    Parameters
    ----------
    tracking_data : track.Tracking
        The tracking data object.
    channel : int
        The channel number to extract waveforms from.

    Returns
    -------
    waveforms : np.ndarray
        The extracted waveforms.
    """
    # Filter the dataframe for the specified channel
    filtered_df = tracking_data.useful_df[tracking_data.useful_df['channel'] == channel]
    
    # Extract the waveforms
    waveforms = filtered_df['waveform'].values
    
    return waveforms


def main():
    # Load example data
    subj = 'airp'
    tracking_data = load_example_data(subj)

    example_channel = 4
    print(f'Number of sessions in this channel: {tracking_data.useful_df[tracking_data.useful_df["channel"] == example_channel].shape[0]}')

    waveforms = get_waveforms_from_channel(tracking_data, example_channel)
    print(f'Number of waveforms in channel {example_channel}: {waveforms.shape[0]}')

    # Potential conflict: 

if __name__ == "__main__":
    main()