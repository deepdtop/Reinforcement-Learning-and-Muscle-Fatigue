#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:47:15 2025

@author: Dan Terracina, PhD
"""
import os
from fitparse import FitFile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter


def extract_fit_metrics():
    """
    vector space [HR, Cadence, stridelength, stance time]
    """
    downloads_path = os.path.expanduser("~/Downloads")
    fit_file = [f for f in os.listdir(downloads_path) if f.endswith('.fit')][0]
    full_path = os.path.join(downloads_path, fit_file)

    fitfile = FitFile(full_path)
    data = []

    for record in fitfile.get_messages('record'):
        entry = {
            'timestamp': None,
            'heart_rate': None,
            'cadence': None,
            'step_length': None,
            'ground_contact_time': None,
            'stamina': None,
            'vertical_ratio': None,
            'enhanced_speed': None,
            'enhanced_altitude': None,
            'power': None
            
        }

        for field in record:
            name = field.name
            
            value = field.value
            #print(name,value)
            if name == 'timestamp':
                entry['timestamp'] = value
            elif name == 'heart_rate':
                entry['heart_rate'] = value
            elif name == 'cadence':
                entry['cadence'] = value
            elif name == 'stride_length':
                entry['step_length'] = value
            elif name == 'stance_time':
                entry['ground_contact_time'] = value
            elif name == 'stamina':
                entry['stamina'] = value
            elif name in ['vertical_ratio', 'vertical_oscillation_ratio']:
                entry['vertical_ratio'] = value
            elif name in ['enhanced_speed']:
                entry['enhanced_speed'] = value
            elif name in ['power']:
                entry['power'] = value
            elif name in ['enhanced_altitude']:
                entry['enhanced_altitude'] = value
            
            
            
        data.append([entry['heart_rate'],
                     entry['cadence'],
                     entry['step_length'], 
                     entry['ground_contact_time'],
                     entry['stamina'],
                     entry['vertical_ratio'],
                     entry['enhanced_speed'],
                     entry['power'],
                     entry['enhanced_altitude']
                     
                     ])
    data_df = pd.DataFrame(data, columns=['HR','Cadence','StepLength','GCT','stamina', 'vertical_ratio', 'enhanced_speed', 'power', 'enhanced_altitude'])
    df_smooth = data_df.apply(lambda col: savgol_filter(col, window_length=5, polyorder=2))

    return data_df, df_smooth

metrics, df_smooth = extract_fit_metrics()

metrics.plot()
df_smooth.plot()

metrics.to_pickle('/Users/danterracina/Documents/FATIGUE AND RL/data_all.pkl')
