#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

###load numpy datas
data_dict = {}

path = '../processedData'
dirs = os.listdir(path)
for x in dirs:
    if os.path.splitext(x)[1] == ".npy":
        filePath = path + '/' + x
        data_key = os.path.splitext(x)[0]        
        data_value = np.load(filePath)   
        data_dict[data_key] = data_value
        
sensor_onset_dict = {'S01_20210419_103410_110410_C1': 1618794899,
                     'S01_20210423_101906_104906_C1': 1619140688,
                     'S01_20210428_091333_093006_C2': 1619572354}


def datetime_to_timestamp(datetime):
    datetime = pd.to_datetime(datetime)
    timestamp = (datetime - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    return timestamp-28800


def read_segment(filename):
    time_parts = filename.split('_')
    date_str = time_parts[1]
    onset_str = time_parts[2]
    offset_str = time_parts[3]
    date = date_str[0:4] + '-' + date_str[4:6] + '-' + date_str[6:8]
    onset = onset_str[0:2] + ':' + onset_str[2:4] + ':' + onset_str[4:6]
    offset = offset_str[0:2] + ':' + offset_str[2:4] + ':' + offset_str[4:6]
    onset_datetime_str = date + ' ' + onset 
    offset_datetime_str = date + ' ' + offset
    return onset_datetime_str, offset_datetime_str


def deal_20210420_1106(filename):
    onset_offset = [('2021-04-20 11:06:47', '2021-04-20 11:16:47'),
                    ('2021-04-20 11:26:48', '2021-04-20 11:36:48')]
    sensor_onset = sensor_onset_dict[filename]
    old_data = data_dict[filename]
    new_data = []
    for onset, offset in onset_offset:
        start_idx = datetime_to_timestamp(onset) - sensor_onset
        end_idx = datetime_to_timestamp(offset) - sensor_onset
        
        if start_idx < 0:
            start_idx = 0
        if end_idx >= len(old_data):
            end_idx = len(old_data) - 1
            
        if len(new_data) == 0:
            new_data = old_data[start_idx:end_idx]
        else:
            new_data = np.vstack((new_data, old_data[start_idx:end_idx]))
    np.save('../codedData/'+filename+'.npy', new_data)
    
    
def deal_20210426_0859(filename):
    onset_offset = [('2021-04-26 10:10:47', '2021-04-26 10:25:47'),
                    ('2021-04-26 10:58:18', '2021-04-26 11:13:18')]
    sensor_onset = sensor_onset_dict[filename]
    old_data = data_dict[filename]
    new_data = []
    for onset, offset in onset_offset:
        start_idx = datetime_to_timestamp(onset) - sensor_onset
        end_idx = datetime_to_timestamp(offset) - sensor_onset
        
        if start_idx < 0:
            start_idx = 0
        if end_idx >= len(old_data):
            end_idx = len(old_data) - 1
            
        if len(new_data) == 0:
            new_data = old_data[start_idx:end_idx]
        else:
            new_data = np.vstack((new_data, old_data[start_idx:end_idx]))
    np.save('../codedData/'+filename+'.npy', new_data)


for datafile in data_dict.keys():
    if datafile == '20210420_1106':
        deal_20210420_1106(datafile)
        continue
    if datafile == '20210426_0859':
        deal_20210426_0859(datafile)
        continue
    
    onset, offset = read_segment(datafile)
    onset_timestamp = datetime_to_timestamp(onset)
    offset_timestamp = datetime_to_timestamp(offset)
    sensor_onset = sensor_onset_dict[datafile]
    start_idx = onset_timestamp - sensor_onset
    end_idx = offset_timestamp - sensor_onset
    old_data = data_dict[datafile]
    
    if start_idx < 0:
        start_idx = 0
    if end_idx >= len(old_data):
        end_idx = len(old_data) - 1
    
    new_data = old_data[start_idx:end_idx+1]
    np.save('../codedData/'+datafile+'.npy', new_data)
