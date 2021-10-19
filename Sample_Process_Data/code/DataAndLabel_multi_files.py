#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import timedelta
import numpy as np

############################################################
#-!-denote where to change when dealing with different data
############################################################

###compute period during which all sensor datas are available
fname_list = ['ACC.csv', 'BVP.csv', 'EDA.csv', 'HR.csv', 'IBI.csv', 'TEMP.csv']
#-!-check the frequency manually
frequency = [32, 64, 4, 1, 'NA', 4]

#-!-datename, filename, sensor_start_time
datename = '20210426_0859'
filename1 = 'S08_20210426_101047_102547_C1'
filename2 = 'S08_20210426_105818_111318_C1'
#filename3 = 'S04_20210421_095307_100653_C2'
sensor_start_time = 1619398804

#-!-check the start time of every sensor
data_ACC = pd.read_csv('../'+datename+'/ACC.csv').drop(labels=range(11))
data_BVP = pd.read_csv('../'+datename+'/BVP.csv').drop(labels=range(11))
data_EDA = pd.read_csv('../'+datename+'/EDA.csv').drop(labels=range(11))
data_HR = pd.read_csv('../'+datename+'/HR.csv').drop(labels=0)
data_TEMP = pd.read_csv('../'+datename+'/TEMP.csv').drop(labels=range(11))

min_duration = min(len(data_ACC) // 32, len(data_BVP) // 64,
                   len(data_EDA) // 4, len(data_HR), len(data_TEMP) // 4)

onset = pd.to_datetime(sensor_start_time, unit='s', origin='unix') + timedelta(hours=8)
offset = pd.to_datetime(sensor_start_time + min_duration - 1, unit='s', origin='unix') + timedelta(hours=8)


###prepare label dataframe from excel file
sheet1 = pd.read_excel('../'+datename+'/'+filename1+'.xlsx', dtype=str)
sheet2 = pd.read_excel('../'+datename+'/'+filename2+'.xlsx', dtype=str)
#sheet3 = pd.read_excel('../'+datename+'/'+filename3+'.xlsx', dtype=str)

#-!-delete teacher's label
sheet1_without_teacher_label = sheet1.drop(labels=[0, 1, 2])
df1 = pd.DataFrame(sheet1_without_teacher_label)
df1.reset_index(drop=True, inplace=True)

sheet2_without_teacher_label = sheet2.drop(labels=[0, 1, 2, 3])
df2 = pd.DataFrame(sheet2_without_teacher_label)
df2.reset_index(drop=True, inplace=True)
'''
sheet3_without_teacher_label = sheet3.drop(labels=[0])
df3 = pd.DataFrame(sheet3_without_teacher_label)
df3.reset_index(drop=True, inplace=True)
'''
###merge sheets into one
df = pd.concat([df1, df2])
df.reset_index(drop=True, inplace=True)

#-!-date
df.loc[:, ['onset', 'offset']] = '2021-4-26 ' + df.loc[:, ['onset', 'offset']]

#binary label
frame = []
for i in range(len(df.index)):
    if df.loc[i, 'onset'] == df.loc[i, 'offset']:
        df1_dict = {'timestamp': df.loc[i, 'onset'], 'label': 1}
        df1 = pd.DataFrame(df1_dict, index=[0])
        df1['timestamp'] = pd.to_datetime(df1['timestamp'])
        df1 = df1.set_index('timestamp')
        frame.append(df1)
    else:
        df1_dict = {'timestamp': [df.loc[i, 'onset'], df.loc[i, 'offset']], 'label': [1, 1]}
        df1 = pd.DataFrame(df1_dict)
        df1['timestamp'] = pd.to_datetime(df1['timestamp'])
        df1 = df1.set_index('timestamp')
        new_onset = df1.index[0]
        new_offset = df1.index[1]
        t_index = pd.date_range(new_onset, new_offset, freq='S')
        df1 = df1.reindex(t_index, fill_value=1)
        frame.append(df1)

result = pd.concat(frame)
result = result[~result.index.duplicated()]
t_index = pd.date_range(onset, offset, freq='S')
result = result.reindex(t_index, fill_value=0)


###combine sensor data and label to form the final numpy data structure
ACC_values = data_ACC.values.flatten()
BVP_values = data_BVP.values.flatten()
EDA_values = data_EDA.values.flatten()
HR_values = data_HR.values.flatten()
TEMP_values = data_TEMP.values.flatten()
label_values = result.values

data_length = len(label_values)

float_data = np.zeros((data_length, 32*3+64+4+1+4+1))

for i in range(data_length):
    float_data[i, 0:96] = ACC_values[i*96:(i+1)*96]
    float_data[i, 96:160] = BVP_values[i*64:(i+1)*64]
    float_data[i, 160:164] = EDA_values[i*4:(i+1)*4]
    float_data[i, 164] = HR_values[i]
    float_data[i, 165:169] = TEMP_values[i*4:(i+1)*4]
    float_data[i, 169] = label_values[i, 0]

np.save('../processedData/'+datename+'.npy', float_data)