---
title: "How to create a NWB file for electrophysiology data"
classes: wide 
categories:
  - DataScience
---




## Five main steps to create a NWB file
> Each step requires several necessary information, and we can add more information relevant to our own experiment. 
<br><br>

1. Set-up a NWB file <br>
    - General information about an experiment session <br>
    - Subject information <br>
    - Electrode information <br><br>
2. Add unit data <br>
    - spike_times <br><br>
3. Add trial data <br>
    - start_time <br>
    - stop_time <br><br>
4. Add acquired raw voltage data (LFP in processing, AP in acquisition) <br>  
   - AP sharing is relatively rare. <br><br>
5. Write NWB file <br>   


## import modules


```python
# NWB related
from pynwb import NWBHDF5IO                        # to read/write
from pynwb import NWBFile                          # to set up
from pynwb.ecephys import LFP, ElectricalSeries    # to add raw data
from pynwb.file import Subject

# datetime related
from datetime import datetime
from dateutil import tz

# matrix manipulation
import numpy as np

# create a generator for a large array
from hdmf.data_utils import GenericDataChunkIterator; 
```

## 1. Set-up a NWB file
- General information about an experiment session <br><br>
- Subject information <br><br>
- Electrode information <br><br>


```python
# Year, Month, Day, Hour, Minute, Second
start_time = datetime(2022, 4, 12, 10, 30, 0, tzinfo=tz.gettz('US/Pacific')); 

### general information
nwbfile = NWBFile(
    identifier='Link_220412',
    session_description='PFC-V4 dual recording during shape discrimination task. PFC cooling',  # required
    session_start_time=start_time,  # required
    experimenter='Erin Kempkes',  # optional
    session_id='l220412',  # optional
    institution='University of Washington',  # optional
    # related_publications='DOI:10.1016/j.neuron.2016.12.011',  # optional    
    lab='Pasupathy lab'   # optional
)
```


```python
### subject info
nwbfile.subject = Subject(
    subject_id='001',
    age='P10Y', 
    description='Monkey A',
    species='Macaca Mulatta', 
    sex='M'
)
```


```python
### Electrode table
device = nwbfile.create_device(
    name='Neuropixels', 
    description="bank0_384ch", 
    manufacturer="IMEC"
)

# add a column in the electrode table
nwbfile.add_electrode_column(name="label", description="label of electrode")

nprobes = 2
locations = ['V4','PFC']; 
nchannels_per_probe = 384
electrode_counter = 0

for iprobe in range(nprobes):

    # electrode_group: probe
    electrode_group = nwbfile.create_electrode_group(
        name = "probe{}".format(iprobe),
        description = "electrode group for probe {}".format(iprobe),
        device = device,
        location = locations[iprobe],
    )
    # channels in the probe
    for ielec in range(nchannels_per_probe):
        nwbfile.add_electrode(
            group = electrode_group,
            label = "probe{}_elec{}".format(iprobe, ielec),
            location = locations[iprobe],
        )
        electrode_counter += 1
```

### Check the electrode table


```python
nwbfile.electrodes.to_dataframe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>group</th>
      <th>group_name</th>
      <th>label</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>V4</td>
      <td>probe0 pynwb.ecephys.ElectrodeGroup at 0x14036...</td>
      <td>probe0</td>
      <td>probe0_elec0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V4</td>
      <td>probe0 pynwb.ecephys.ElectrodeGroup at 0x14036...</td>
      <td>probe0</td>
      <td>probe0_elec1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V4</td>
      <td>probe0 pynwb.ecephys.ElectrodeGroup at 0x14036...</td>
      <td>probe0</td>
      <td>probe0_elec2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V4</td>
      <td>probe0 pynwb.ecephys.ElectrodeGroup at 0x14036...</td>
      <td>probe0</td>
      <td>probe0_elec3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V4</td>
      <td>probe0 pynwb.ecephys.ElectrodeGroup at 0x14036...</td>
      <td>probe0</td>
      <td>probe0_elec4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>PFC</td>
      <td>probe1 pynwb.ecephys.ElectrodeGroup at 0x14036...</td>
      <td>probe1</td>
      <td>probe1_elec379</td>
    </tr>
    <tr>
      <th>764</th>
      <td>PFC</td>
      <td>probe1 pynwb.ecephys.ElectrodeGroup at 0x14036...</td>
      <td>probe1</td>
      <td>probe1_elec380</td>
    </tr>
    <tr>
      <th>765</th>
      <td>PFC</td>
      <td>probe1 pynwb.ecephys.ElectrodeGroup at 0x14036...</td>
      <td>probe1</td>
      <td>probe1_elec381</td>
    </tr>
    <tr>
      <th>766</th>
      <td>PFC</td>
      <td>probe1 pynwb.ecephys.ElectrodeGroup at 0x14036...</td>
      <td>probe1</td>
      <td>probe1_elec382</td>
    </tr>
    <tr>
      <th>767</th>
      <td>PFC</td>
      <td>probe1 pynwb.ecephys.ElectrodeGroup at 0x14036...</td>
      <td>probe1</td>
      <td>probe1_elec383</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 4 columns</p>
</div>



## 2. Add unit data <br>
In addition to "spike_times" which is necessary, we can add more information to the Units table. <br>
Here, I will add "unit_id", "probe_num", "best_ch", "depth", "sorting quality", "waveform_mean" 

- spike_times <br><br>


```python
nwbfile.add_unit_column(name="unit_id", description="unit id in session")
nwbfile.add_unit_column(name="probe_num", description="probe number")
nwbfile.add_unit_column(name="best_ch", description="best channel")
nwbfile.add_unit_column(name="depth", description="distance from the electrode tip")
nwbfile.add_unit_column(name="quality", description="sorting quality")
# we don't need to add "spike_times", "waveform_mean" column, because they were already defined in NWB

"""
each column assumes to get a single value in each row
if you want to input an array (more than a single value), we need to add "index=True" as below

nwbfile.add_unit_column(name="CoV",description="noise level in responses", data=cov, index=True); 
"""
```




    '\neach column assumes to get a single value in each row\nif you want to input an array (more than a single value), we need to add "index=True" as below\n\nnwbfile.add_unit_column(name="CoV",description="noise level in responses", data=cov, index=True); \n'



### example "for loop" to add unit one-by-one

You can edit the code to read real information from your data source and add information


```python
poisson_lambda = 20
firing_rate = 20
unit_ids = [0,3,4,5,7]

for unit_id in unit_ids:
    n_spikes = np.random.poisson(lam=poisson_lambda)
    spike_times = np.round(
        np.cumsum(np.random.exponential(1 / firing_rate, n_spikes)), 5
    )
    nwbfile.add_unit(
        unit_id = unit_id,
        probe_num = 0,
        best_ch = unit_id*2,
        depth = unit_id*20,
        spike_times=spike_times, 
        quality="good", 
        waveform_mean=[1.0, 2.0, 3.0, 4.0, 5.0]
    )
```

### Check the Units table


```python
nwbfile.units.to_dataframe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unit_id</th>
      <th>probe_num</th>
      <th>best_ch</th>
      <th>depth</th>
      <th>quality</th>
      <th>spike_times</th>
      <th>waveform_mean</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>good</td>
      <td>[0.00292, 0.08732, 0.19565, 0.20005, 0.25644, ...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>60</td>
      <td>good</td>
      <td>[0.10555, 0.12065, 0.2513, 0.26576, 0.26632, 0...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0</td>
      <td>8</td>
      <td>80</td>
      <td>good</td>
      <td>[0.14184, 0.1539, 0.26271, 0.34417, 0.36359, 0...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0</td>
      <td>10</td>
      <td>100</td>
      <td>good</td>
      <td>[0.03225, 0.03678, 0.06059, 0.12144, 0.16295, ...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>0</td>
      <td>14</td>
      <td>140</td>
      <td>good</td>
      <td>[0.03399, 0.06343, 0.09904, 0.32214, 0.35185, ...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0]</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Add trial data <br>
Making a Trial table is very similar to making an Units table. 
In addition to "start_time", "stop_time" which are necessary, we can add more information to the Trial table. <br>
Here, I will add "stim_id", "stim_on", "stim_off", "stim_color", "correct"

- start_time <br>
- stop_time <br><br>



```python
nwbfile.add_trial_column(name="stim_id", description="stimulus id"); 
nwbfile.add_trial_column(name="stim_on", description="stimulus onset time (sec)"); 
nwbfile.add_trial_column(name="stim_off", description="stimulus offset time (sec)"); 
nwbfile.add_trial_column(name="stim_color", description="stimulus offset time (sec)", index=True);   # here, I added index=True, because this column requires a list in each cell.
nwbfile.add_trial_column(name="correct", description="whether the trial was correct"); 
```

### Example code to add trial information

You can edit the code to read real information from your data source and add information


```python
nwbfile.add_trial(start_time=1.0, stop_time=5.0, stim_id = 3, stim_on = 1.2, stim_off = 3.2, stim_color = [100,100,100], correct=True)
nwbfile.add_trial(start_time=6.0, stop_time=10.0, stim_id = 31, stim_on = 6.2, stim_off = 8.2, stim_color = [100,200,50], correct=True)
```

### Check the Trial table


```python
nwbfile.trials.to_dataframe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>start_time</th>
      <th>stop_time</th>
      <th>stim_id</th>
      <th>stim_on</th>
      <th>stim_off</th>
      <th>stim_color</th>
      <th>correct</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>3</td>
      <td>1.2</td>
      <td>3.2</td>
      <td>[100, 100, 100]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.0</td>
      <td>10.0</td>
      <td>31</td>
      <td>6.2</td>
      <td>8.2</td>
      <td>[100, 200, 50]</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Add acquired raw voltage data (LFP in processing, AP in acquisition) <br>  
- AP sharing is relatively rare. <br><br>

> Here, I will add raw LFP signals from two probes. <br><br>
> When dealing with large arrays of data in NWB writes, it is often not possible to load all the data into memory. <br><br> 
> Using an iterative data write process allows us to avoid this problem by writing the data one-subblock-at-a-time, so that we only need to hold a small subset of the array in memory at any given time. <br><br>



```python
# electrode associated with LFP
table_probe0 = nwbfile.create_electrode_table_region(
    region=list(range(384)),  
    description="electrodes in probe0",
)
table_probe1 = nwbfile.create_electrode_table_region(
    region=list(range(384,electrode_counter)),
    description='electrodes in probe1',
)
```


```python
# Data Chunk Iterator
class LgArr_DataChunkIterator(GenericDataChunkIterator):
    def __init__(self, array: np.ndarray, **kwargs):
        self.array = array; 
        super().__init__(**kwargs); 

    def _get_data(self, selection):
        return self.array[selection]; 

    def _get_maxshape(self):
        return self.array.shape; 

    def _get_dtype(self):
        return self.array.dtype; 
```

### read LFP, define iterator


```python
### raw LFP data files
data_folder = '/Volumes/TK_exHDD1/NPX/V4_PFC_dual/220412/'; 
binname0 = data_folder + 'l220412_OcclTaskDual_g1_t5.imec0.lf.bin'; 
binname1 = data_folder + 'l220412_OcclTaskDual_g1_t5.imec1.lf.bin'; 


### check data shape
lfp_data0 = np.memmap(binname0, dtype='int16', mode='r');
lfp_data0 = lfp_data0.reshape((-1,385),order='C'); 
data_shape0 = np.shape(lfp_data0); 
starting_time0 = 0.5;   # timestamp of the first sample in seconds relative to the session start time 
                        # (here I used a fake number for demo)

lfp_data1 = np.memmap(binname1, dtype='int16', mode='r');
lfp_data1 = lfp_data1.reshape((-1,385),order='C'); 
data_shape1 = np.shape(lfp_data1); 
starting_time1 = 0.6;   # timestamp of the first sample in seconds relative to the session start time 
                        # (here I used a fake number for demo)

del lfp_data0, lfp_data1; 


### define iterator
my_iterator0 = LgArr_DataChunkIterator(array=np.memmap(binname0, dtype='int16', mode='r', shape=data_shape0)); 
my_iterator1 = LgArr_DataChunkIterator(array=np.memmap(binname1, dtype='int16', mode='r', shape=data_shape1)); 

```


```python
print(data_shape0)
```

    (6710311, 385)


### define LFP electrical series for each probe


```python
lfp_probe0 = ElectricalSeries(
    name="LFP_probe0",
    #data = H5DataIO(data=my_iterator0, compression="gzip", compression_opts=4),
    data=my_iterator0,
    electrodes=table_probe0,
    starting_time=starting_time0,  # timestamp of the first sample in seconds relative to the session start time
    rate=2500.0,  # in Hz
)

lfp_probe1 = ElectricalSeries(
    name="LFP_probe1",
    #data = H5DataIO(data=my_iterator1, compression="gzip", compression_opts=4),
    data=my_iterator1,
    electrodes=table_probe1,
    starting_time=starting_time1,  # timestamp of the first sample in seconds relative to the session start time
    rate=2500.0,  # in Hz
)
```

    /Users/taekjunkim/opt/anaconda3/lib/python3.9/site-packages/pynwb/ecephys.py:90: UserWarning: ElectricalSeries 'LFP_probe0': The second dimension of data does not match the length of electrodes. Your data may be transposed.
      warnings.warn("%s '%s': The second dimension of data does not match the length of electrodes. "
    /Users/taekjunkim/opt/anaconda3/lib/python3.9/site-packages/pynwb/ecephys.py:90: UserWarning: ElectricalSeries 'LFP_probe1': The second dimension of data does not match the length of electrodes. Your data may be transposed.
      warnings.warn("%s '%s': The second dimension of data does not match the length of electrodes. "


### put LFP in a ecephys processing module


```python
lfp = LFP(electrical_series=[lfp_probe0, lfp_probe1]);    # in case of one probe: LFP(electrical_series=lfp_probe0); 

ecephys_module = nwbfile.create_processing_module(
    name="ecephys", description="processed extracellular electrophysiology data"
)
ecephys_module.add(lfp); 
```

## 5. Write NWB file


```python
nwb_filename = data_folder + 'sub-A13008_ses-220412.nwb'; 
with NWBHDF5IO(nwb_filename, 'w') as io:
    io.write(nwbfile)
```

## Read NWB file


```python
del nwbfile, nwb_filename; 

nwb_filename = data_folder + 'sub-A13008_ses-220412.nwb'; 
with NWBHDF5IO(nwb_filename, 'r') as io:
    nwbfile = io.read(); 
    units = nwbfile.units.to_dataframe(); 
    trials = nwbfile.trials.to_dataframe(); 
    print(nwbfile.processing["ecephys"]["LFP"]["LFP_probe0"].data[:])
    print(np.shape(nwbfile.processing["ecephys"]["LFP"]["LFP_probe0"].data[:]))
```

    /Users/taekjunkim/opt/anaconda3/lib/python3.9/site-packages/pynwb/ecephys.py:90: UserWarning: ElectricalSeries 'LFP_probe0': The second dimension of data does not match the length of electrodes. Your data may be transposed.
      warnings.warn("%s '%s': The second dimension of data does not match the length of electrodes. "
    /Users/taekjunkim/opt/anaconda3/lib/python3.9/site-packages/pynwb/ecephys.py:90: UserWarning: ElectricalSeries 'LFP_probe1': The second dimension of data does not match the length of electrodes. Your data may be transposed.
      warnings.warn("%s '%s': The second dimension of data does not match the length of electrodes. "


    [[-15 -12 -26 ... -14   6   0]
     [-27 -24 -38 ... -26   6   0]
     [-31 -25 -40 ... -27   4   0]
     ...
     [ -7  -1 -15 ...  -7  24   0]
     [ -7  -1 -15 ...  -9  24   0]
     [ -4   2 -13 ...  -7  25   0]]
    (6710311, 385)



```python
units
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unit_id</th>
      <th>probe_num</th>
      <th>best_ch</th>
      <th>depth</th>
      <th>quality</th>
      <th>spike_times</th>
      <th>waveform_mean</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>good</td>
      <td>[0.00292, 0.08732, 0.19565, 0.20005, 0.25644, ...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>60</td>
      <td>good</td>
      <td>[0.10555, 0.12065, 0.2513, 0.26576, 0.26632, 0...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0</td>
      <td>8</td>
      <td>80</td>
      <td>good</td>
      <td>[0.14184, 0.1539, 0.26271, 0.34417, 0.36359, 0...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0</td>
      <td>10</td>
      <td>100</td>
      <td>good</td>
      <td>[0.03225, 0.03678, 0.06059, 0.12144, 0.16295, ...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>0</td>
      <td>14</td>
      <td>140</td>
      <td>good</td>
      <td>[0.03399, 0.06343, 0.09904, 0.32214, 0.35185, ...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0]</td>
    </tr>
  </tbody>
</table>
</div>




```python
trials
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>start_time</th>
      <th>stop_time</th>
      <th>stim_id</th>
      <th>stim_on</th>
      <th>stim_off</th>
      <th>stim_color</th>
      <th>correct</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>3</td>
      <td>1.2</td>
      <td>3.2</td>
      <td>[100, 100, 100]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.0</td>
      <td>10.0</td>
      <td>31</td>
      <td>6.2</td>
      <td>8.2</td>
      <td>[100, 200, 50]</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


