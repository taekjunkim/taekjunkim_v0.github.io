---
title: "How to create a NWB file for optical physiology data"
classes: wide 
categories:
  - Data Science
---



## Main steps to create a NWB file for imaging data
> https://pynwb.readthedocs.io/en/stable/tutorials/domain/ophys.html#sphx-glr-tutorials-domain-ophys-py <br>
> Each step requires several necessary information, and we can add more information relevant to our own experiment. 
<br><br>

1. Set-up a NWB file <br>
    - General information about an experiment session <br>
    - Subject information <br><br>
2. Add trial data <br>
    - start_time <br>
    - stop_time <br><br>
3. Create imaging plane <br>
    - Device <br>
    - Optical channel <br><br>
4. Raw data <br>
    - Add acquired two-photon images <br><br>
5. Processed data <br>
    - Add motion correction (optional): not included in this tutorial <br>
    - Add image segmentation <br>
    - Add fluorescence and dF/F responses <br><br>

## import modules


```python
# NWB related
from pynwb import NWBHDF5IO                        # to read/write
from pynwb import NWBFile                          # to set up
#from pynwb import TimeSeries                       # needed for motion correction (but not here)
from pynwb.base import Images
from pynwb.image import GrayscaleImage
#from pynwb.image import ImageSeries                # needed for motion correction (but not here)
from pynwb.ophys import (
    #CorrectedImageStack,                           # needed for motion correction (but not here)
    Fluorescence,
    ImageSegmentation,
    #MotionCorrection,                              # needed for motion correction (but not here)
    OpticalChannel,
    RoiResponseSeries,
    TwoPhotonSeries,
)
from pynwb.file import Subject

# to read TIFF (pip install pylibtiff), and suite2p output files
from libtiff import TIFF
import glob 
import os 

# datetime related
from datetime import datetime
from dateutil import tz

# matrix manipulation
import numpy as np

# to read exp information
import pandas as pd; 

# create a generator for a large array
from hdmf.data_utils import DataChunkIterator
```

## 1. Set-up a NWB file
- General information about an experiment session <br><br>
- Subject information <br><br>
- Electrode information <br><br>


```python
# nwb save folder
subject_num = '137'; 
site_num = 's143'; 
exp_num = '137_143_01'; 
rawdata_path = f'/Volumes/TK_exHDD1/Imaging/{subject_num}/tif/{site_num}/{exp_num}/'; 
nwb_folder = rawdata_path + 'suite2p/'; 

# exp_info_path
exp_info_path = '/Volumes/TK_exHDD1/Imaging/137/txt/'; 

# Year, Month, Day, Hour, Minute, Second
#start_time = datetime(2022, 4, 12, 10, 30, 0, tzinfo=tz.gettz('US/Pacific')); 
start_time = datetime.now(tz.tzlocal())

### general information

nwbfile = NWBFile(
    session_description="two-photon imaging. macaque V1",
    identifier='A137_143_01',
    session_start_time=datetime.now(tz.tzlocal()),
    lab="Bair Lab",
    institution="University of Washington",
    experiment_description="Sinusoidal grating: direction, SF",
    session_id="A137_136_03",
)
```


```python
### subject info
nwbfile.subject = Subject(
    subject_id='A137',
    age='P10Y', 
    description='Monkey A137',
    species='Macaca Mulatta', 
    sex='M'
)
```

## 2. Add trial data
Making a Trial table. 
In addition to "start_time", "stop_time" which are necessary, we can add more information to the Trial table. <br>
Here, I will add "stim_on", "stim_off", "dir", "sf"

- start_time <br>
- stop_time <br>


```python
nwbfile.add_trial_column(name="stim_on", description="stimulus onset time (sec)"); 
nwbfile.add_trial_column(name="stim_off", description="stimulus offset time (sec)"); 
nwbfile.add_trial_column(name="dir", description="stimulus moving direction (deg)");   
nwbfile.add_trial_column(name="sf", description="stimulus spatial frequency (c/deg)"); 

# If we want to have a list in a cell, we should add "index=True"
# nwbfile.add_trial_column(name="stim_color", description="stimulus color (RGB)", index=True);   # here, I added index=True, because this column requires a list in each cell.
```


```python
scan_txt = exp_info_path + f'{exp_num}_scan.txt'; 
trial_txt = exp_info_path + f'{exp_num}_trial.txt'; 

with open(scan_txt) as f:
    lines = f.readlines()
frame_ons = lines[0].split()    
frame_ons = np.array(list(map(int, frame_ons)));

# first frame starting time
frame_start = np.round(frame_ons[0]/1000,1); 
print(f"first frame starts at {frame_start}"); 

# imaging rate
img_rate = np.round(1000/np.mean(frame_ons[1:]-frame_ons[:-1]),1)
print(f"img_rate = {img_rate}")

### get stimulus information
stim_info = pd.read_csv(trial_txt, sep="\s", header=None, engine='python')
stim_info = stim_info.iloc[:,[0,1,3,5]]; 
stim_info.columns = ['stim_on','stim_dur','sf','dir']; 
stim_info
```

    first frame starts at 2.2
    img_rate = 19.9





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
      <th>stim_on</th>
      <th>stim_dur</th>
      <th>sf</th>
      <th>dir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3441</td>
      <td>2000</td>
      <td>0.32</td>
      <td>180</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6708</td>
      <td>2000</td>
      <td>0.48</td>
      <td>120</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9949</td>
      <td>2000</td>
      <td>0.64</td>
      <td>90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13191</td>
      <td>2000</td>
      <td>0.96</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16424</td>
      <td>2000</td>
      <td>3.84</td>
      <td>240</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>643</th>
      <td>2094415</td>
      <td>2000</td>
      <td>3.84</td>
      <td>60</td>
    </tr>
    <tr>
      <th>644</th>
      <td>2097656</td>
      <td>2000</td>
      <td>1.28</td>
      <td>330</td>
    </tr>
    <tr>
      <th>645</th>
      <td>2100881</td>
      <td>2000</td>
      <td>3.84</td>
      <td>240</td>
    </tr>
    <tr>
      <th>646</th>
      <td>2104106</td>
      <td>2000</td>
      <td>3.84</td>
      <td>90</td>
    </tr>
    <tr>
      <th>647</th>
      <td>2107322</td>
      <td>2000</td>
      <td>0.64</td>
      <td>270</td>
    </tr>
  </tbody>
</table>
<p>648 rows × 4 columns</p>
</div>




```python
### add trials
for i in np.arange(len(stim_info)):
    stim_on = stim_info['stim_on'].values[i]; 
    stim_off = stim_info['stim_on'].values[i] + stim_info['stim_dur'].values[i]; 
    start_time = stim_on-500; 
    stop_time = stim_off+500; 

    nwbfile.add_trial(
        start_time=start_time/1000, 
        stop_time=stop_time/1000, 
        stim_on=stim_on/1000, 
        stim_off=stim_off/1000,
        dir=stim_info['dir'].values[i],
        sf=stim_info['sf'].values[i],                
    )    
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
      <th>stim_on</th>
      <th>stim_off</th>
      <th>dir</th>
      <th>sf</th>
    </tr>
    <tr>
      <th>id</th>
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
      <td>2.941</td>
      <td>5.941</td>
      <td>3.441</td>
      <td>5.441</td>
      <td>180</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.208</td>
      <td>9.208</td>
      <td>6.708</td>
      <td>8.708</td>
      <td>120</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.449</td>
      <td>12.449</td>
      <td>9.949</td>
      <td>11.949</td>
      <td>90</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.691</td>
      <td>15.691</td>
      <td>13.191</td>
      <td>15.191</td>
      <td>60</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.924</td>
      <td>18.924</td>
      <td>16.424</td>
      <td>18.424</td>
      <td>240</td>
      <td>3.84</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>643</th>
      <td>2093.915</td>
      <td>2096.915</td>
      <td>2094.415</td>
      <td>2096.415</td>
      <td>60</td>
      <td>3.84</td>
    </tr>
    <tr>
      <th>644</th>
      <td>2097.156</td>
      <td>2100.156</td>
      <td>2097.656</td>
      <td>2099.656</td>
      <td>330</td>
      <td>1.28</td>
    </tr>
    <tr>
      <th>645</th>
      <td>2100.381</td>
      <td>2103.381</td>
      <td>2100.881</td>
      <td>2102.881</td>
      <td>240</td>
      <td>3.84</td>
    </tr>
    <tr>
      <th>646</th>
      <td>2103.606</td>
      <td>2106.606</td>
      <td>2104.106</td>
      <td>2106.106</td>
      <td>90</td>
      <td>3.84</td>
    </tr>
    <tr>
      <th>647</th>
      <td>2106.822</td>
      <td>2109.822</td>
      <td>2107.322</td>
      <td>2109.322</td>
      <td>270</td>
      <td>0.64</td>
    </tr>
  </tbody>
</table>
<p>648 rows × 6 columns</p>
</div>



## 3. Imaging plane
- Device <br>
- Optical channel <br><br>


```python
### Imaging plane
device = nwbfile.create_device(
    name="Microscope",
    description="Sutter MOM",
    manufacturer="Olympus lens",
)
optical_channel = OpticalChannel(
    name="OpticalChannel",
    description="green",                                            ## check
    emission_lambda=510.0,                                          ## check
)

imaging_plane = nwbfile.create_imaging_plane(
    name="ImagingPlane",
    optical_channel=optical_channel,
    imaging_rate=19.9,                                              ## check
    description="Depth xx microns, AAV(PHP.eB)-CAG-GCaMP6s, 2P",    ## check
    device=device,
    excitation_lambda=900.0,                                        ## check
    indicator="GCaMP6s",                                            ## check
    location="V1",                                                  ## check
    #grid_spacing=[0.01, 0.01],
    #grid_spacing_unit="meters",
    #origin_coords=[1.0, 2.0, 3.0],
    #origin_coords_unit="meters",
)
```

## 4. Raw data 
### - Add acquired Two-photon series


```python
# Iteratively read tiff ophys data from "allen_ophys_interface.py"
def tiff_iterator(paths_tiff):
    for tf in paths_tiff:
        tif = TIFF.open(tf)
        for image in tif.iter_images():
            yield image
        tif.close()
```


```python
rawdata_path = '/Volumes/TK_exHDD1/Imaging/137/tif/s143/137_143_01/'; 
paths_tiff = glob.glob(rawdata_path+"*.tif")
paths_tiff.sort()

raw_data_iterator = DataChunkIterator(data=tiff_iterator(paths_tiff))

two_p_series = TwoPhotonSeries(
    name="TwoPhotonSeries",
    data=raw_data_iterator,    
    imaging_plane=imaging_plane,
    starting_time=frame_start,
    rate=img_rate,
    unit='no unit',
)
nwbfile.add_acquisition(two_p_series); 
```

## 5. Processed data
### - Add motion correction (optional): not included in this tutorial

### - Add image segmentation


```python
# how many planes in this dataset
plane_folders = glob.glob(nwb_folder+'plane*');   
ops1 = [np.load(f+'/ops.npy', allow_pickle=True).item() for f in plane_folders]
nchannels = min([ops["nchannels"] for ops in ops1])

if len(ops1) > 1:
    multiplane = True
else:
    multiplane = False

# processing
ophys_module = nwbfile.create_processing_module(
    name="ophys", description="optical physiology processed data"); 

img_seg = ImageSegmentation()
ps = img_seg.create_plane_segmentation(
    name="PlaneSegmentation",
    description="suite2p output",
    imaging_plane=imaging_plane,
    reference_images=two_p_series,
)
ophys_module.add(img_seg)

file_strs = ["F.npy", "Fneu.npy", "spks.npy"]
file_strs_chan2 = ["F_chan2.npy", "Fneu_chan2.npy"]
traces, traces_chan2 = [], []
ncells = np.zeros(len(ops1), dtype=np.int_)
Nfr = np.array([ops["nframes"] for ops in ops1]).max()

for iplane, ops in enumerate(ops1):
    if iplane == 0:
        iscell = np.load(os.path.join(ops["save_path"], "iscell.npy"))
        for fstr in file_strs:
            traces.append(np.load(os.path.join(ops["save_path"], fstr)))
        if nchannels > 1:
            for fstr in file_strs_chan2:
                traces_chan2.append(
                    np.load(plane_folders[iplane].joinpath(fstr)))
        PlaneCellsIdx = iplane * np.ones(len(iscell))
    else:
        iscell = np.append(
            iscell,
            np.load(os.path.join(ops["save_path"], "iscell.npy")),
            axis=0,
        )
        for i, fstr in enumerate(file_strs):
            trace = np.load(os.path.join(ops["save_path"], fstr))
            if trace.shape[1] < Nfr:
                fcat = np.zeros((trace.shape[0], Nfr - trace.shape[1]),
                                "float32")
                trace = np.concatenate((trace, fcat), axis=1)
            traces[i] = np.append(traces[i], trace, axis=0)
        if nchannels > 1:
            for i, fstr in enumerate(file_strs_chan2):
                traces_chan2[i] = np.append(
                    traces_chan2[i],
                    np.load(plane_folders[iplane].joinpath(fstr)),
                    axis=0,
                )
        PlaneCellsIdx = np.append(
            PlaneCellsIdx, iplane * np.ones(len(iscell) - len(PlaneCellsIdx)))

    stat = np.load(os.path.join(ops["save_path"], "stat.npy"),
                    allow_pickle=True)
    ncells[iplane] = len(stat)
    for n in range(ncells[iplane]):
        if multiplane:
            pixel_mask = np.array([
                stat[n]["ypix"],
                stat[n]["xpix"],
                iplane * np.ones(stat[n]["npix"]),
                stat[n]["lam"],
            ])
            ps.add_roi(voxel_mask=pixel_mask.T)
        else:
            pixel_mask = np.array(
                [stat[n]["ypix"], stat[n]["xpix"], stat[n]["lam"]])
            ps.add_roi(pixel_mask=pixel_mask.T)

ps.add_column("iscell", "two columns - iscell & probcell", iscell)
```

### - Add FLUORESCENCE (all are required)


```python
rt_region = []
for iplane, ops in enumerate(ops1):
    if iplane == 0: 
        rt_region.append(
            ps.create_roi_table_region(
                region=list(np.arange(0, ncells[iplane]),),
                description=f"ROIs for plane{int(iplane)}",
            ))
    else:
        rt_region.append(
            ps.create_roi_table_region(
                region=list(
                    np.arange(
                        np.sum(ncells[:iplane]),
                        ncells[iplane] + np.sum(ncells[:iplane]),
                    )),
                description=f"ROIs for plane{int(iplane)}",
            ))

# FLUORESCENCE (all are required)
name_strs = ["Fluorescence", "Neuropil", "Deconvolved"]
name_strs_chan2 = ["Fluorescence_chan2", "Neuropil_chan2"]

for i, (fstr, nstr) in enumerate(zip(file_strs, name_strs)):
    for iplane, ops in enumerate(ops1):
        roi_resp_series = RoiResponseSeries(
            name=f"plane{int(iplane)}",
            data=np.transpose(traces[i][PlaneCellsIdx == iplane]),
            rois=rt_region[iplane],
            unit="lumens",
            rate=ops["fs"],
        )
        if iplane == 0:
            fl = Fluorescence(roi_response_series=roi_resp_series, name=nstr)
        else:
            fl.add_roi_response_series(roi_response_series=roi_resp_series)
    ophys_module.add(fl)

if nchannels > 1:
    for i, (fstr, nstr) in enumerate(zip(file_strs_chan2, name_strs_chan2)):
        for iplane, ops in enumerate(ops1):
            roi_resp_series = RoiResponseSeries(
                name=f"plane{int(iplane)}",
                data=np.transpose(traces_chan2[i][PlaneCellsIdx == iplane]),
                rois=rt_region[iplane],
                unit="lumens",
                rate=ops["fs"],
            )

            if iplane == 0:
                fl = Fluorescence(roi_response_series=roi_resp_series,
                                    name=nstr)
            else:
                fl.add_roi_response_series(roi_response_series=roi_resp_series)

        ophys_module.add(fl)
```

    /Users/taekjunkim/opt/anaconda3/lib/python3.9/site-packages/hdmf/common/table.py:1427: UserWarning: The linked table for DynamicTableRegion 'rois' does not share an ancestor with the DynamicTableRegion.
      warn(msg)


### - Add Background


```python
# BACKGROUNDS
# (meanImg, Vcorr and max_proj are REQUIRED)
bg_strs = ["meanImg", "Vcorr", "max_proj", "meanImg_chan2"]
for iplane, ops in enumerate(ops1):
    images = Images("Backgrounds_%d" % iplane)
    for bstr in bg_strs:
        if bstr in ops:
            if bstr == "Vcorr" or bstr == "max_proj":
                img = np.zeros((ops["Ly"], ops["Lx"]), np.float32)
                img[
                    ops["yrange"][0]:ops["yrange"][-1],
                    ops["xrange"][0]:ops["xrange"][-1],
                ] = ops[bstr]
            else:
                img = ops[bstr]
            images.add_image(GrayscaleImage(name=bstr, data=img))

    ophys_module.add(images)
```

## 5. Write NWB file


```python
nwb_filename = nwb_folder + f'sub-{subject_num}_ses-{exp_num}.nwb'; 
with NWBHDF5IO(nwb_filename, 'w') as io:
    io.write(nwbfile)
```

## Read NWB file


```python
del nwbfile, nwb_filename; 

nwb_filename = nwb_folder + f'sub-{subject_num}_ses-{exp_num}.nwb'; 
with NWBHDF5IO(nwb_filename, 'r') as io:
    nwbfile = io.read(); 
    print(nwbfile.acquisition["TwoPhotonSeries"])
    #print(nwbfile.processing["ophys"])
    #print(nwbfile.processing["ophys"]["Fluorescence"])
    #print(nwbfile.processing["ophys"]["Fluorescence"]["plane0"])
    #print(nwbfile.processing["ophys"]["Fluorescence"]["plane0"].data[:])
    ps = nwbfile.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"].to_dataframe()
ps
```

    TwoPhotonSeries pynwb.ophys.TwoPhotonSeries at 0x140484743013952
    Fields:
      comments: no comments
      conversion: 1.0
      data: <HDF5 dataset "data": shape (42077, 512, 512), type "<i2">
      description: no description
      imaging_plane: ImagingPlane pynwb.ophys.ImagingPlane at 0x140484743014720
    Fields:
      conversion: 1.0
      description: Depth xx microns, AAV(PHP.eB)-CAG-GCaMP6s, 2P
      device: Microscope pynwb.device.Device at 0x140484743015536
    Fields:
      description: Sutter MOM
      manufacturer: Olympus lens
    
      excitation_lambda: 900.0
      imaging_rate: 19.9
      indicator: GCaMP6s
      location: V1
      optical_channel: (
        OpticalChannel <class 'pynwb.ophys.OpticalChannel'>
      )
      unit: meters
    
      offset: 0.0
      rate: 19.9
      resolution: -1.0
      starting_time: 2.2
      starting_time_unit: seconds
      unit: no unit
    





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
      <th>pixel_mask</th>
      <th>iscell</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[[304, 216, 4.905288], [304, 217, 5.192344], [...</td>
      <td>[1.0, 0.9837135301234261]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[[376, 242, 1.9588753], [376, 243, 2.3426766],...</td>
      <td>[1.0, 0.9893293009170964]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[[394, 54, 4.841867], [394, 55, 4.921011], [39...</td>
      <td>[1.0, 0.8700730163261875]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[[319, 325, 1.6202793], [319, 326, 1.7638327],...</td>
      <td>[1.0, 0.7618596907331583]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[[346, 299, 4.05323], [346, 300, 4.8016133], [...</td>
      <td>[0.0, 0.05419699782352736]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>306</th>
      <td>[[420, 455, 2.0643792], [420, 456, 2.9144716],...</td>
      <td>[0.0, 0.46740236366329335]</td>
    </tr>
    <tr>
      <th>307</th>
      <td>[[357, 452, 2.5490983], [357, 453, 2.8698127],...</td>
      <td>[0.0, 0.46654810993518236]</td>
    </tr>
    <tr>
      <th>308</th>
      <td>[[155, 288, 2.0837245], [155, 289, 2.4107697],...</td>
      <td>[0.0, 0.2144801458847969]</td>
    </tr>
    <tr>
      <th>309</th>
      <td>[[325, 320, 3.4148602], [325, 321, 4.135799], ...</td>
      <td>[0.0, 0.17496257160544343]</td>
    </tr>
    <tr>
      <th>310</th>
      <td>[[142, 245, 1.6140407], [142, 246, 1.4526925],...</td>
      <td>[1.0, 0.6333532736144983]</td>
    </tr>
  </tbody>
</table>
<p>311 rows × 2 columns</p>
</div>


