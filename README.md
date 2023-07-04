# Video Frame Interpolation with Stereo Event and Intensity Cameras

The stereo event-intensity camera setup is widely applied to leverage the advantages of both event cameras with low latency and intensity cameras that capture accurate brightness and texture information. However, such a setup commonly encounters cross-modality parallax that is difficult to be eliminated solely with stereo rectification especially for real-world scenes with complex motions and varying depths, posing artifacts and distortion for existing Event-based Video Frame Interpolation (E-VFI) approaches. 
To tackle this problem, we propose a novel Stereo Event-based VFI (SE-VFI) network to generate high-quality intermediate frames and corresponding disparities from misaligned inputs consisting of two consecutive keyframes and event streams emitted between them.
Specifically, we propose a Feature Aggregation Module (FAM) to alleviate the parallax and achieve spatial alignment in the feature domain. We then exploit the fused features accomplishing accurate optical flow and disparity estimation, and achieving impressive interpolated results through flow-based and synthesis-based ways.
We also build a stereo visual acquisition system composed of an event camera and an RGB-D camera to collect a new Stereo Event-Intensity Dataset (SEID) containing diverse scenes with complex motions and varying depths. 
Experiments on public real-world stereo datasets, i.e., DSEC and MVSEC, and our SEID dataset demonstrate that our proposed network outperforms state-of-the-art methods by a large margin.

## Environment setup
- python 3.7
- Pytorch 1.9.1
- opencv-python 4.6.0
- NVIDIA GPU + CUDA 11.1
- numpy, argparse, h5py

You can create a new [Anaconda](https://www.anaconda.com/products/individual) environment as follows.
<br>
```buildoutcfg
conda create -n sevfi python=3.7
conda activaet sevfi
```
Clone this repository.
```buildoutcfg
git clone git@github.com:dingchao1214/SEVFI.git
```
Install the above dependencies.
```buildoutcfg
cd SEVFI
pip install -r requirements.txt
```

## Download model and data
Pretrained models and sample data can be downloaded via Google Drive.
<br>
In our work, we conduct experiments on three real-world stereo event-intensity datasets:
<br>
- **DSEC** is a large-scale outdoor stereo event dataset especially for driving scenarios.
- **MVSEC** contains a single indoor scene and multi vehicle outdoor driving scenes. The events and frames are captured by two DAVIS346 cameras.
- **SEID** We build a stereo visual acquisition system containing an event camera and an RGB-D camera, and collect a new Stereo Event-Intensity Dataset. Our SEID captures dynamic scenes with complex motions and varying depths. (The data is coming soon.)

## Quick start
### Initialization 
- Copy the pretrined models to directory './PreTrained/'
- Copy the sample data to directory './sample/dataset/'

### Test
- Test on SEID data
```buildoutcfg
python test.py --dataset SEID --model_path ./PreTrained/ --origin_path ./sample/dataset/ --save_path ./sample/result/ --num_skip 5 --num_insert 5
```
- Test on DSEC data
```buildoutcfg
python test.py --dataset DSEC --model_path ./PreTrained/ --origin_path ./sample/dataset/ --save_path ./sample/result/ --num_skip 3 --num_insert 3
```
- Test on MVSEC data
```buildoutcfg
python test.py --dataset MVSEC --model_path ./PreTrained/ --origin_path ./sample/dataset/ --save_path ./sample/result/ --num_skip 5 --num_insert 5
```
**Main Parameters**:
- `--dataset`: dataset name.
- `--model_path`: path of models.
- `--origin_path`: path of sample data.
- `--save_path`: path of frame interpolation results.
- `--num_skip`: number of skip frames.
- `--num_insert`: number of insert frames.

## Citation
If you find our work useful in your research, please cite:
```buildoutcfg

```
