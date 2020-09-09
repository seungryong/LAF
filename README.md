# LAF-Net: Locally Adaptive Fusion Networks for Stereo Confidence Estimation
Official PyTorch code of "LAF-Net: Locally Adaptive Fusion Networks for Stereo Confidence Estimation," 
Sunok Kim, [Seungryong Kim](https://seungryong.github.io/), [Dongbo Min](http://cvl.ewha.ac.kr/), [Kwanghoon Sohn](http://diml.yonsei.ac.kr/), CVPR 2019 [[Project Page](https://seungryong.github.io/LAFNet/)].

<p align="center">
  <img src="LAF.png" width="600px" alt="LAF"></img>
</p>

## Requirements ##
* `Python` 
* `PyTorch`

## Download datasets ##
Download KITTI data (20 images in KITTI 2012, 200 images in KITTI 2015):
* [KITTI-data](https://people.eecs.berkeley.edu/~tinghuiz/projects/flowWeb/)

## Training ##
Launch the following command:
```shell
sh train.sh
```
or 
```shell
python train.py --stereo_matcher 'mccnn'
```

Optional arguments:
* `--base_lr`: learning rate
* `--batch_size`: batch size
* `--num_epochs`: maximum epochs
* `--step_size_lr`: step size for adjusting learning rate
* `--gamma_lr`: gamma for adjusting learning rate
* `--stereo_matcher`: stereo matching method

## Evaluation ##
Launch the following command:
```shell
python evaluate.py
```
Optional arguments:
* `--output_path`: output path
* `--stereo_matcher`: stereo matching method

## Citation
```shell
@inproceedings{Kim_CVPR_2019,
  title     = {LAF-Net: Locally Adaptive Fusion Networks for Stereo Confidence Estimation},
  author    = {Kim, Sunok and Kim, Seungryong and Min, Dongbo and Sohn, Kwanghoon},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```   

## Acknowledgements

Thanks to Matteo Poggi for sharing KITTI data and AUC code.
