# [SWIN-RIND: Edge Detection for Reflectance, Illumination, Normal and Depth Discontinuity with Swin Transformer](https://proceedings.bmvc2023.org/707/)

##  Abstract
Edges are caused by the discontinuities in surface-reflectance, illumination, surface-normal, and depth (RIND). However, extensive research into the detection of specific edge types has not been conducted. Thus, in this paper, we propose a Swin Transformer-based method (referred to as SWIN-RIND) to detect these four edge types from a single input image. Attention-based approaches have performed well in general edge detection and are expected to work effectively for RIND edges. The proposed method utilizes the Swin Transformer as the encoder and a top-down and bottom-up multilevel feature aggregation block as the decoder. The encoder extracts cues at different levels, and the decoder integrates these cues into shared features containing rich contextual information. Then, each specific edge type is predicted through independent decision heads. To train and evaluate the proposed model, we used the public BSDS-RIND benchmark, which is based on the Berkeley Segmentation Dataset and contains annotations for the four RIND-edge types. The proposed method was evaluated experimentally, and the results demonstrate that the proposed SWIN-RIND method outperforms several state-of-the-art methods.
![image text](https://github.com/miaolun/SWIN-RIND/blob/1e3adc25be2b39619801c275878c9f85f4dd81a3/figure/network.png)

##  Dependencies
You need first to install CUDA and the corresponding PyTorch following  [PyTorch documentation](https://pytorch.org/get-started/locally/).
We used cuda 11.6 and PyTorch 1.12.1 in our experiments.
##  Installation
The necessary libraries are in [requirements.txt](). 
 ```
 pip install -r requirements.txt
 ```
 
##  Data
We trained our model on BSDS-RIND. BSDS-RIND is created by labeling images from the BSDS500.
Download [BSDS-RIND](https://github.com/MengyangPu/RINDNet) to the local [data](https://github.com/miaolun/SWIN-RIND/blob/main/data) folder. 
If you need original images you can download original images from [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).
##  Execution
###  Training
Download the pre-trained [swin-transformer model]((https://1drv.ms/u/s!AktMXPD2ejblnAIAWh8tYqbGpkLQ?e=dRQviV)) to the [model_zoo](https://github.com/miaolun/SWIN-RIND/blob/main/model_zoo) folder.
Modify the "your_checkpoint" path in [train.py](https://github.com/miaolun/SWIN-RIND/blob/main/train.py) and run. 
###  Testing
Download the pre-trained  [SWIN-RIND model](https://1drv.ms/u/s!AktMXPD2ejblnAPQRJxbE4--oe-z?e=cmNnEt) to the [model_zoo](https://github.com/miaolun/SWIN-RIND/blob/main/model_zoo) folder.
Set the "resume option" in  [my_options.py](https://github.com/miaolun/SWIN-RIND/blob/main/option/my_options.py)  and make sure the path of "your_checkpoint" is correct.


##  Precomputed Results
You can download the precomputed results [here](https://1drv.ms/u/s!AktMXPD2ejblnARSpQHL_niRsCzQ?e=w9xIGH) if you want to compare your method with SWIN-RIND.

##  Acknowledgments

 - Please note that this work is solely intended for academic research purposes only. Any other use is strictly prohibited.
 - We express our gratitude to the anonymous reviewers for their valuable feedback and inspiring suggestions. We appreciate their time and effort in providing us with insightful comments that have helped us to improve our work.
 - Thanks to the previous open-sourced repo:
[RINDNet](https://github.com/MengyangPu/RINDNet)
[HED-pytorch](https://github.com/xwjabc/hed)  
[RCF-pytorch](https://github.com/meteorshowers/RCF-pytorch)  
[BDCN](https://github.com/pkuCactus/BDCN)  
[DexiNed](https://github.com/xavysp/DexiNed)  
[DFF](https://github.com/Lavender105/DFF)  
[DOOBNet-pytorch](https://github.com/yuzhegao/doob)
##  Citation
```
@inproceedings{MIAO_2023_BMVC,
author    = {LUN MIAO and Takeshi Oishi and Ryoichi Ishikawa},
title     = {SWIN-RIND: Edge Detection for Reflectance, Illumination, Normal and Depth Discontinuity with Swin Transformer},
booktitle = {34th British Machine Vision Conference 2023, {BMVC} 2023, Aberdeen, UK, November 20-24, 2023},
publisher = {BMVA},
year      = {2023},
url       = {https://papers.bmvc2023.org/0707.pdf}
}
```
