# APGL4SR: A Generic Framework with Adaptive and Personalized Global Collaborative Information in Sequential Recommendation (CIKM'2023) 
Source code for paper: [APGL4SR: A Generic Framework with Adaptive and Personalized Global Collaborative Information in Sequential Recommendation (CIKM'2023)](https://dl.acm.org/doi/abs/10.1145/3583780.3614781)

## Introduction
We incorporate adaptive and peronalized global collaborative information into sequential recommendation with the proposed APGL4SR framework.

## Reference
If you find our article or implemented codes helpful, please kindly cite our work. Thank you!

>@inproceedings{yin2023apgl4sr,<br>
   title={APGL4SR: A Generic Framework with Adaptive and Personalized Global Collaborative Information in Sequential Recommendation},<br>
   author={Yin, Mingjia and Wang, Hao and Xu, Xiang and Wu, Likang and Zhao, Sirui and Guo, Wei and Liu, Yong and Tang, Ruiming and Lian, Defu and Chen, Enhong},<br>
   booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},<br>
   pages={3009--3019},<br>
   year={2023}<br>
}


## Implementation
### Requirements

Python >= 3.7  
Pytorch >= 1.2.0  
tqdm == 4.26.0 
faiss-gpu == 1.7.1
nni == 2.10

### Datasets
Four prepared datasets are included in `data` folder.


### Train & Eval Model

```
cd src
chmod +x ./scripts/run_<DATASET>.sh
./scripts/run_<DATASET>.sh
```
where \<DATASET\> is the name of the four datasets.


## Acknowledgment
 - Transformer and training pipeline are implemented based on [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec) and [ICLRec](https://github.com/salesforce/ICLRec). Thanks to them for providing efficient implementation.

