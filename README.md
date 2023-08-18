# APGL4SR: A Generic Framework with Adaptive and Personalized Global Collaborative Information in Sequential Recommendation

Source code for paper: [APGL4SR: A Generic Framework with Adaptive and Personalized Global Collaborative Information in Sequential Recommendation]

## Introduction
We incorporate adaptive and peronalized global collaborative information into sequential recommendation with the proposed APGL4SR framework.

## Reference

TODO

## Implementation
### Requirements

Python >= 3.7  
Pytorch >= 1.2.0  
tqdm == 4.26.0 
faiss-gpu==1.7.1

### Datasets
Four prepared datasets are included in `data` folder.


### Train & Eval Model

```
cd src
chmod +x ./scripts/run_<DATASET>.sh
./scripts/run_<DATASET>.sh
```
where <DATASET> is the name of the four datasets.


## Acknowledgment
 - Transformer and training pipeline are implemented based on [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec) [ICLRec](https://github.com/salesforce/ICLRec). Thanks them for providing efficient implementation.

