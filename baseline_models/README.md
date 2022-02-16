# Baseline Deep Learning Models

`train_baseline.py` is executable and trains baseline models. This repo contains reimplementations of:

* **DanQ**: `models/DanQ_model.py`,  Quang, Daniel, and Xiaohui Xie. "DanQ: a hybrid convolutional and recurrent deep neural network for quantifying the function of DNA sequences." Nucleic acids research 44.11 (2016): e107-e107.
* **DeepSea**: `models/DeepSea.py`, Zhou, Jian, and Olga G. Troyanskaya. "Predicting effects of noncoding variants with deep learning–based sequence model." Nature methods 12.10 (2015): 931-934.
* **NCNET_RR**, **NCNet_bRR**: `models/NCNet_RR_model.py`, `models/NCNet_bRR_model.py`: Zhang, Hanyu, et al. "NCNet: Deep learning network models for predicting function of non-coding DNA." Frontiers in genetics 10 (2019): 432.

Model training and evaluation happens with `Train`, `Valid`, and `Test` functions from `train_and_val_*.py`. DanQ, NCNet_RR, NCNet_bRR use `train_and_val_baseline.py`, DeepSea uses `train_and_val_deepsea.py`.

## Other files

* `performance.py`: analysis of results
* `models/DeepVirFinder_model.py`: not fully implemented.
