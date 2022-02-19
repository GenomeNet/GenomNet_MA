## Introduction

The provided framework enables the user to benchmark Neural Architecture Search (NAS) algorithms on genomic sequences, using the DeepSEA data. There is little research on how NAS algorithms perform on genomic sequences, because most NAS algorithms are applied on image classification, object detection or semantic segmentation. Due to this lack of research, we investigate how state-of-the art NAS algorithms, such as DARTS, Progressive Differentiable Architecture Search (P-DARTS) and Bayesian Optimized Neural Architecture Search (BONAS) can be used to find high-performance deep learning architectures in the field of genomics. Our search space combines convolutional and recurrent layer in a novel way, because popular genome models such as DanQ or NCNet also use these hybrid models which showed superior performance compared to pure convolution neural network architectures. We call the algorithms using our new search space which includes CNN and Recurrent Neural Network (RNN) operations *genomeDARTS*, *genomeP-DARTS*, and *genomeBONAS*. Furthermore, we build novel DARTS algorithms such as *CWP-DARTS*, which enables continuous weight sharing across different P-DARTS stages by transferring the neural network weights and architecture weights between P-DARTS iterations. In another P-DARTS extension,  we discard not only bad performing operations but also bad performing edges, which we call *genomeDEP-DARTS*. Additionally, we implement a novel algorithm which we call *OSP-NAS*. OSP-NAS starts with a super-network model which includes all randomly sampled operations and edges and then gradually discard randomly sampled operations based on the validation accuracy of the remaining super-network. 


## Tested with

- Ubuntu 20.04
- conda
- Python 3.6

- torch 1.7.1+cu101

(see `requirements.txt` for more details)

## Setup

1. Clone this repo and `cd` into it

2. Set up the python environment, using `conda`, which you get e.g. by installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

    ```sh
    conda create --name genomeNAS -c pytorch --file <( grep -v 'pypi_0$' requirements.txt)

    conda activate genomeNAS

    pip install -r  <( sed -n 's/=/==/; /pypi_0$/ s/=pypi_0$// p' requirements.txt )

    export PYTHONPATH="$PYTHONPATH:$PWD"
    ```

3. To get the DeepSea data for the prediction of chromatin effects of noncoding variants, download the data (3644 MB) from their [website](http://deepsea.princeton.edu) into the `data/` folder (expect this to take a while!):

    ```sh
    curl -L 'http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz' | tar xzv -C data
    ```

## Run baseline Models

These write the results into the folder `results/<model>`. See [baseline_models/README.md](baseline_models/README.md) for baseline references.

### DeepSea Model

```sh
mkdir results/deepsea

python baseline_models/train_baseline.py --num_steps=3000 --seq_size=1000 --batch_size=100 --train_directory='data/deepsea_train/train.mat' --valid_directory='data/deepsea_train/valid.mat' --test_directory='data/deepsea_train/test.mat' --epochs=50 --patience=40 --task='TF_bindings' --model='DeepSEA' --save='deepsea1' --save_dir=results/deepsea --model_path='results/deepsea/deepsea1.pth'```
```

### NCNet Models

**NCNet_bRR**

```sh
mkdir results/ncnet_brr

python baseline_models/train_baseline.py --num_steps=3000 --seq_size=1000 --batch_size=100 --train_directory='data/deepsea_train/train.mat' --valid_directory='data/deepsea_train/valid.mat' --test_directory='data/deepsea_train/test.mat' --epochs=50 --patience=40 --task='TF_bindings' --model='NCNet_bRR' --save='ncnet_brr1' --save_dir=results/ncnet_brr --model_path='results/ncnet_brr/ncnet_brr1.pth'
```

**NCNet_RR**

```sh
mkdir results/ncnet_rr

python baseline_models/train_baseline.py --num_steps=3000 --seq_size=1000 --batch_size=100 --train_directory='data/deepsea_train/train.mat' --valid_directory='data/deepsea_train/valid.mat' --test_directory='data/deepsea_train/test.mat' --epochs=50 --patience=40 --task='TF_bindings' --model='NCNet_RR' --save='ncnet_rr1' --save_dir=ncnet_rr --model_path='results/ncnet_rr/ncnet_rr1.pth'
```

### DanQ Model

```sh
mkdir results/danq

python baseline_models/train_baseline.py --num_steps=3000 --seq_size=1000 --batch_size=100 --train_directory='data/deepsea_train/train.mat' --valid_directory='data/deepsea_train/valid.mat' --test_directory='data/deepsea_train/test.mat' --epochs=50 --patience=40 --task='TF_bindings' --model='DanQ' --save='danq' --save_dir=results/danq --model_path='results/danq/danq1.pth'
```

## Run NAS algorithms for Genomics

These run a genomeNAS algorithm and store them in `results/<method>`.

### genomeDARTS

```
mkdir results/darts_search

python train_genomicDARTS.py --num_steps=2000 --seq_size=1000 --batch_size=64 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --one_clip=True --clip=0.25 --validation=True --report_validation=1 --epochs=50 --task='TF_bindings' --save='darts_1' --save_dir=results/darts_search
```

### Hyperband-NAS

```
mkdir results/hb_search

python train_genomicHyperbandNAS.py --num_steps=2000 --seq_size=1000 --batch_size=64 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --epochs=20 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --one_clip=True --clip=0.25 --validation=True --report_validation=1 --budget=3 --num_samples=25 --iterations=3 --task='TF_bindings'  --save='hb_1' --save_dir=results/hb_search
```

### genomeP-DARTS

```
mkdir results/pdarts_search

python train_genomicPDARTS.py --num_steps=2000 --seq_size=1000 --batch_size=64 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --one_clip=True --clip=0.25 --validation=True --report_validation=1 --epochs=25 --task='TF_bindings' --save='pdarts_1' --save_dir=results/pdarts_search
```

### Random search

```
mkdir results/random_search

python train_genomicRandom.py --num_steps=2000 --seq_size=1000 --batch_size=64 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --one_clip=True --clip=0.25 --validation=True --report_validation=1 --num_samples=20 --epochs=7 --task='TF_bindings' --save='random_1' --save_dir=results/random_search

```

### genomeBONAS

```
mkdir results/bonas_search

python train_genomicBONAS.py --num_steps=2000 --seq_size=1000  --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --one_clip=True --clip=0.25  --super_batch_size=64 --sub_batch_size=128 --generate_num=1000 --bo_sample_num=60 --epochs=60 --train_supernet_epochs=15 --iterations=2 --gcn_epochs=100 --sample_method='ea' --task='TF_bindings' --save='bonas_1' --save_dir=results/bonas_search
```

### genomeCWP-DARTS

```
mkdir results/cwp_search

python train_genomicCWP_DARTS.py --num_steps=2000 --seq_size=1000 --batch_size=64 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --one_clip=True --clip=0.25 --validation=True --report_validation=1 --task='TF_bindings' --save='cwp_1' --save_dir=results/cwp_search
```

### genomeDEP-DARTS

```
mkdir results/dep_search

python train_genomicDEP_DARTS.py --num_steps=2000 --seq_size=1000 --batch_size=64 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --one_clip=True --clip=0.25 --validation=True --report_validation=1 --task='TF_bindings' --save='dep_1' --save_dir=results/dep_search
```

### genomeOSP-NAS

```
mkdir results/osp_search

python train_genomicOSP_NAS.py --num_steps=2000 --seq_size=1000 --batch_size=64 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --epochs=60 --one_clip=True --clip=0.25 --validation=True --report_validation=1 --budget=5 --pretrain_epochs=10 --num_init_archs=108 --num_ops=7 --task='TF_bindings' --save='osp_1' --save_dir=results/osp_search
```


## Train and validate final Architectures

```
cd generalNAS_tools

mkdir darts_finalArchs

python train_finalArchitecture.py --num_steps=3000 --seq_size=1000 --batch_size=100 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --dropouth=0.05 --dropoutx=0.1 --epochs=50 --rhn_lr=8 --task='TF_bindings' --save_dir=darts_finalArchs --save='darts_arch_1' --model_path='darts_finalArchs/darts_arch_1.pth' --genotype_file='/home/ascheppa/GenomNet_MA/genomicNAS_Algorithms/darts_search/darts_geno-darts_1.npy'

```


## Test final Architectures

```
cd generalNAS_tools

mkdir darts_test

python test_finalArchitecture.py --seq_size=1000 --batch_size=100 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --task='TF_bindings' --save_dir=darts_test --save='darts_test_1' --model='/home/ascheppa/GenomNet_MA/generalNAS_tools/darts_finalArchs/darts_arch_1.pth'

```


# Folders

* `BO_tools`:
* `BONAS_search_space`:
* `baseline_models`:
* `darts_tools`:
* `data_generators`:
* `generalNAS_tools`
* `genomicNAS_Algorithms`: Location of the actual algorithms
* `predictors`:
* `randomSearch_and_Hyperband_Tools`:
* `samplers`:
* `opendomain_utils`:
* `preliminary_study_results`:
* `data`: Directory where the DeepSea-data should be downloaded into.
* `results`: Directory where results get stored.
