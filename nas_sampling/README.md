# Sampling-Based NAS Methods

## Executable Scripts:

* `train_genomicOSP_NAS.py` (OSP-NAS)
* `train_genomicRS.py` (Random Search)
* `train_genomicSH.py` (Successive Halving)

## Loaded Modules

### OSP-NAS
* `create_masks.py`
* `hyperband_final_tools.py`: `final_stage_run()`-function
* `hyperbandSampler.py`
* `model_search.py`: `RNNModelSearch`-class

### RS / SH
* `random_Sampler.py`: `generate_random_architectures()`-function
* `hb_iteration.py`: `hb_step()`-function (SH only)

### Both
* `model.py`: `RNNModel`-class, used indirectly by OSP-NAS
* `utils.py`: `mask2geno()`-function, `mask2switch()`-function (OSP-NAS only)
