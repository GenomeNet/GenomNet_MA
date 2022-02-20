# DARTS-Based NAS Methods

## Executable Scripts:

* `train_genomicDARTS.py` (DARTS)
* `train_genomicPDARTS.py` (P-DARTS)
* `train_genomicDEP_DARTS.py` (DEP-DARTS)
* `train_genomicCWP_DARTS.py` (CWP-DARTS)

## Loaded Modules

* `genotype_parser.py`: `parse_genotype()`-function, used by `comp.aux.py`

### DARTS

* `architect.py`: `Architect`-class

### *P-DARTS
(P-DARTS and derivatives)
* `auxiliary_functions.py`: small helperfunctions
* `discard_operations.py`: `discard_cnn_ops()`, `discard_rhn_ops()` functions

### DEP-DARTS

* `model_search_de.py`: `RNNModelSearch`-class, equivalent of `nas_utils.model_search.py`
* `model_de.py`: auxiliary used by `model_search_de.py`
* `discard_edges.py`: `discard_cnn_edges()`, `discard_rhn_edges()` functions
