# Guided Diffusion for Molecule Generation with Interaction Embedding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/guanjq/targetdiff/blob/main/LICIENCE)



## Python Environment Install
Please use [_Mamba_](https://mamba.readthedocs.io/en/latest/micromamba-installation.html) to manage the environment.
```bash
conda create -n interdiff python=3.8 -y
conda activate interdiff
mamba install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

### url: https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html
### SYSTEM_TYPE: win_amd64/linux_x86_64
### We provide the *.whl for torch_geometric in _env.
pip install _env/SYSTEM_TYPE/torch_scatter-2.1.1+pt113cu116-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
pip install _env/SYSTEM_TYPEtorch_cluster-1.6.1+pt113cu116-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
pip install _env/SYSTEM_TYPEtorch_sparse-0.6.17+pt113cu116-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
pip install _env/SYSTEM_TYPEtorch_spline_conv-1.2.2+pt113cu116-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
pip install _env/SYSTEM_TYPEtorch_geometric -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

mamba install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

## Data Preprocess
We provide the preprocessed data on [data](https://drive.google.com/drive/folders/1QoKZsCFnJeGtQs14uSI1LVxIll0FlEnr?usp=sharing) Google Drive folder:
* `pocket_with_prompt.lmdb`
* `prompt_split.lmdb`

If you want to process the dataset from scratch, you need to download CrossDocked2020 v1.1 [data](https://drive.google.com/file/d/1T9jyEv7wq0nzn_G4JHyTQeevG5ULX8a6/view?usp=drive_link) processed by [Guan et al.](https://github.com/guanjq/targetdiff), the original dataset is filtered to keep the data with RMSD < 1A.
1. run the script [extract_pockets.py](scripts%2Fdata_preparation%2Fextract_pockets.py) to extract the pocket:
    ```bash
    python -m scripts.data_preparation.extract_pockets \ 
   --source_data_path ../interdiff_data/crossdocked_v1.1_rmsd1.0 \
   --save_pocket_path ../interdiff_data/crossdocked_v1.1_rmsd1.0_pocket \
   --save_db_path ../interdiff_data/pocket.lmdb \
   --num_workers 128
    ```
2. run the script [extract_prompt.py](scripts%2Fdata_preparation%2Fextract_prompt.py) for detecting interaction types and adding prompts to the dataset:
    ```bash
    python -m scripts.data_preparation.extract_prompt \
   --source_data_path ../interdiff_data/crossdocked_v1.1_rmsd1.0 \
   --source_db_path ../interdiff_data/pocket.lmdb \
   --temp_path ../interdiff_data/temp \
   --save_db_path ../interdiff_data/pocket_with_prompt.lmdb \
   --num_workers 128
    ```
After processing the data, you will obtain a `pocket_with_prompt.lmdb` database, which contains the data required for training, and the `train/test` split file `prompt_split.pt`.

## Training
```python
python -m scripts.train_diffusion configs/training.yml
```
## Sampling
Sampling for arbitrary pocket, the input format should be in PDB (Protein Data Bank) format, and the pockets need to be pre-extracted by yourself.
```python
python -m scripts.sample_for_pocket configs/sampling.yml --pdb_path examples.pdb
```