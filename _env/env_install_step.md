```bash
conda create -n interdiff python=3.8 -y
conda activate interdiff
mamba install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
mamba install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

### url: https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html
### SYSTEM_TYPE: win_amd64/linux_x86_64
pip install torch_scatter-2.1.1+pt113cu116-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
pip install torch_cluster-1.6.1+pt113cu116-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
pip install torch_sparse-0.6.17+pt113cu116-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
pip install torch_spline_conv-1.2.2+pt113cu116-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
pip install torch_geometric -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

mamba install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```