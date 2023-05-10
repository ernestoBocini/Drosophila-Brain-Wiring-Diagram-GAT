# Network Machine Learning Project - _Drosophila_ brain connectome analysis

Installation (Sibo's Linux machine with NVIDIA GPU)
```
conda install cuda-toolkit=11.7 -c nvidia
pip install torch torchvision torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install networkx pandas numpy scipy scikit-learn matplotlib tqdm jupyter
```
