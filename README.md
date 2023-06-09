# Network Machine Learning Project - _Drosophila_ brain connectome analysis

## Installation
```
conda install cuda-toolkit=11.7 -c nvidia
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

Note that the URL above needs to be changed depending on the PyTorch and CUDA
Toolkit versions as well as the OS. Find the correct URL on the [Installation
page](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) of PyG's documentation.

## Reproducibility
1. Download the raw data from [FlyWire Codex](https://codex.flywire.ai/api/download). We used the April 2023 version of the dataset (Snapshot 630).
2. Run `scripts/construct_graph.py` to generate the NetworkX DiGraph object based on the raw data. The result is saved as a pickle file.
3. Run `scripts/build_node_embedding.py` to generate the Node2Vec and spectral node features. The results are Pandas DataFrames saved as pickle files. This might take several hours.
4. Follow `notebooks/exploration.ipynb` to reproduce figures in the Exploration section of the report.
5. Follow `notebooks/exploitation.ipynb` to train the models discussed in the Exploitation section of the report. The accuracies and F1 scores of the models in different task configurations are saved in a CSV file. Note that it takes about an hour to train all models with a GPU.
6. Follow `notebooks/model_comparison.ipynb` to reproduce the result benchmarking figure and the confusion matrices.
