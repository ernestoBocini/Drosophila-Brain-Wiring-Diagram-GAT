import pickle
import torch
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from pathlib import Path
from node2vec import Node2Vec

# Config
num_dims_per_nt = 8  # number of dimensions per neurotransmitter type
walk_length = 30  # length of random walk
num_walks = 50  # number of random walks per node
num_workers = 15  # number of workers for parallel execution

# Load preprocessd data
data_dir = Path(__file__).absolute().parent.parent / "data"
node_info = pd.read_pickle(data_dir / "preprocessed/node_info.pkl")
edge_info = pd.read_pickle(data_dir / "preprocessed/edge_info.pkl")

# Exclude optic neurons
node_info = node_info[node_info["super_class"] != "optic"]
edge_info = edge_info[
    edge_info["pre_root_id"].isin(node_info["root_id"])
    & edge_info["post_root_id"].isin(node_info["root_id"])
]
print(f"|V|={len(node_info)}, |E|={len(edge_info)}")

# Build node embedding: build a node2vec embedding for each neurotransmitter
# type, and concatenate the feature vectors
nt_types = edge_info["nt_type"].unique()
embeddings = []
for nt_type in nt_types:
    print(f"Building embedding for {nt_type}")
    edges = edge_info[edge_info["nt_type"] == nt_type]
    g = nx.from_pandas_edgelist(
        edges,
        source="pre_root_id",
        target="post_root_id",
        edge_attr=["neuropil", "syn_count", "nt_type"],
        create_using=nx.DiGraph,
    )
    # Some nodes are not connected to any other nodes if we consider only one
    # neurotransmitter type, so we need to add them to ensure the embedding
    # is complete
    missing_nodes = set(node_info["root_id"]) - set(g.nodes)
    g.add_nodes_from(missing_nodes)
    assert len(g) == len(node_info)
    print("  setting up node2vec model")
    node2vec = Node2Vec(
        g,
        dimensions=num_dims_per_nt,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=num_workers,
        weight_key="syn_count",
    )
    print("  fitting node2vec model")
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    emb_df = pd.DataFrame(
        model.wv.vectors,
        index=np.array([int(x) for x in model.wv.index_to_key], dtype=np.int64),
        columns=[f"{nt_type}_{i}" for i in range(num_dims_per_nt)],
    )
    embeddings.append(emb_df)

# Merge embeddings for all layers and save
output_dir = data_dir / "embeddings"
output_dir.mkdir(exist_ok=True)
# with open(output_dir / "_node2vec_outputs.pkl", "wb") as f:
#     pickle.dump(embeddings, f)
embedding_df = reduce(
    lambda df1, df2: df1.merge(df2, left_index=True, right_index=True), embeddings
)
embedding_df.to_pickle(output_dir / "node2vec_embedding.pkl")
