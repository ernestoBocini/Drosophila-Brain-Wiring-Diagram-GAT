import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from functools import reduce
from pathlib import Path
from time import time
from node2vec import Node2Vec


def build_node2vec_embedding(
    g: nx.DiGraph,
    num_dims: int,
    walk_length: int = 30,
    num_walks: int = 50,
    window: int = 10,
    min_count: int = 1,
    batch_words: int = 4,
    num_workers: int = 1,
    col_prefix: str = "node2vec_",
):
    """Build Node2Vec embedding for a NetworkX graph.

    Parameters
    ----------
    g : networkx.DiGraph
        The graph to embed.
    num_dims: int
        Desired dimensionality of the embedding.
    walk_length, num_walks, window, min_count, batch_words : int
        Node2Vec parameters.
    num_workers : int
        Number of workers for parallel execution.
    col_prefix : str
        Prefix for column names in the output dataframe.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing embedded features indexed by node ID.
    """
    node2vec = Node2Vec(
        g,
        dimensions=num_dims,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=num_workers,
        weight_key="syn_count",
    )
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
    emb_df = pd.DataFrame(
        model.wv.vectors,
        index=np.array([int(x) for x in model.wv.index_to_key], dtype=np.int64),
        columns=[f"{col_prefix}{i}" for i in range(num_dims)],
    )
    return emb_df


def build_spectral_embedding(
    g: nx.DiGraph, num_dims: int, col_prefix: str = "spectral_"
):
    """Build spectral embedding for a NetworkX graph.

    Parameters
    ----------
    g : networkx.DiGraph
        The graph to embed.
    num_dims: int
        Desired dimensionality of the embedding.
    col_prefix : str
        Prefix for column names in the output dataframe.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing embedded features indexed by node ID.
    """
    print("    building laplacian matrix")
    lap = nx.directed_laplacian_matrix(g)
    print("    computing sparse matrix")
    lap_sparse = sparse.coo_matrix(lap)
    print("    computing eigenvectors")
    eigvals, eigvecs = sparse.linalg.eigsh(lap_sparse, k=num_dims, which="SM")
    features = np.array(eigvecs)
    # eigvals, eigvecs = np.linalg.eigh(np.array(lap))
    # features = eigvecs[:, np.argsort(eigvals)[:num_dims]]
    print("    converting to dataframe")
    emb_df = pd.DataFrame(
        features,
        index=g.nodes(),
        columns=[f"{col_prefix}{i}" for i in range(num_dims)],
    )
    return emb_df


if __name__ == "__main__":
    # Config
    num_dims_per_nt = 8  # number of dimensions per neurotransmitter type
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
    node2vec_embeddings = []
    spectral_embeddings = []
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

        # Build node2vec embedding
        print("  building node2vec embedding")
        st = time()
        node2vec_emb = build_node2vec_embedding(
            g,
            num_dims_per_nt,
            num_workers=num_workers,
            col_prefix=f"node2vec_{nt_type}_",
        )
        print(f"  done in {time() - st:.2f} seconds")
        node2vec_embeddings.append(node2vec_emb)

        # Build spectral embedding
        print("  building spectral embedding")
        st = time()
        spectral_emb = build_spectral_embedding(
            g, num_dims_per_nt, col_prefix=f"spectral_{nt_type}_"
        )
        print(f"  done in {time() - st:.2f} seconds")
        spectral_embeddings.append(spectral_emb)

    # Merge embeddings for all layers and save
    output_dir = data_dir / "embeddings"
    output_dir.mkdir(exist_ok=True)

    node2vec_embedding_all = reduce(
        lambda df1, df2: df1.merge(df2, left_index=True, right_index=True),
        node2vec_embeddings,
    )
    node2vec_embedding_all.to_pickle(output_dir / "node2vec_embedding.pkl")

    spectral_embedding_all = reduce(
        lambda df1, df2: df1.merge(df2, left_index=True, right_index=True),
        spectral_embeddings,
    )
    spectral_embedding_all.to_pickle(output_dir / "spectral_embedding.pkl")
