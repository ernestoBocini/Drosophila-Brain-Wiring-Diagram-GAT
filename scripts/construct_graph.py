import pickle
import pandas as pd
import networkx as nx
from pathlib import Path


# Configs
proj_root = Path(__file__).absolute().parent.parent
connectome_version = 'v630'
codex_dump_dir = proj_root / 'data/codex_dump' / connectome_version
preprocessed_data_dir = proj_root / 'data/preprocessed'

# Load raw CSVs
print('Loading Codex dump...')
neurons = pd.read_csv(codex_dump_dir / 'neurons.csv')
morphology = pd.read_csv(codex_dump_dir / 'morphology_clusters.csv')
classification = pd.read_csv(codex_dump_dir / 'classification.csv')
cell_stats = pd.read_csv(codex_dump_dir / 'cell_stats.csv')
connections = pd.read_csv(codex_dump_dir / 'connections.csv')

# Build node attribute dataframe
print('Building node attribute dataframe...')
node_info = pd.merge(neurons, morphology, on='root_id', how='left')
node_info = pd.merge(node_info, classification, on='root_id', how='left')
node_info = pd.merge(node_info, cell_stats, on='root_id', how='left')

# Build edge attribute dataframe
print('Building edge attribute dataframe...')
edge_info = connections.copy()
# fine for now? might add effective weights in the future
# (ie. excitatory vs inhibitory)

# Build NetworkX graph
print('Building NetworkX representation...')
graph = nx.from_pandas_edgelist(
    edge_info,
    source='pre_root_id',
    target='post_root_id',
    edge_attr=['neuropil', 'syn_count', 'nt_type'],
    create_using=nx.DiGraph
)
node_attr_names = [
    'name', 'group', 'nt_type', 'nt_type_score', 'cluster',
    'flow', 'super_class', 'class', 'sub_class', 'cell_type',
    'hemibrain_type', 'hemilineage', 'side', 'nerve',
    'length_nm', 'area_nm', 'size_nm'
]
node_info_sel = node_info[['root_id', *node_attr_names]].set_index('root_id')
nx.set_node_attributes(graph, node_info_sel.to_dict(orient='index'))

# Serialize
print('Serializing...')
preprocessed_data_dir.mkdir(parents=True, exist_ok=True)
with open(preprocessed_data_dir / 'networkx_graph.pkl', 'wb') as f:
    pickle.dump(graph, f)
node_info.to_pickle(preprocessed_data_dir / 'node_info.pkl')
edge_info.to_pickle(preprocessed_data_dir / 'edge_info.pkl')
