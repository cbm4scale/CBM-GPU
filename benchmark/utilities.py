from cbm.cbm4mm import cbm4mm
from cbm.cbm4ad import cbm4ad
from cbm.cbm4dad import cbm4dad
from cbm.csr4mm import csr4mm
from cbm.csr4ad import csr4ad
from cbm.csr4dad import csr4dad
from ogb.nodeproppred import PygNodePropPredDataset
from torch import sparse_coo_tensor, sparse_csr_tensor, Tensor, zeros, ones, int32, float32, arange, randint, stack, tensor
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset, SuiteSparseMatrixCollection, Planetoid
from torch_geometric.nn import GCNConv, GINConv, SAGEConv
from torch_geometric.transforms import OneHotDegree
from torch.nn import MSELoss, CrossEntropyLoss, Sequential, Linear, BatchNorm1d, ReLU
from torch.nn.functional import one_hot
from torch.optim import SGD, Adam


criterions = {
    'mse': MSELoss,
    'cross-entropy': CrossEntropyLoss
}

optimizers = {
    'sgd': SGD,
    'adam': Adam
}


def set_adjacency_matrix(layer, edge_index, alpha):
    if layer == "cbm-ax":
        return cbm4mm(edge_index.to(int32), ones(edge_index.size(1), dtype=float32), alpha=alpha), None
    elif layer == "cbm-adx":
        return cbm4ad(edge_index.to(int32), ones(edge_index.size(1), dtype=float32), alpha=alpha), None
    elif layer == "cbm-dadx":
        return cbm4dad(edge_index.to(int32), ones(edge_index.size(1), dtype=float32), alpha=alpha), None
    elif layer == "csr-ax":
        return csr4mm(edge_index), None
    elif layer == "csr-adx":
        return csr4ad(edge_index), None
    elif layer == "csr-dadx":
        return csr4dad(edge_index), None
    else:
        raise NotImplementedError(f"Layer {layer} is not valid")

############################################################
######################### DATASETS #########################
############################################################

# examples:
# load_tudataset('PROTEINS', 'node')
# load_tudataset('PROTEINS', 'graph')
# load_tudataset('COLLAB', 'node')
# load_tudataset('COLLAB', 'graph')
# ...
def print_dataset_info(name, dataset):
    print('------------------------------------------------------')
    print(f'Dataset: {name}')
    print(f'Number of Nodes: {dataset.num_nodes}')
    print(f'Number of Edges: {dataset.num_edges}')
    print('------------------------------------------------------')

def load_tudataset(name, task):     # graph and node prediction
    dataset = Batch.from_data_list(TUDataset(root="../data", name=name))

    # add node features (x) if not present
    if dataset.num_node_features == 0:
        degrees = dataset.edge_index[0].bincount(minlength=dataset.num_nodes)
        max_degree = degrees.max().item()
        transform = OneHotDegree(max_degree=max_degree)
        dataset = transform(dataset)
    
    # tudataset is for graph prediction (y.shape = (num_graphs) or (num_graphs, num_classes))
    # if task == "node" then y must be converted to node prediction (y.shape = (num_nodes) or (num_nodes, num_classes))
    # we follow a random approach
    if task == "node":
        # dataset.y = randint(0, 4, (dataset.num_nodes,), dtype=long)
        # dataset.y = ones((dataset.num_nodes, 1), dtype=float32)
        # dataset.num_classes = 4
        # dataset.num_classes = 1
        dataset.num_classes = 2
        y = zeros((dataset.num_nodes, dataset.num_classes), dtype=float32)
        y[arange(dataset.num_nodes), randint(0, dataset.num_classes, (dataset.num_nodes,))] = 1
        dataset.y = y
    # assert that num_classes is set (somes datasets loose this information after batching)
    elif task == "graph":
        dataset.num_classes = dataset.y.unique().size(0)
    
    
    
    return dataset

def load_snap(name):      # node prediction
    dataset = SuiteSparseMatrixCollection(root="../data", name=name, group='SNAP')[0]

    # add node features (x)
    degrees = dataset.edge_index[0].bincount(minlength=dataset.num_nodes)
    max_degree = degrees.max().item()
    transform = OneHotDegree(max_degree=max_degree)
    dataset = transform(dataset)

    # add node prediction (y)
    dataset.num_classes = 2
    y = zeros((dataset.num_nodes, dataset.num_classes), dtype=float32)
    y[arange(dataset.num_nodes), randint(0, dataset.num_classes, (dataset.num_nodes,))] = 1
    dataset.y = y
    
    

    return dataset

# examples:
# load_ogbn_proteins(0, 0.5)
# load_ogbn_proteins(1, 0.5)
# ...
# load_ogbn_proteins(7, 0.5)
# load_ogbn_proteins('all', None)
# load_ogbn_proteins(None, None)
def load_ogbn_proteins(edge_attr_feature, edge_attr_threshold):     # node prediction
    dataset = PygNodePropPredDataset(name='ogbn-proteins', root="../data")[0]

    # pick 1 feature and threshold the edge_attr
    if isinstance(edge_attr_feature, int):
        dataset.edge_index = dataset.edge_index[:, dataset.edge_attr[:, edge_attr_feature] >= edge_attr_threshold]
    # average all features row-wise and threshold the edge_attr
    elif edge_attr_feature == 'all':
        # 0.049 is the weighted mean of edge_attr
        edge_attr_threshold = 0.04856166988611221
        dataset.edge_index = dataset.edge_index[:, dataset.edge_attr.mean(dim=1) >= edge_attr_threshold]
    # else just ignore edge_attr and use existing edges
    dataset.edge_attr = None
    dataset.num_edge_features = 0
    dataset.num_edges = dataset.edge_index.size(1)
    dataset.num_classes = dataset.y.size(1)     # this dataset has y.shape = (num_nodes, num_classes)
    dataset.y = dataset.y.to(float32)

    # while graphs that are meant for graph prediction can be converted to node prediction
    # graphs meant for node prediction cannot be converted to graph prediction since we typically only have 1 graph

    # ogbn-proteins is for node prediction but is missing node features (x) so we need to add them
    degrees = dataset.edge_index[0].bincount(minlength=dataset.num_nodes)
    max_degree = degrees.max().item()
    transform = OneHotDegree(max_degree=max_degree)
    dataset = transform(dataset)
    
    print_dataset_info('ogbn-proteins', dataset)

    return dataset

# examples:
# load_planetoid('PubMed')
def load_planetoid(name):       # node prediction
    dataset = Planetoid(root="../data", name=name)[0]
    dataset.num_classes = dataset.y.unique().size(0)
    dataset.y = one_hot(dataset.y, num_classes=dataset.num_classes).to(float32)
    
    

    return dataset

# examples:
# load_dimacs('coPapersCiteseer')
def load_dimacs(name):      # node prediction
    dataset = SuiteSparseMatrixCollection(root="../data", name=name, group='DIMACS10')[0]

    # add node features (x)
    degrees = dataset.edge_index[0].bincount(minlength=dataset.num_nodes)
    max_degree = degrees.max().item()
    transform = OneHotDegree(max_degree=max_degree)
    dataset = transform(dataset)

    # add node prediction (y)
    dataset.num_classes = 2
    y = zeros((dataset.num_nodes, dataset.num_classes), dtype=float32)
    y[arange(dataset.num_nodes), randint(0, dataset.num_classes, (dataset.num_nodes,))] = 1
    dataset.y = y


    return dataset


def load_dataset(name):
    if name == "ca-HepPh":
        return load_snap("ca-HepPh"), 32
    elif name == "ca-AstroPh":
        return load_snap("ca-AstroPh"), 32
    elif name == "ogbn-proteins-raw":
        return load_ogbn_proteins(None, None), 32
    elif name == "PubMed":
        return load_planetoid("PubMed"), 32
    elif name == "Cora":
        return load_planetoid("Cora"), 32
    elif name == "coPapersCiteseer":
        return load_dimacs("coPapersCiteseer"), 32
    elif name == "coPapersDBLP":
        return load_dimacs("coPapersDBLP"), 32
    elif name == "COLLAB":
        return load_tudataset("COLLAB", "node"), 32
    else:
        raise NotImplementedError(f"Dataset {name} is not valid")