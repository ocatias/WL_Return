import os

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset
import torch.optim as optim
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected, Compose
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.utils.features import get_atom_feature_dims

from GrapTransformations.cell_encoding import CellularRingEncoding, CellularCliqueEncoding
from GrapTransformations.subgraph_bag_encoding import SubgraphBagEncoding, policy2transform
from Models.gnn import GNN
from Models.encoder import NodeEncoder, EdgeEncoder, ZincAtomEncoder, EgoEncoder
from Models.ESAN.conv import GINConv, OriginalGINConv, GCNConv, ZINCGINConv 
from Models.ESAN.models import DSSnetwork
from Models.mlp import MLP

def get_transform(args):
    if args.use_cliques:
        transform = CellularCliqueEncoding(args.max_struct_size, aggr_edge_atr=args.use_aggr_edge_atr, aggr_vertex_feat=args.use_aggr_vertex_feat,
            explicit_pattern_enc=args.use_expl_type_enc, edge_attr_in_vertices=args.use_edge_attr_in_vertices)
    elif args.use_rings:
        transform = CellularRingEncoding(args.max_struct_size, aggr_edge_atr=args.use_aggr_edge_atr, aggr_vertex_feat=args.use_aggr_vertex_feat,
            explicit_pattern_enc=args.use_expl_type_enc, edge_attr_in_vertices=args.use_edge_attr_in_vertices)
    elif not args.use_esan and args.policy != "":
        policy = policy2transform(args.policy, args.num_hops)
        transform = SubgraphBagEncoding(policy, explicit_type_enc=args.use_expl_type_enc)
    elif args.use_esan and args.policy != "":
        transform = policy2transform(args.policy, args.num_hops)
    else:
        transform = None
    return transform

def load_dataset(args, config):
    transform = get_transform(args)

    if transform is None:
        dir = os.path.join(config.DATA_PATH, args.dataset, "Original")
    else:
        dir = os.path.join(config.DATA_PATH, args.dataset, repr(transform))

    if args.dataset.lower() == "zinc":
        datasets = [ZINC(root=dir, subset=True, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "cifar10":
        datasets = [GNNBenchmarkDataset(name ="CIFAR10", root=dir, split=split, pre_transform=Compose([ToUndirected(), transform])) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "cluster":
        datasets = [GNNBenchmarkDataset(name ="CLUSTER", root=dir, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-ppa", "ogbg-code2", "ogbg-molpcba", "ogbg-moltox21"]:
        dataset = PygGraphPropPredDataset(root=dir, name=args.dataset.lower(), pre_transform=transform)
        split_idx = dataset.get_idx_split()
        datasets = [dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]]
        
    if args.use_esan:
        print("Using ESAN")
        train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True, follow_batch=['subgraph_idx'])
        val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False, follow_batch=['subgraph_idx'])
        test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False, follow_batch=['subgraph_idx'])
    else:
        train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_model(args, num_classes, num_vertex_features, num_tasks):
    if not args.use_esan:
        node_feature_dims = []

        if args.use_expl_type_enc:
            if args.use_rings:
                node_feature_dims = [2,2,2]
            if args.use_cliques:
                for _ in range(args.max_struct_size):
                    node_feature_dims.append(2)
            elif args.policy != "":
                node_feature_dims = [2,2,2]

        if args.dataset.lower() == "zinc":
            node_feature_dims.append(21)
            node_encoder = NodeEncoder(emb_dim=args.emb_dim, feature_dims=node_feature_dims)
            edge_encoder =  EdgeEncoder(emb_dim=args.emb_dim, feature_dims=[4])
        elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21"]:

            node_feature_dims += get_atom_feature_dims()
            print("node_feature_dims: ", node_feature_dims)
            node_encoder, edge_encoder = NodeEncoder(args.emb_dim, feature_dims=node_feature_dims), EdgeEncoder(args.emb_dim)
        else:
            node_encoder, edge_encoder = lambda x: x, lambda x: x
                
        if args.model.lower() == "gin":
            # Cell Encoding
            if args.use_cliques or args.use_rings:
                return GNN(num_classes, num_tasks, args.num_layers, args.emb_dim, 
                        gnn_type = 'gin', virtual_node = args.use_virtual_node, drop_ratio = args.drop_out, JK = "last", 
                        graph_pooling = "type" if args.use_dim_pooling else "mean", 
                        max_type = 3 if args.use_rings else args.max_struct_size, edge_encoder=edge_encoder, node_encoder=node_encoder)
            # Without Cell Encoding
            else:
                return GNN(num_classes, num_tasks, args.num_layers, args.emb_dim, 
                        gnn_type = 'gin', virtual_node = args.use_virtual_node, drop_ratio = args.drop_out, JK = "last", 
                        graph_pooling = "type" if args.use_dim_pooling else "mean", 
                        max_type = 2 if args.policy != "" else 0, edge_encoder=edge_encoder, node_encoder=node_encoder)
        elif args.model.lower() == "mlp":
                return MLP(num_features=num_vertex_features, num_layers=args.num_layers, hidden=args.emb_dim, 
                        num_classes=num_classes, num_tasks=num_tasks, dropout_rate=args.drop_out)
        else: # Probably don't need other models
            pass

    # ESAN
    else:
        encoder = lambda x: x
        if 'ogb' in args.dataset:
            encoder = AtomEncoder(args.emb_dim) if args.policy != "ego_nets_plus" else EgoEncoder(AtomEncoder(args.emb_dim))
        elif 'ZINC' in args.dataset:
            encoder = ZincAtomEncoder(policy=args.policy, emb_dim=args.emb_dim)

        if args.model == 'GIN':
            GNNConv = GINConv
        elif args.model == 'originalgin':
            GNNConv = OriginalGINConv
        elif args.model == 'graphconv':
            GNNConv = GraphConv
        elif args.model == 'gcn':
            GNNConv = GCNConv
        elif args.model == 'ZINCGIN':
            GNNConv = ZINCGINConv
        else:
            raise ValueError('Undefined GNN type called {}'.format(args.gnn_type))

        if 'ogb' in args.dataset or 'ZINC' in args.dataset:
            in_dim = args.emb_dim if args.policy != "ego_nets_plus" else args.emb_dim + 2
        elif args.dataset == 'CSL':
            in_dim = 6 if args.policy != "ego_nets_plus" else 6 + 2  # used deg as node feature
        else:
            in_dim = dataset.num_features

        model = DSSnetwork(num_layers=args.num_layers, in_dim=in_dim, emb_dim=args.emb_dim, num_tasks=num_tasks*num_classes,
                           feature_encoder=encoder, GNNConv=GNNConv)
        return model


def get_optimizer_scheduler(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    args.lr_scheduler_decay_steps,
                                                    gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    elif args.lr_scheduler == "ReduceLROnPlateau":
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                mode='min',
                                                                factor=args.lr_scheduler_decay_rate,
                                                                patience=args.lr_schedule_patience,
                                                                verbose=True)
    else:
        raise NotImplementedError(f'Scheduler {args.lr_scheduler} is not currently supported.')

    return optimizer, scheduler

def get_loss(args):
    metric_method = None
    if args.dataset.lower() == "zinc":
        loss = torch.nn.L1Loss()
        metric = "mae"
    elif args.dataset.lower() == "cifar10":
        loss = torch.nn.CrossEntropyLoss()
        metric = "accuracy"
    elif args.dataset in ["ogbg-molhiv", "ogbg-moltox21"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "rocauc (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset == "ogbg-ppa":
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "accuracy (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset in ["ogbg-molpcba"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "ap (ogb)" 
        metric_method = get_evaluator(args.dataset)
    else:
        raise NotImplementedError("No loss for this dataset")
    
    return {"loss": loss, "metric": metric, "metric_method": metric_method}

def get_evaluator(dataset):
    evaluator = Evaluator(dataset)
    eval_method = lambda y_true, y_pred: evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return eval_method