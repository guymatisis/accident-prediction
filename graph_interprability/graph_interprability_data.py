from pathlib import Path
from typing import Optional, Callable

import numpy as np
import os.path as osp
import pandas as pd
import torch
import networkx as nx

from pyproj import Transformer
from datetime import datetime

from torch_geometric.data import Data, InMemoryDataset, download_url


# node_features = \
#     [
#         lat_norm,
#         lon_norm,
#         node_degree,
#         betweenness,
#         neighbor_density,
#         log_accident_count,
#         day_night_ratio,
#         weekend_ratio
#     ]
# edge_features = \
#     [
#         direction,
#         angle,
#         length,
        
#     ]

def latlon_to_meters(coords):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    lats = coords[:, 0]
    lons = coords[:, 1]

    xs, ys = transformer.transform(lons, lats)
    
    return np.column_stack([xs, ys])
 


def build_graph(edge_index, num_nodes):
    src, dst = edge_index.numpy()
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(src, dst))
    return G


def normalize_geo(coords):
    lat = coords[:, 0]
    lon = coords[:, 1]

    lat_norm = (lat - lat.mean()) / (lat.std() + 1e-8)
    lon_norm = (lon - lon.mean()) / (lon.std() + 1e-8)

    return np.stack([lat_norm, lon_norm], axis=1)

def get_node_degree(edge_index):
    _, counts = np.unique(edge_index, return_counts=True)
    return counts

def get_betweenness(G, num_nodes, k=100):
    bc = nx.betweenness_centrality(G, k=min(k, num_nodes), seed=42)

    # convert dict → array
    bc_arr = np.zeros(num_nodes, dtype=np.float32)
    for n, v in bc.items():
        bc_arr[n] = v
    return bc_arr

def get_neighbor_density(xy_coords, radius=100):
    
    N = xy_coords.shape[0]
    density = np.zeros(N, dtype=np.int32)

    for i in range(N):
        d2 = np.sum((xy_coords - xy_coords[i])**2, axis=1)
        density[i] = np.sum(d2 <= radius*radius) - 1  # exclude self

    return density


def get_log_accident_count(cnt_labels):
    acc_counts = np.asarray(cnt_labels, dtype=float)
    return np.log1p(acc_counts)

def get_day_night_ratio(crash_time):
    N = len(crash_time)
    ratio = np.zeros(N, dtype=float)

    # use start time for simplicity
    start_crash_time = crash_time.T[0]

    for i in range(N):
        ts = start_crash_time[i]
        if not any(ts):
            ratio[i] = 0.0
            continue

        day = 0
        night = 0

        for t in ts:
            t = datetime.fromisoformat(t.replace("Z", ""))

            h = t.hour
            if 6 <= h < 18:
                day += 1
            else:
                night += 1

        ratio[i] = day / (night + 1)

    return ratio


def get_weekend_ratio(crash_time):
    N = len(crash_time)
    ratio = np.zeros(N, dtype=float)

    # use start time for simplicity
    start_crash_time = crash_time.T[0]
    for i in range(N):
        ts = start_crash_time[i]
        if not any(ts):
            ratio[i] = 0.0
            continue

        weekend = 0
        weekday = 0

        for t in ts:
            t = datetime.fromisoformat(t.replace("Z", ""))

            # Monday=0 ... Sunday=6
            d = t.weekday()
            if d >= 5:   # Sat=5, Sun=6
                weekend += 1
            else:
                weekday += 1

        ratio[i] = weekend / (weekday + 1)

    return ratio

def get_edge_length(xy_coords, edge_index):
    edge_from = xy_coords[edge_index[0]]
    edge_to = xy_coords[edge_index[1]]
    
    return np.sqrt(np.sum((edge_from - edge_to) ** 2, axis=-1))
    
def normalize_feature(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-8)

def read_proccess_npz(path):
    with np.load(path, allow_pickle=True) as f:
        crash_time = f['crash_time']
        coords = torch.from_numpy(f['coordinates']).to(torch.float)
        cnt_labels = torch.from_numpy(f['cnt_labels']).to(torch.long)
        edge_attr_dir = torch.from_numpy(f['edge_attr_dir']).to(torch.float)
        edge_attr_ang = torch.from_numpy(f['edge_attr_ang']).to(torch.float)
        severity_labels = torch.from_numpy(f['severity_8labels']).to(torch.long)
        edge_index = torch.from_numpy(f['edge_index']).to(torch.long).t().contiguous()
        
        # precursors to features
        coords = np.c_[coords[:,1], coords[:,0]]
        num_nodes = len(coords)
        xy_coords = latlon_to_meters(coords)
        G = build_graph(edge_index, num_nodes)
        
        # edge features
        edge_length = normalize_feature(get_edge_length(xy_coords, edge_index))
        edge_attr_dir
        edge_attr_ang
        #node features
        weekend_ratio = get_weekend_ratio(crash_time)
        day_night_ratio = get_day_night_ratio(crash_time)
        log_accident_count = get_log_accident_count(cnt_labels)
        neighbor_density = normalize_feature(get_neighbor_density(xy_coords))
        betweenness = normalize_feature(get_betweenness(G, num_nodes))
        node_degree = get_node_degree(edge_index)
        normalize_coords = normalize_geo(coords)
        
                # node features
        node_features = np.column_stack([
            weekend_ratio,
            day_night_ratio,
            log_accident_count,
            neighbor_density,
            betweenness,
            node_degree,
            normalize_coords,
                        
        ])
        x = torch.from_numpy(node_features).to(torch.float32)
        
        edge_attr = np.column_stack([
            edge_length,
            edge_attr_dir,
            edge_attr_ang,
        ])
        edge_attr = torch.from_numpy(edge_attr).to(torch.float32)

        y = torch.tensor([torch.sum(cnt_labels)], dtype=torch.float32)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )


PROCESSED_DATA_DIR = Path(__file__).resolve().parent / "processed_data"


def load_cached_npz(path: str, data_dir) -> Data:
    """Return cached processed graph if present, otherwise process and cache."""
    import os
    src_path = Path(path)
    cache_path = PROCESSED_DATA_DIR / f"{src_path.stem}.pt"

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    
    if not os.path.exists(path):
        template = 'https://github.com/baixianghuang/travel/raw/main/TAP-city/{}'
        fname = path.split('/')[-1]
        url = template.format(fname)
        download_url(url, data_dir)

    data = read_proccess_npz(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    return data

