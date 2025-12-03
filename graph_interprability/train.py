from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from torch_geometric.loader import DataLoader
import argparse
from tqdm import tqdm

from graph_interprability_data import load_cached_npz
from gnn_model import DiffPoolGNN
from sklearn.model_selection import train_test_split


def load_data(
    batch_size: int = 8,
    num_cities=None
) -> DataLoader:
    num_cities = num_cities or -1
    
    data_dir = Path(__file__).resolve().parent.parent / "travel2" / "TAP-city"
    npz_files = sorted(data_dir.glob("*.npz"))[:num_cities]
    
    graphs = [
        load_cached_npz(str(path), data_dir)
        for path in tqdm(npz_files, desc="Loading TAP-city graphs")
    ]
    
    train_graphs, temp_graphs = train_test_split(
        graphs, test_size=0.30, shuffle=True, random_state=42
    )

    val_graphs, test_graphs = train_test_split(
        temp_graphs, test_size=0.50, shuffle=True, random_state=42
    )

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size)

    return train_loader, val_loader, test_loader



class LitDiffPool(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, batch):
        return self.model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )

    def training_step(self, batch, batch_idx):
        pred = self(batch).squeeze()
        loss = F.mse_loss(pred, batch.y.squeeze())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch).squeeze()
        loss = F.mse_loss(pred, batch.y.squeeze())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


node_feat_dim = 8
edge_feat_dim = 6
lit_model = LitDiffPool(model=DiffPoolGNN(
    in_node_dim=node_feat_dim,
    in_edge_dim=edge_feat_dim,
    hidden_dim=64,
    embed_dim=128
))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run TRAVEL experiments")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--num_cities", type=int, default=10, help="Number of cities to train")
    args = parser.parse_args()
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=device,   
        devices=1,
        log_every_n_steps=1
    )
    train_loader, val_loader, _ = load_data(num_cities=args.num_cities)
    trainer.fit(lit_model, train_loader, val_loader)
