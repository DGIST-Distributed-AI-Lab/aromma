import os, sys

base_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
)
sys.path.append(base_path)

import torch
import torch.nn as nn
import torch_geometric as pyg
from src.model.pom.utils.model import GraphNets, GLM, EndToEndModule
from src.model.pom.utils import graph_utils
from src.model.pom.utils.data import GraphDataset
from rdkit import Chem


def make_nonstereo(smiles):
    smiles = smiles.replace("@", "")
    smiles = smiles.replace("/", "")
    smiles = smiles.replace("\\", "")
    return smiles


class POM(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        self.output_dim = 196

    def load_model(self):
        self.model = EndToEndModule(
            GraphNets(
                node_dim=85,
                edge_dim=14,
                hidden_dim=320,
                global_dim=196,
                depth=4,
                dropout=0.1,
            ),
            GLM(input_dim=196, output_dim=138),
        ).to(self.device)

        gnn_state = torch.load(os.path.join(base_path, "models/pom/gnn_embedder.pt"), map_location=self.device)
        self.model.gnn_embedder.load_state_dict(gnn_state)
        org_glm_state = torch.load(os.path.join(base_path, "models/pom/nn_predictor.pt"), map_location=self.device)
        glm_state = {}
        for k, v in org_glm_state.items(): # models.gs-lf.linear.weight => linear.weight, models.gs-lf.linear.bias => linear.bias

            if k.startswith("models.gs-lf."):
                new_key = k.replace("models.gs-lf.", "")
                glm_state[new_key] = v
        self.model.nn_predictor.load_state_dict(glm_state)
        self.model.eval()

    def forward(self, smiles, return_emb=False):
        graphs = [
            graph_utils.from_smiles(
                Chem.MolToSmiles(Chem.MolFromSmiles(make_nonstereo(s)), canonical=True)
            )
            for s in smiles
        ]
        dataset = GraphDataset(graphs, [0.0] * len(smiles))
        loader = pyg.loader.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        data, _ = next(iter(loader))

        with torch.no_grad():
            u = self.model.gnn_embedder(data.to(self.device))

            if return_emb:
                return u
            logits = self.model.nn_predictor(u)

        return logits
