import torch.nn as nn
from e3nn import o3

_MAX_ATOM_TYPE = 64  # Set to some large value
_AVG_DEGREE = 12  # For simplicity, use the same statistics for MD17

from .graph_attention import EdgeDegreeEmbeddingNetwork, NodeEmbeddingNetwork


class EquiformerEmbedding(nn.Module):
    def __init__(
        self,
        number_of_basis=128,
        fc_neurons=[64, 64],
        irreps_node_embedding="128x0e+64x1e+32x2e",
        irreps_sh="1x0e+1x1e+1x2e",
        dtype="float32",
    ):
        super().__init__()

        self.number_of_basis = number_of_basis
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE, dtype=dtype)
        self.irreps_edge_attr = (
            o3.Irreps(irreps_sh) if irreps_sh is not None else o3.Irreps.spherical_harmonics(self.lmax)
        )
        self.fc_neurons = [self.number_of_basis] + fc_neurons

        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(
            self.irreps_node_embedding, self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE
        )

    def forward(self, node_atom, dis_feat, edge_sh, edge_index, pos=None):
        edge_dst, edge_src = edge_index
        atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom)
        edge_length_embedding = dis_feat
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst)
        node_features = atom_embedding + edge_degree_embedding
        return node_features
