import math

import torch
import torch.nn as nn
from e3nn import o3
from torch_scatter import scatter

from .gaussian_rbf import GaussianRadialBasisLayer
from .graph_attention import TransBlock
from .graph_norm import EquivariantGraphNorm
from .instance_norm import EquivariantInstanceNorm
from .irreps_projection import IrrepsProjection
from .layer_norm import EquivariantLayerNormV2
from .pair_net import Expansion, PairNetLayer, SelfNetLayer
from .radial_basis import RadialBasis
from .tensor_product_rescale import LinearRS


class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi * (2 * (distances - self.cutoff_lower) / (self.cutoff_upper - self.cutoff_lower) + 1.0)
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            cutoffs = cutoffs * (distances > self.cutoff_lower)
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            return cutoffs


# https://github.com/torchmd/torchmd-net/blob/main/torchmdnet/models/utils.py#L111
class ExpNormalSmearing(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=False):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class EquiformerFunction(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=5.0,
        number_of_basis=128,
        basis_type="gaussian",
        fc_neurons=[64, 64],
        irreps_feature="512x0e",
        irreps_head="32x0e+16x1o+8x2e",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        irreps_mlp_mid="128x0e+64x1e+32x2e",
        irrep_bottle_hidden="128x0e+128x1e+128x2e+128x3e+128x4e",
        irreps_ham="32x0e+32x1e+32x2e+32x3e",
        irreps_out="3x0e+2x1e+1x2e",
        use_attn_head=False,
        norm_layer="layer",
        alpha_drop=0.2,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        mean=None,
        std=None,
        scale=None,
        atomref=None,
        deq_func=True,
    ):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.use_attn_head = use_attn_head
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.deq_func = deq_func
        self.register_buffer("atomref", atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_ham = o3.Irreps(irreps_ham)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = (
            o3.Irreps(irreps_sh) if irreps_sh is not None else o3.Irreps.spherical_harmonics(self.lmax)
        )
        self.irreps_bottle_hidden = o3.Irreps(irrep_bottle_hidden)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)

        self.basis_type = basis_type

        self.blocks = torch.nn.ModuleList()
        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        self.diag_lins = nn.ModuleList()
        self.non_diag_lins = nn.ModuleList()

        # special irreps
        irrep_output = o3.Irreps(irreps_out)  # irreps for def2-SVP basis

        self.build_blocks()

        #######################
        # deexpansion modules #
        #######################

        sphere_channels = self.irreps_node_embedding.dim
        hs = 128

        if self.deq_func:
            self.irreps_proj_ii = IrrepsProjection(self.irreps_ham, irrep_output, irrep_output)

            self.fc_ii_de = torch.nn.Sequential(
                nn.Linear(sphere_channels, hs), nn.SiLU(), nn.Linear(hs, self.irreps_proj_ii.num_path_weight)
            )
            self.fc_ii_bias_de = torch.nn.Sequential(
                nn.Linear(sphere_channels, hs), nn.SiLU(), nn.Linear(hs, self.irreps_proj_ii.num_bias)
            )
            self.irreps_proj_ij = IrrepsProjection(self.irreps_ham, irrep_output, irrep_output)
            self.fc_ij_de = torch.nn.Sequential(
                nn.Linear(sphere_channels * 2, hs), nn.SiLU(), nn.Linear(hs, self.irreps_proj_ij.num_path_weight)
            )
            self.fc_ij_bias_de = torch.nn.Sequential(
                nn.Linear(sphere_channels * 2, hs), nn.SiLU(), nn.Linear(hs, self.irreps_proj_ij.num_bias)
            )
        orbital_mask = self.get_orbital_mask(full_orbitals=irrep_output.dim)
        for key, tensor in orbital_mask.items():
            self.register_buffer(f"orbital_mask_{key}", tensor, persistent=False)

        #####################
        # expansion modules #
        #####################

        self.expand_ii, self.expand_ij, self.fc_ii, self.fc_ij, self.fc_ii_bias, self.fc_ij_bias = (
            nn.ModuleDict(),
            nn.ModuleDict(),
            nn.ModuleDict(),
            nn.ModuleDict(),
            nn.ModuleDict(),
            nn.ModuleDict(),
        )
        for name in {"hamiltonian"}:
            # input_expand_ii = o3.Irreps(
            #     f"{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e")

            self.expand_ii[name] = Expansion(self.irreps_bottle_hidden, irrep_output, irrep_output)
            self.fc_ii[name] = torch.nn.Sequential(
                nn.Linear(sphere_channels, hs), nn.SiLU(), nn.Linear(hs, self.expand_ii[name].num_path_weight)
            )
            self.fc_ii_bias[name] = torch.nn.Sequential(
                nn.Linear(sphere_channels, hs), nn.SiLU(), nn.Linear(hs, self.expand_ii[name].num_bias)
            )

            self.expand_ij[name] = Expansion(
                self.irreps_bottle_hidden,
                irrep_output,
                irrep_output,
            )

            self.fc_ij[name] = torch.nn.Sequential(
                nn.Linear(sphere_channels * 2, hs), nn.SiLU(), nn.Linear(hs, self.expand_ij[name].num_path_weight)
            )

            self.fc_ij_bias[name] = torch.nn.Sequential(
                nn.Linear(sphere_channels * 2, hs), nn.SiLU(), nn.Linear(hs, self.expand_ij[name].num_bias)
            )

        self.output_ii = LinearRS(self.irreps_bottle_hidden, self.irreps_bottle_hidden)
        self.output_ij = LinearRS(self.irreps_bottle_hidden, self.irreps_bottle_hidden)
        # self.apply(self._init_weights)

    def set_masked_finetune(self, n_atoms, full_orbitals):
        self.molecule_diag_mask_weights = nn.Parameter(
            torch.ones(n_atoms, full_orbitals, full_orbitals), requires_grad=True
        )
        self.molecule_non_diag_mask_weights = nn.Parameter(
            torch.ones(n_atoms * (n_atoms - 1), full_orbitals, full_orbitals), requires_grad=True
        )

        self.molecule_diag_mask_bias = nn.Parameter(
            torch.zeros(n_atoms, full_orbitals, full_orbitals), requires_grad=True
        )
        self.molecule_non_diag_mask_bias = nn.Parameter(
            torch.zeros(n_atoms * (n_atoms - 1), full_orbitals, full_orbitals), requires_grad=True
        )

    def get_orbital_mask(self, full_orbitals=14):
        if full_orbitals == 14:
            idx_1s_2s = torch.tensor([0, 1])
            idx_2p = torch.tensor([3, 4, 5])
            orbital_mask_line1 = torch.cat([idx_1s_2s, idx_2p])
            orbital_mask_line2 = torch.arange(14)
            orbital_mask = {}
            for i in range(1, 11):
                orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2
            # atom_index 1/2: [0, 1, 3, 4, 5] 1s 2s 2p
            # atom_index 3-11: [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13] 1s 2s 3s 2p 3p 3d
        elif full_orbitals == 18:
            orbital_mask = {}
            idx_line1 = torch.tensor([0, 1, 4, 5, 6])
            idx_line2 = torch.tensor([0, 1, 2, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17])
            orbital_mask_line1 = idx_line1
            orbital_mask_line2 = idx_line2
            orbital_mask_line3 = torch.arange(full_orbitals)
            for i in range(1, 18):
                if i <= 2:
                    orbital_mask[i] = orbital_mask_line1
                elif i <= 10:
                    orbital_mask[i] = orbital_mask_line2
                else:
                    orbital_mask[i] = orbital_mask_line3
            # atom_index 1/2: [0, 1, 4, 5, 6] 1s 2s 2p
            # atom_index 3-11: [0, 1, 2, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17] 1s 2s 3s 2p 3p 3d
        else:
            raise NotImplementedError
        return orbital_mask

    def build_blocks(self):
        for i in range(self.num_layers):
            blk = TransBlock(
                irreps_node_input=self.irreps_node_embedding,
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_node_output=self.irreps_node_embedding,
                fc_neurons=self.fc_neurons,
                irreps_head=self.irreps_head,
                num_heads=self.num_heads,
                irreps_pre_attn=self.irreps_pre_attn,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=self.norm_layer,
            )
            if i > self.num_layers - 3:
                self.e3_gnn_node_layer.append(
                    SelfNetLayer(
                        irrep_in_node=self.irreps_bottle_hidden,
                        irrep_bottle_hidden=self.irreps_bottle_hidden,
                        irrep_out=self.irreps_bottle_hidden,
                        sh_irrep=None,
                        edge_attr_dim=None,
                        node_attr_dim=None,
                        resnet=True,
                    )
                )
                self.e3_gnn_node_pair_layer.append(
                    PairNetLayer(
                        irrep_in_node=self.irreps_bottle_hidden,
                        irrep_bottle_hidden=self.irreps_bottle_hidden,
                        irrep_out=self.irreps_bottle_hidden,
                        sh_irrep=None,
                        edge_attr_dim=self.number_of_basis,
                        node_attr_dim=None,
                        invariant_layers=1,
                        resnet=True,
                    )
                )
            if self.deq_func:
                self.diag_lins.append(LinearRS(self.irreps_ham, self.irreps_node_embedding))
                self.non_diag_lins.append(LinearRS(self.irreps_ham, self.irreps_node_embedding))
            self.blocks.append(blk)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormV2,
                    EquivariantInstanceNorm,
                    EquivariantGraphNorm,
                    GaussianRadialBasisLayer,
                    RadialBasis,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and "weight" in parameter_name:
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)

    # the gradient of energy is following the implementation here:
    # https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/spinconv.py#L186
    # @torch.enable_grad()
    def forward(self, z, batch, **kwargs):
        edge_src, edge_dst = batch.edge_index
        edge_sh = batch.vec_feat_sh
        edge_src_all, edge_dst_all = batch.edge_index_all
        edge_sh_all = batch.vec_feat_sh_all
        node_features = batch.node_emb.clone()
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        edge_length_embedding = batch.dis_feat.clone()

        ##################################
        # deexpansion of pre_hamiltonian #
        ##################################
        # z must be a tensor; batch.z saves hamiltonian matrix and mask before concat.
        # init fii, fij with input Linear(IrrepsProjection(z))
        ori_node_emb = batch.node_emb.clone()
        node_pair_embedding = torch.cat(
            [ori_node_emb[batch.edge_index_all[0]], ori_node_emb[batch.edge_index_all[1]]], dim=-1
        )

        if self.deq_func:
            diag_ham, non_diag_ham = z[: len(batch.diag_ham_mask)], z[len(batch.diag_ham_mask) :]
            diag_feat = self.irreps_proj_ii(diag_ham, self.fc_ii_de(ori_node_emb), self.fc_ii_bias_de(ori_node_emb))

            non_diag_feat = self.irreps_proj_ij(
                non_diag_ham, self.fc_ij_de(node_pair_embedding), self.fc_ij_bias_de(node_pair_embedding)
            )

        #######################
        # equiformer backbone #
        #######################
        # z must be a tensor; batch.z saves hamiltonian matrix and mask before concat.
        # init fii, fij with input Linear(IrrepsProjection(z))
        fii, fij = None, None

        for i, blk in enumerate(self.blocks):
            if self.deq_func:
                x_hami = self.diag_lins[i](diag_feat) + scatter(
                    self.non_diag_lins[i](non_diag_feat), batch.edge_index_all[0], dim=0, dim_size=len(diag_feat)
                )
            else:
                x_hami = 0

            node_features = blk(
                node_input=node_features + x_hami,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=batch,
            )
            if i > self.num_layers - 3:
                fii = self.e3_gnn_node_layer[i - self.num_layers + 2](batch, node_features, fii)
                fij = self.e3_gnn_node_pair_layer[i - self.num_layers + 2](batch, node_features, fij)

        # node_features = self.norm(node_features, batch=batch)

        ##########################
        # get hamiltonian matrix #
        ##########################

        # get hamiltonian matrix with node_emb and node_pair_embedding
        fii = self.output_ii(fii)
        fij = self.output_ij(fij)

        hamiltonian_diagonal_matrix = self.expand_ii["hamiltonian"](
            fii, self.fc_ii["hamiltonian"](ori_node_emb), self.fc_ii_bias["hamiltonian"](ori_node_emb)
        )
        hamiltonian_non_diagonal_matrix = self.expand_ij["hamiltonian"](
            fij, self.fc_ij["hamiltonian"](node_pair_embedding), self.fc_ij_bias["hamiltonian"](node_pair_embedding)
        )

        if hasattr(self, "molecule_diag_mask_weights"):
            batchsize = len(batch.ptr) - 1
            hamiltonian_diagonal_matrix = hamiltonian_diagonal_matrix * self.molecule_diag_mask_weights.repeat(
                batchsize, 1, 1
            ) + self.molecule_diag_mask_bias.repeat(batchsize, 1, 1)
            hamiltonian_non_diagonal_matrix = (
                hamiltonian_non_diagonal_matrix * self.molecule_non_diag_mask_weights.repeat(batchsize, 1, 1)
                + self.molecule_non_diag_mask_bias.repeat(batchsize, 1, 1)
            )

        # the transpose should considers the i, j
        ret_hamiltonian_diagonal_matrix = hamiltonian_diagonal_matrix + hamiltonian_diagonal_matrix.transpose(-1, -2)
        ret_hamiltonian_non_diagonal_matrix = hamiltonian_non_diagonal_matrix + hamiltonian_non_diagonal_matrix[
            batch["all_transpose_index"]
        ].transpose(-1, -2)

        results = torch.concat(
            [ret_hamiltonian_diagonal_matrix, ret_hamiltonian_non_diagonal_matrix], dim=0
        ) * torch.concat([batch["diag_ham_mask"], batch["non_diag_ham_mask"]], dim=0)
        return results

    def build_final_matrix(self, data, diagonal_matrix, non_diagonal_matrix):
        # concate the blocks together and then select once.
        final_matrix = []
        dst, src = data.edge_index_all
        data.atoms = data.atoms.long()

        non_diagonal_idx = 0
        for graph_idx in range(data.ptr.shape[0] - 1):
            start, end = data.ptr[graph_idx], data.ptr[graph_idx + 1]
            natoms, len_non_diag = end - start, (end - start) * (end - start - 1)

            full_orbitals = diagonal_matrix.shape[-1]
            temp_final_matrix = torch.zeros(
                natoms * natoms,
                full_orbitals,
                full_orbitals,
            ).type_as(diagonal_matrix)
            orbital_mask = torch.tensor(data.orbital_mask[graph_idx]).type_as(data.ptr)
            temp_diagonal_matrix = diagonal_matrix[start:end]
            temp_non_diagonal_matrix = non_diagonal_matrix[non_diagonal_idx : non_diagonal_idx + len_non_diag]
            temp_src = src[non_diagonal_idx : non_diagonal_idx + len_non_diag] - data.ptr[graph_idx]
            temp_dst = dst[non_diagonal_idx : non_diagonal_idx + len_non_diag] - data.ptr[graph_idx]
            non_diagonal_idx += len_non_diag
            temp_non_diagonal_index = temp_dst * natoms + temp_src
            temp_final_matrix[temp_non_diagonal_index] = temp_non_diagonal_matrix
            temp_diagonal_index = torch.arange(natoms).type_as(natoms) * natoms + torch.arange(natoms).type_as(natoms)
            temp_final_matrix[temp_diagonal_index] = temp_diagonal_matrix
            temp_final_matrix = (
                temp_final_matrix.reshape(natoms, natoms, full_orbitals, full_orbitals)
                .permute(0, 2, 1, 3)
                .reshape(natoms * full_orbitals, natoms * full_orbitals)
            )
            temp_final_matrix = temp_final_matrix.index_select(-2, orbital_mask).index_select(-1, orbital_mask)
            final_matrix.append(temp_final_matrix)
        try:
            final_matrix = torch.stack(final_matrix, dim=0)
        except RuntimeError:
            pass
        return final_matrix

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
