import torch
from e3nn import o3
from torch import nn


def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out


# TBD: w.o. weight
class IrrepsProjection(nn.Module):
    def __init__(self, irrep_out, irrep_in_1, irrep_in_2):
        super().__init__()
        self.irrep_in_1 = irrep_in_1
        self.irrep_in_2 = irrep_in_2
        self.irrep_out = irrep_out
        self.instructions = self.get_expansion_path(irrep_out, irrep_in_1, irrep_in_2)
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum([ins[-1][0] for ins in self.instructions if ins[1:3] == [0, 0]])
        self.num_weights = self.num_path_weight + self.num_bias

    def forward(self, x_in, weights, bias_weights=None):
        batch_num = x_in.shape[0]

        x_in_s = []
        for slice1 in self.irrep_in_1.slices():
            for slice2 in self.irrep_in_2.slices():
                x_in_s.append(x_in[:, slice1, slice2])
        outputs = {}
        flat_weight_index = 0
        bias_weight_index = 0
        for i, ins in enumerate(self.instructions):
            idx = ins[1] * len(self.irrep_in_1) + ins[2]
            w3j_matrix = o3.wigner_3j(ins[1], ins[2], ins[0]).type_as(x_in)
            x1 = x_in_s[idx].reshape(batch_num, ins[-1][1], w3j_matrix.shape[0], ins[-1][2], w3j_matrix.shape[1])

            if weights is not None:
                weight = weights[:, flat_weight_index : flat_weight_index + prod(ins[-1])].reshape([-1] + ins[-1])
                result = torch.einsum("bwuv, buivj-> bwij", weight, x1)
                if ins[1:3] == [0, 0] and bias_weights is not None:
                    bias_weight = bias_weights[:, bias_weight_index : bias_weight_index + ins[-1][0]]
                    bias_weight_index += ins[-1][0]
                    result = result + bias_weight.reshape(bias_weight.shape + (1, 1))
                result = torch.einsum("ijk, bwij-> bwk", w3j_matrix, result) / (
                    self.irrep_in_1[ins[1]].mul * self.irrep_in_2[ins[2]].mul
                )
                result = result.reshape(batch_num, -1)
                flat_weight_index += prod(ins[-1])
            else:
                raise NotImplementedError
            key = ins[0] * ins[3]
            if key in outputs:
                outputs[key] = outputs[key] + result
            else:
                outputs[key] = result
        sorted_keys = sorted(outputs.keys())
        # sorted_values = [outputs[key] for key in sorted_keys]
        sorted_values = []
        for mul, ir in self.irrep_out:
            if ir.p * ir.l in outputs:
                sorted_values.append(outputs[ir.p * ir.l])
            else:
                sorted_values.append(torch.zeros((batch_num, mul * (2 * ir.l + 1))).type_as(x_in))
        return torch.concat(sorted_values, dim=-1)

    def get_expansion_path(self, irrep_out, irrep_in_1, irrep_in_2):
        instructions = []
        for i, (num_out, ir_out) in enumerate(irrep_out):
            for j, (num_in1, ir_in1) in enumerate(irrep_in_1):
                for k, (num_in2, ir_in2) in enumerate(irrep_in_2):
                    if ir_out in ir_in1 * ir_in2:
                        instructions.append(
                            # l3 l1 l2
                            [ir_out.l, ir_in1.l, ir_in2.l, True, ir_in1.p * ir_in2.p, [num_out, num_in1, num_in2]]
                        )
                        assert abs(ir_in2.l - ir_out.l) <= ir_in1.l and ir_in1.l <= ir_in2.l + ir_out.l
        return instructions
