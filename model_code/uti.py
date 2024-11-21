import numpy as np
import torch
from Bio.PDB import PDBParser
from typing import List
from openfold.np import residue_constants as rc

def pdb_to_tensor(pdb_file_path):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file_path)
    
    coordinates = []
    model = structure[0]
    for chain in model:
        for residue in chain:
            if residue.get_resname().strip() not in rc.residue_atoms.keys():
                continue
            atom_coordinates = np.zeros((37,3))
            for atom in residue:
                atom_coord = atom.get_coord()
                atom_type = atom.get_id()
                if atom_type in rc.atom_order.keys():
                    atom_po = rc.atom_order[atom_type]
                    atom_coordinates[atom_po] = atom_coord
            coordinates.append(atom_coordinates)
    
    coordinates = np.array(coordinates)
    atom_coordinates_tensor = torch.tensor(coordinates, dtype=torch.float32)
    atom_coordinates_tensor = atom_coordinates_tensor.unsqueeze(0)  # Add batch dimension
    
    asym_id = []
    chain_num = 1.0
    
    for chain in model:
        for residue in chain:
            if residue.get_resname().strip() not in rc.residue_atoms.keys():
                continue
            asym_id.append(chain_num)
        chain_num += 1.0
    asym_id = np.array(asym_id)
    asym_id_tensor = torch.tensor(asym_id, dtype=torch.float32)
    asym_id_tensor = asym_id_tensor.unsqueeze(0)
    
    return atom_coordinates_tensor,asym_id_tensor

def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def insert_zero_matrices(original_tensor,aysm_id, insert_positions):
    """
    在原张量的指定位置插入全零矩阵，并更新后续位置。

    Parameters:
    - original_tensor: 原始张量
    - insert_positions: 要插入全零矩阵的位置列表
    - zero_matrix: 要插入的全零矩阵

    Returns:
    - updated_tensor: 更新后的张量
    """

    zero_matrix = torch.zeros((1, 25, 37, 3))
    for position, insert_matrix in zip(insert_positions, [zero_matrix] * len(insert_positions)):
        original_tensor = torch.cat((original_tensor[:, :position, ...], insert_matrix, original_tensor[:, position:, ...]), dim=1)
        insert_positions = [pos + zero_matrix.shape[1] if pos > position else pos for pos in insert_positions]
    
    zero_matrix = torch.ones((1, 25))
    for position, insert_matrix in zip(insert_positions, [zero_matrix] * len(insert_positions)):
        aysm_id = torch.cat((aysm_id[:, :position, ...], insert_matrix, aysm_id[:, position:, ...]), dim=1)
        insert_positions = [pos + zero_matrix.shape[1] if pos > position else pos for pos in insert_positions]

    return original_tensor,aysm_id

def compute_lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_positions[..., None, :] - all_atom_positions[..., None, :, :])
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_pred_pos[..., None, :] - all_atom_pred_pos[..., None, :, :]) ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff) 
        * (all_atom_mask)
        * (permute_final_dims(all_atom_mask, (1, 0)))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )
    print(dists_to_score.shape)

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=-1))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=-1))

    return score

def compute_distogram(
    positions,
    mask,
    min_bin=2.3125,
    max_bin=21.6875,
    num_bins=64,
):
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        num_bins - 1,
        device=positions.device,
    )
    boundaries = boundaries**2
    positions = positions.float()

    dists = torch.sum(
        (positions[..., None, :] - positions[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    ).detach()

    mask = mask.float()
    pair_mask = mask[..., None] * mask[..., None, :]

    return torch.sum(dists > boundaries, dim=-1), pair_mask

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_mask):
    """Create pseudo beta features."""
    if aatype.shape[0] > 0:
        is_gly = torch.eq(aatype, rc.restype_order["G"])
        ca_idx = rc.atom_order["CA"]
        cb_idx = rc.atom_order["CB"]
        pseudo_beta = torch.where(
            torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
            all_atom_positions[..., ca_idx, :],
            all_atom_positions[..., cb_idx, :],
        )
    else:
        pseudo_beta = all_atom_positions.new_zeros(*aatype.shape, 3)
    if all_atom_mask is not None:
        if aatype.shape[0] > 0:
            pseudo_beta_mask = torch.where(
                is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx]
            )
        else:
            pseudo_beta_mask = torch.zeros_like(aatype).float()
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta

def get_asym_mask(asym_id):
    """get the mask for each asym_id. [*, NR] -> [*, NC, NR]"""
    # this func presumes that valid asym_id ranges [1, NC] and is dense.
    asym_type = torch.arange(1, torch.amax(asym_id) + 1, device=asym_id.device)  # [NC]
    return (asym_id[..., None, :] == asym_type[:, None]).float()

