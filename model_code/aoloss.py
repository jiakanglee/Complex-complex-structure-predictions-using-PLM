import torch
import numpy as np
from Bio.PDB import PDBParser
from typing import List
from openfold.np import residue_constants 
from uti import pdb_to_tensor,compute_lddt,compute_distogram,get_asym_mask,pseudo_beta_fn,insert_zero_matrices
from torch import nn
from openfold.utils.feats import atom14_to_atom37

def one_hot(x, num_classes, dtype=torch.float32):
    x_one_hot = torch.zeros(*x.shape, num_classes, dtype=dtype, device=x.device)
    x_one_hot.scatter_(-1, x.long().unsqueeze(-1), 1)
    return x_one_hot

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss

def masked_mean(mask, value, dim, eps=1e-10, keepdim=False):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim, keepdim=keepdim) / (
        eps + torch.sum(mask, dim=dim, keepdim=keepdim)
    )

def lddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    # resolution: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]

    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim
    logits = logits[..., ca_pos, :]

    score = compute_lddt(
        all_atom_pred_pos, 
        all_atom_positions, 
        all_atom_mask, 
        cutoff=cutoff, 
        eps=eps
    )

    score = score.detach()

    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(
        bin_index, num_classes=no_bins
    )

    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (
        eps + torch.sum(all_atom_mask, dim=-1)
    )

    # loss = loss * (
    #     (resolution >= min_resolution) & (resolution <= max_resolution)
    # )

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss

def distogram_loss(
    logits,
    pseudo_beta,
    pseudo_beta_mask,
    min_bin=2.3125,
    max_bin=21.6875,
    num_bins=64,
    eps=1e-6,
    loss_dict=None,
    **kwargs,
):
    distogram, mask = compute_distogram(
        pseudo_beta, pseudo_beta_mask, min_bin, max_bin, num_bins)

    errors = softmax_cross_entropy(logits, one_hot(distogram, num_bins))

    loss = masked_mean(mask, errors, dim=(-1, -2), eps=eps)
    
    return loss

def chain_centre_mass_loss(
    pred_atom_positions: torch.Tensor,
    true_atom_positions: torch.Tensor,
    atom_mask: torch.Tensor,
    asym_id: torch.Tensor,
    eps: float = 1e-10,
    # loss_dict=None,
    **kwargs,
) -> torch.Tensor:

    ca_pos = residue_constants.atom_order["CA"]
    pred_atom_positions = pred_atom_positions[..., ca_pos, :].float()  # [B, NR, 3]
    true_atom_positions = true_atom_positions[..., ca_pos, :].float()  # [B, NR, 3]
    atom_mask = atom_mask[..., ca_pos].bool()  # [B, NR]
    assert len(pred_atom_positions.shape) == 3

    asym_mask = get_asym_mask(asym_id) * atom_mask[..., None, :]  # [B, NC, NR]
    asym_exists = torch.any(asym_mask, dim=-1).float()  # [B, NC]

    def get_asym_centres(pos):
        pos = pos[..., None, :, :] * asym_mask[..., :, :, None]  # [B, NC, NR, 3]
        return torch.sum(pos, dim=-2) / (torch.sum(asym_mask, dim=-1)[..., None] + eps)

    pred_centres = get_asym_centres(pred_atom_positions)  # [B, NC, 3]
    true_centres = get_asym_centres(true_atom_positions)  # [B, NC, 3]

    def get_dist(p1: torch.Tensor, p2: torch.Tensor):
        return torch.sqrt(
            (p1[..., :, None, :] - p2[..., None, :, :]).square().sum(-1) + eps
        )

    pred_centres2 = pred_centres
    true_centres2 = true_centres
    pred_dists = get_dist(pred_centres, pred_centres2)  # [B, NC, NC]
    true_dists = get_dist(true_centres, true_centres2)  # [B, NC, NC]
    losses = (pred_dists - true_dists).square() * 0.0025
    # print(losses)
    loss_mask = asym_exists[..., :, None] * asym_exists[..., None, :]  # [B, NC, NC]

    loss = masked_mean(loss_mask, losses, dim=(-1, -2))
    # loss_dict["chain_centre_loss"] = loss.data

    return loss


class AlphaFoldLoss_change(nn.Module):
    def __init__(self):
        super(AlphaFoldLoss_change, self).__init__()
        
    def forward(self, output, realpdb_file,gra_position):
        final_position = atom14_to_atom37(output["positions"][-1], output)
        aatype = output["aatype"]
        distogram_logits = output["distogram_logits"]
        all_atom_pred_pos = final_position
        all_atom_position,aysm_id = pdb_to_tensor(realpdb_file)
        # print(all_atom_position.shape,aysm_id.shape)
        all_atom_position,aysm_id = insert_zero_matrices(all_atom_position,aysm_id,gra_position)
        # print(all_atom_position.shape,aysm_id.shape)
        all_atom_position = all_atom_position.cuda()
        aysm_id = aysm_id.cuda()
        
        all_atom_mask = (output["atom37_atom_exists"])
        # assert all_atom_position.shape == all_atom_pred_pos.shape, f"all_atom_position和all_atom_pred_pos的形状不相等,文件夹名称,{realpdb_file}"
        if all_atom_position.shape != all_atom_pred_pos.shape:
            loss = torch.ones(1,dtype=float)
            loss = loss.cuda()
            return loss
        
        #lddt_loss
        logits = output['lddt_head'][-1]
        result1 = lddt_loss(logits,all_atom_pred_pos,all_atom_position,all_atom_mask)
        # print(result1)
        
        #pseudo_loss
        pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(aatype,all_atom_position,all_atom_mask)
        result2 = distogram_loss(distogram_logits,pseudo_beta,pseudo_beta_mask)
        # print(result2)
        
        #chain_center_loss
        result3 = chain_centre_mass_loss(all_atom_pred_pos,all_atom_position,all_atom_mask,aysm_id)
        # print(result3)
        
        loss = result1 + result2 + result3
        
        return loss