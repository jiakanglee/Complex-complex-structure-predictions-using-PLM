import torch
import esm
# from ps import ESMFold
from esm.esmfold.v1.esmfold import ESMFold
from openfold.np import residue_constants as rc
from aoloss import AlphaFoldLoss_change

model_data = torch.load("/root/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt",map_location="cpu")
cfg = model_data["cfg"]["model"]
model_state = model_data["model"]
# model_state = torch.load('model_weights.pt',map_location="cpu")
model = ESMFold(esmfold_config=cfg)

expected_keys = set(model.state_dict().keys())
found_keys = set(model_state.keys())

missing_essential_keys = []
for missing_key in expected_keys - found_keys:
    if not missing_key.startswith("esm."):
        missing_essential_keys.append(missing_key)

if missing_essential_keys:
    raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")

model.load_state_dict(model_state, strict=False)

model = model.eval().cuda()

sequence_file = 'Benchmark2-multimer_fasta/7aye.fasta'
sequences = []
with open(sequence_file, "r") as fasta:
    for seq in fasta:
        seq = seq.strip()
        if seq and not seq.startswith(">"):
            sequences.append(seq)

# Join the sequences using ':' as a separator
sequence = ":".join(sequences)
# sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

gra_position = [idx for idx, char in enumerate(sequence) if char == ':']

with torch.no_grad():
    output = model.infer(sequence)

    # for key, tensor in output.items():
    #     print(f"Key: {key}, Tensor Shape: {tensor.shape}")

    pdb_file = "Benchmark2-multimer_pdb/7aye.pdb"
    loss_f = AlphaFoldLoss_change()
    loss = loss_f(output,pdb_file,gra_position)
    print(loss)
with torch.no_grad():
    output = model.infer_pdb(sequence)

with open("7aye.pdb", "w") as f:
    f.write(output)

import biotite.structure.io as bsio
struct = bsio.load_structure("7aye.pdb", extra_fields=["b_factor"])
print(struct.b_factor.mean())  # this will be the pLDDT