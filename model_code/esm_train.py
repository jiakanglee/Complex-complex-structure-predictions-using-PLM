from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
# from esm.esmfold.v1.esmfold import ESMFold
from ps import ESMFold
from aoloss import AlphaFoldLoss_change

class trainer(pl.LightningModule):
    def __init__(self):
        super(trainer, self).__init__()
        model_data = torch.load("/root/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt",map_location="cpu")
        cfg = model_data["cfg"]["model"]
        # model_state = model_data["model"]
        # model_state = torch.load("lightning_logs/version_2/checkpoints/best_model.ckpt",map_location="cpu")
        model_state = torch.load("model_weights.pt",map_location="cpu")
        model = ESMFold(esmfold_config=cfg)
        # model.load_state_dict(model_state, strict=False)

        expected_keys = set(model.state_dict().keys())
        found_keys = set(model_state.keys())

        missing_essential_keys = []
        for missing_key in expected_keys - found_keys:
            if not missing_key.startswith("esm."):
                missing_essential_keys.append(missing_key)
        if missing_essential_keys:
            raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")
        model.load_state_dict(model_state, strict=False)
        self.model = model
        self.loss = AlphaFoldLoss_change()
        
    def forward(self, batch):
        output = self.model.infer(batch['sequence'][0])
        return output
    
    def training_step(self, batch,batch_idx):
        output = self.model.infer(batch['sequence'][0])
        # print(batch['pdb_file'][0])
        gra_position = [idx for idx, char in enumerate(batch['sequence'][0]) if char == ':']
        loss = self.loss(output,batch['pdb_file'][0],gra_position)
        # loss = torch.nan_to_num(loss,nan=10.0)
        self.log('train_loss',loss)
        loss.requires_grad_(True)
        return loss
    
    def validation_step(self, batch,batch_idx):
        output = self.model.infer(batch['sequence'][0])
        print(batch['pdb_file'][0])
        gra_position = [idx for idx, char in enumerate(batch['sequence'][0]) if char == ':']
        loss = self.loss(output,batch['pdb_file'][0],gra_position)
        self.log('val_loss',loss)
        loss.requires_grad_(True)
        
        return loss
    
    def configure_optimizers(self, 
        learning_rate: float = 1e-3,
        eps: float = 1e-5,
    ) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            eps=eps
        )
        return optimizer
    
    
