import os
import glob
import torch
from torch.utils.data import DataLoader
from esm_train import trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# 设置序列文件夹和 pdb 文件夹的路径
sequence_folder2 = 'Benchmark2-multimer_fasta'
pdb_folder2 = 'Benchmark2-multimer_pdb'
sequence_folder = 'filter_fasta'
pdb_folder = 'filter_pdb'

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # 选择要监视的验证指标，通常是验证损失
    # dirpath='/path/to/save/models',  # 保存模型的目录
    filename='best_model',  # 模型文件名的模板
    save_top_k=1,  # 保存最佳模型的数量
    mode='min'  # 模式可以是 'min'（最小化监视的指标）或 'max'（最大化监视的指标）
)

# 获取序列文件夹中的所有文件路径
sequence_files = glob.glob(os.path.join(sequence_folder, '*.fasta'))

# 创建一个空列表，用于存储样本数据
data = []

# 遍历序列文件夹中的文件
for sequence_file in sequence_files:
    # 获取序列文件的文件名（不含路径）
    sequence_filename = os.path.basename(sequence_file)
    
    # 构造对应的 pdb 文件路径
    pdb_file = os.path.join(pdb_folder, sequence_filename.replace('.fasta', '.pdb'))
    # 检查对应的 pdb 文件是否存在
    if os.path.exists(pdb_file):
        # 读取序列文件内容
        sequences = []
        with open(sequence_file, "r") as fasta:
            for seq in fasta:
                seq = seq.strip()
                if seq and not seq.startswith(">"):
                    sequences.append(seq)

        # 如果sequences为空或只有一行，则直接跳过处理该文件
        if len(sequences) <= 1 or len(":".join(sequences)) > 600 :
            # print(f"Skipping {sequence_filename} - Invalid or empty FASTA content or sequence too long.")
            continue

        # Join the sequences using ':' as a separator
        sequence = ":".join(sequences)
        
        # 将序列文件和 pdb 文件路径组合成一个样本数据字典
        sample_data = {
            'sequence': sequence,
            'pdb_file': pdb_file,
        }
        
        # 将样本数据添加到数据列表中
        data.append(sample_data)
print(len(data))


sequence_files2 = glob.glob(os.path.join(sequence_folder2, '*.fasta'))

# 创建一个空列表，用于存储样本数据
data2 = []

# 遍历序列文件夹中的文件
for sequence_file in sequence_files2:
    # 获取序列文件的文件名（不含路径）
    sequence_filename = os.path.basename(sequence_file)
    
    # 构造对应的 pdb 文件路径
    pdb_file = os.path.join(pdb_folder2, sequence_filename.replace('.fasta', '.pdb'))
    # 检查对应的 pdb 文件是否存在
    if os.path.exists(pdb_file):
        # 读取序列文件内容
        sequences = []
        with open(sequence_file, "r") as fasta:
            for seq in fasta:
                seq = seq.strip()
                if seq and not seq.startswith(">"):
                    sequences.append(seq)

        # 如果sequences为空或只有一行，则直接跳过处理该文件
        if len(sequences) <= 1 or len(":".join(sequences)) > 600 :
            # print(f"Skipping {sequence_filename} - Invalid or empty FASTA content or sequence too long.")
            continue

        # Join the sequences using ':' as a separator
        sequence = ":".join(sequences)
        
        # 将序列文件和 pdb 文件路径组合成一个样本数据字典
        sample_data = {
            'sequence': sequence,
            'pdb_file': pdb_file,
        }
        
        # 将样本数据添加到数据列表中
        data2.append(sample_data)


train_loader = DataLoader(data, batch_size=1, shuffle=True)
# for batch in train_loader:
#     sequence_1 = batch["sequence"]
#     pdb_file_1 = batch["pdb_file"]
#     print(sequence_1)
#     print(pdb_file_1)
val_loader = DataLoader(data2, batch_size=1, shuffle=False)
model = trainer()
tr = pl.Trainer(max_epochs=3, gpus=1,callbacks = [checkpoint_callback])
#,val_check_interval = 100
tr.fit(model,train_dataloaders= val_loader ,val_dataloaders= val_loader,ckpt_path = "lightning_logs/version_1/checkpoints/best_model.ckpt")
# ckpt_path = "lightning_logs/version_2/checkpoints/best_model.ckpt"
torch.save(model.model.state_dict(), 'model_weights.pt')

# 现在，data 列表包含了每个样本的序列和对应的 pdb 文件路径