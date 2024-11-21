import os
import glob
import shutil

sequence_folder = 'Zhang_dataset/Protein_Complex'
pdb_folder = 'Zhang_dataset_tarred/protein_complex_decompressed'
filter_fasta_folder = 'filter_fasta'
filter_pdb_folder = 'filter_pdb'

# 创建目标文件夹，如果它们不存在
os.makedirs(filter_fasta_folder, exist_ok=True)
os.makedirs(filter_pdb_folder, exist_ok=True)

# 获取序列文件夹中的所有文件路径
sequence_files = glob.glob(os.path.join(sequence_folder, '*.fasta'))

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
        if len(sequences) <= 1 or len(":".join(sequences)) > 600:
            # print(f"Skipping {sequence_filename} - Invalid or empty FASTA content or sequence too long.")
            continue

        # 复制fasta文件到filter_fasta文件夹
        shutil.copy(sequence_file, os.path.join(filter_fasta_folder, sequence_filename))

        # 复制pdb文件到filter_pdb文件夹
        shutil.copy(pdb_file, os.path.join(filter_pdb_folder, sequence_filename.replace('.fasta', '.pdb')))
