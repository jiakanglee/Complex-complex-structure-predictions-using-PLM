import os
import sys
import torch
import esm
import biotite.structure.io as bsio

model = esm.pretrained.esmfold_v1().eval().cuda()
torch.cuda.set_per_process_memory_fraction(0.9)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "0:8192"
# torch.backends.cuda.caching_allocator = True
def process_fasta_file(input_file, output_dir):
    global model
    output_file = os.path.join(output_dir, os.path.basename(input_file).replace(".fasta", ".pdb"))

    # Skip conversion if the output .pdb file already exists
    if os.path.exists(output_file):
        print(f"Skipping {input_file} - {output_file} already exists.")
        return

    # 读取FASTA文件
    sequences = []
    with open(input_file, "r") as fasta:
        for seq in fasta:
            seq = seq.strip()
            if seq and not seq.startswith(">"):
                sequences.append(seq)

    # 如果sequences为空或只有一行，则直接跳过处理该文件
    if len(sequences) <= 1 or len(":".join(sequences)) > 800 or len(sequences) > 2:
        print(f"Skipping {input_file} - Invalid or empty FASTA content or sequence too long.")
        return

    # Join the sequences using ':' as a separator
    sequence = ":".join(sequences)
    with torch.no_grad():
        output = model.infer_pdb(sequence)
    

    with open(output_file, "w") as f:
        f.write(output)

    struct = bsio.load_structure(output_file, extra_fields=["b_factor"])
    print(f"Processed {input_file}. pLDDT: {struct.b_factor.mean()}")
    torch.cuda.empty_cache()
    

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        sys.exit(1)

    fasta_files = [f for f in os.listdir(input_dir) if f.endswith(".fasta")]

    if not fasta_files:
        print(f"Error: No FASTA files found in '{input_dir}'.")
        sys.exit(1)

    total_files = len(fasta_files)
    processed_files = 0

    # 在主函数之外加载模型

    # 每批次处理的文件数量
    batch_size = 5
    num_batches = (total_files + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_files)
        batch_files = fasta_files[start_idx:end_idx]

        for fasta_file in batch_files:
            input_file_path = os.path.join(input_dir, fasta_file)
            process_fasta_file(input_file_path, output_dir)

            processed_files += 1
            print(f"Processed {processed_files}/{total_files} files.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python esmfold_batch.py input_folder output_folder")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    main(input_folder, output_folder)

