import os

def merge_fasta_files(directory, output_file):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(directory):
            if filename.endswith('.fasta'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as infile:
                    outfile.write(infile.read())
                    outfile.write('\n')  # 添加换行符分隔各个fasta序列

# 示例用法
directory = "/lijiakang/dataset/Train_dataset/protein_complexes_frompdb/after_0.8_filter"  # 替换为实际的目录路径
output_file = '/lijiakang/dataset/Train_dataset/protein_complexes_frompdb/DB.txt'  # 替换为实际的输出文件路径
merge_fasta_files(directory, output_file)