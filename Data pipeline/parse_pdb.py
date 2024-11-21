#parse_pdb
"""
aa_codes = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E',
    'PHE':'F','GLY':'G','HIS':'H','LYS':'K',
    'ILE':'I','LEU':'L','MET':'M','ASN':'N',
    'PRO':'P','GLN':'Q','ARG':'R','SER':'S',
    'THR':'T','VAL':'V','TYR':'Y','TRP':'W'
}
seq = ''
for line in open("5td6.pdb"):
    if line[0:6] =="SEQRES":
        columns = line.split()
        for resname in columns[4:]:
            seq = seq + aa_codes[resname]
i = 0
print(">5td6")
while i < len(seq) :
    print(seq[i:i+64])
    i =i+64
   """
aa3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
          'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
          'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
          'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
import os
import Bio
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
def pdb_to_fasta(pdb_file, fasta_file):

    sequences = []
    with open(pdb_file, 'r') as pdb:
        for record in SeqIO.parse(pdb, 'pdb-seqres'):
            sequences.append(record.seq)

    with open(fasta_file, 'w') as fasta:
        for i, seq in enumerate(sequences):
            fasta.write(f'>sequence{i + 1}\n')
            fasta.write(str(seq) + '\n')

#newest update:
def pdb_fasta(input_folder,output_folder):
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        structure_name = filename.split(".")[0]
        output_filename = filename.split(".")[0] + ".fasta"
        if filename.split(".")[-1] == "gz" or filename.split(".")[-1] == "fasta":
            continue
        # Parse the mmCIF file
        parser = PDBParser()
        file_path = file_path.encode('raw-unicode-escape').decode('utf-8','ignore')
        structure = parser.get_structure(structure_name, file_path)
        sequences = {}
        for model in structure:
            for chain in model:
                sequence = ""
                for residue in chain:
                    resname = residue.get_resname()
                    if resname in aa3to1:
                        sequence += seq1(residue.get_resname())
                    else:
                        continue
                if bool(sequence):
                    sequences[chain.get_id()] = sequence


        with open(output_folder+ '/' +output_filename, "w", newline=os.linesep) as f:
            for chain_id, sequence in sequences.items():
                f.write(">{}|{}\n".format(chain_id, structure_name))
                f.write(sequence.replace('X', ''))
                f.write("\n")

# Usage for single path
input_folder = "/lijiakang/dataset/Test_Dataset/Benchmark1/raw_pdb_unzip"
output_folder = "/lijiakang/dataset/Test_Dataset/Benchmark1/raw_fasta"
pdb_fasta(input_folder,output_folder)

#usage for mutiple files under multiple paths
#input = "/lijiakang/dataset/Zhang_dataset_tarred/protein_complex_decompressed"
#output_folder = "/lijiakang/dataset/Zhang_dataset/dimer"
#for input_file in os.listdir(input):
#    input_file_path = os.path.join(input, input_file)
#    for input_filee in os.listdir(input_file_path):
#        if len(input_filee) != 4:
#            continue
#        input_folder = os.path.join(input_file_path, input_filee)
#        #for filename in os.listdir(input_folder):
        #    if filename.split(".")[-1] != "pdb":
        #        inputfolder = os.path.join(input_folder, filename)
        #        pdb_fasta(inputfolder, output_folder)
#        pdb_fasta(input_folder,output_folder)
