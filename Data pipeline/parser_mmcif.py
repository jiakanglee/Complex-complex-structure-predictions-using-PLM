from Bio.SeqUtils import seq1
import os
import os
import json
import copy
import numpy as np
from tqdm import tqdm
from Bio import pairwise2
from Bio.PDB import PDBParser, FastMMCIFParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1

aa3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
          'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
          'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
          'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
mmcif_parser = FastMMCIFParser(QUIET=True)
mmcif_io = MMCIFIO()
pdb_parser = PDBParser(QUIET=True)



def mmcif_to_fasta(input_filename):
    # Define the input and output filenames
    output_filename = input_filename.split(".")[0] +".fasta"
    structure_name = input_filename.split(".")[0]
    # Parse the mmCIF file
    parser = MMCIFParser()
    structure = parser.get_structure(structure_name, input_filename)

    # Extract the sequence information from each chain in the structure
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
            sequences[chain.get_id()] = sequence

    # Write the sequences to a FASTA file
    with open(output_filename, "w", newline=os.linesep) as f:
        for chain_id, sequence in sequences.items():
            f.write(">{}|{}\n".format(chain_id,input_filename))
            f.write(sequence.replace('X',''))
            f.write("\n")

def parse_all_files(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        structure_name = filename.split(".")[0]
        output_filename = filename.split(".")[0] + ".fasta"
        # Parse the mmCIF file
        parser = MMCIFParser()
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
                sequences[chain.get_id()] = sequence


        with open(output_folder+ '/' +output_filename, "w", newline=os.linesep) as f:
            for chain_id, sequence in sequences.items():
                f.write(">{}|{}\n".format(chain_id, structure_name))
                f.write(sequence.replace('X', ''))
                f.write("\n")


input_folder = "/lijiakang/dataset/Benchmark2-multimer_cif"
output_folder = "/lijiakang/dataset/Benchmark2-multimer_fasta"
parse_all_files(input_folder,output_folder)