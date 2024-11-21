from Bio import PDB
import os

def extract_sequence_from_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    for model in structure:
        for chain in model:
            sequence = ""
            for residue in chain:
                if PDB.is_aa(residue):
                    sequence += PDB.Polypeptide.three_to_one(residue.get_resname())
            return sequence

def pdb_folder_to_fasta_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdb"):
            pdb_file = os.path.join(input_folder, filename)
            fasta_file = os.path.join(output_folder, filename.replace(".pdb", ".fasta"))
            protein_sequence = extract_sequence_from_pdb(pdb_file)

            with open(fasta_file, "w") as f:
                f.write(f">{filename}\n{protein_sequence}")

if __name__ == "__main__":
    input_folder = "selected_pdbs"
    output_folder = "selected_fastas_from"

    pdb_folder_to_fasta_folder(input_folder, output_folder)
