import os
import subprocess
import sys
import shutil

def convert_pdb_to_renum(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    pdb_files = [f for f in os.listdir(input_dir) if f.endswith(".pdb")]

    if not pdb_files:
        print(f"No PDB files found in '{input_dir}'.")
        return

    for pdb_file in pdb_files:
        pdb_file_path = os.path.join(input_dir, pdb_file)
        renum_file_path = os.path.join(output_dir, pdb_file.replace(".pdb", ".pdb.renum"))

        try:
            subprocess.run(["DockQ/scripts/renumber_pdb.pl", pdb_file_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error converting '{pdb_file}': {e}")
            continue
        shutil.move(pdb_file_path + ".renum", renum_file_path)
        print(f"Converted '{pdb_file}' to '{renum_file_path}'")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_folder.py input_folder output_folder")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    convert_pdb_to_renum(input_folder, output_folder)
