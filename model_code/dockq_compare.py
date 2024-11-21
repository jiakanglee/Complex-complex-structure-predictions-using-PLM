import os
import subprocess

def run_dockq_comparison(reference_folder, target_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(target_folder):
        if filename.endswith(".pdb.renum"):
            pdb_renum_file = os.path.join(target_folder, filename)
            pdb_file = os.path.join(reference_folder, filename.replace(".pdb.renum", ".pdb"))
            output_file = os.path.join(output_folder, filename.replace(".pdb.renum", ".txt"))

            command = f"DockQ/DockQ.py {pdb_renum_file} {pdb_file}"
            try:
                output = subprocess.check_output(command, shell=True, text=True)
                with open(output_file, "w") as f:
                    f.write(output)
                print(f"Comparison for {filename} saved to {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error while executing the DockQ command for {filename}: {e}")

if __name__ == "__main__":
    reference_folder = "selected_pdbs"
    target_folder = "selected_renum"
    output_folder = "selected_dockq"

    run_dockq_comparison(reference_folder, target_folder, output_folder)


