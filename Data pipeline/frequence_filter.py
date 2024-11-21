import os
from Bio import SeqIO

def filter_and_save_fastas(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    num_sequence = 0
    for filename in os.listdir(input_directory):
        flag = 0
        total_len = 0
        if filename.endswith(".fasta"):
            input_file = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, filename)

            sequences = list(SeqIO.parse(input_file, "fasta"))

            amino_acid_counts = {}
            total_sequences = len(sequences)

            # Count the occurrence of each amino acid in all sequences
            for sequence in sequences:
                total_len += len(sequence)
                for amino_acid in sequence.seq:
                    amino_acid_counts[amino_acid] = amino_acid_counts.get(amino_acid, 0) + 1

            # Calculate the occupancy frequency of each amino acid
            for amino_acid, count in amino_acid_counts.items():

                occupancy_frequency = count / total_len
                if occupancy_frequency > 0.8:
                    flag = 1
                    break

            if flag != 1:
                # Save the non-filtered fasta file to the output directory
                # count num of sequences
                for sequence in sequences:
                    num_sequence+=1
                SeqIO.write(sequences, output_file, "fasta")
                print(f"Saved {filename} to {output_directory}")
    #print(num_sequence)



# Usage example
input_directory = "/lijiakang/dataset/Test_Dataset/PDB_dataset/raw_fasta"
output_directory = "/lijiakang/dataset/Test_Dataset/PDB_dataset/fasta_filter"
filter_and_save_fastas(input_directory, output_directory)