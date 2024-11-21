def process_file(input_file_path):
    lines_to_write = []

    with open(input_file_path, 'r') as file:
        for line in file:
            # Check if the line has more than 40 characters
            if len(line) > 40:
                # Check if the 22nd character is 'C'
                if line[21] == 'C':
                    # Replace the 22nd character with 'B'
                    modified_line = line[:21] + 'B' + line[22:]
                    lines_to_write.append(modified_line)
                else:
                    lines_to_write.append(line)
            else:
                lines_to_write.append(line)

    # Write the modified content back to the file
    with open(input_file_path, 'w') as file:
        file.writelines(lines_to_write)

# Example usage:
file_path = 'Benchmark2-multimer_pdbs_renum/7m5f.pdb.renum'
process_file(file_path)
