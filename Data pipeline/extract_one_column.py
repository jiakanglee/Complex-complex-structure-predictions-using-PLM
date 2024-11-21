import os

# Input directory path
input_directory = '/lijiakang/dataset/Test_Dataset/Benchmark1'

# Get the list of files in the input directory
files = os.listdir(input_directory)

# Iterate over the files in the input directory
for file_name in files:
    # Check if the file is a text file
    if file_name.endswith('.txt'):
        # Create the input file path
        input_file_path = os.path.join(input_directory, file_name)

        # Open the input text file
        with open(input_file_path, 'r') as file:
            # Read the contents
            contents = file.readlines()

        # Extract and keep the first column for each row
        first_columns = [row.split()[0] for row in contents]

        # Create the output file path
        output_file_path = os.path.join(input_directory, 'output_' + file_name)

        # Open the output text file
        with open(output_file_path, 'w') as output_file:
            # Write the first columns separated by commas to the output file
            output_file.write(','.join(first_columns))

