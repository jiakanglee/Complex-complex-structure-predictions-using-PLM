import os
import gzip

def gunzip_all_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Get a set of already uncompressed files
    already_uncompressed = set(os.listdir(output_folder))

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if filename.endswith(".gz"):
            uncompressed_file = os.path.join(output_folder, filename.replace(".gz", ""))
            if filename.replace(".gz", "") not in already_uncompressed:
                try:
                    with gzip.open(file_path, "rb") as gz_file:
                        with open(uncompressed_file, "wb") as output_file:
                            output_file.write(gz_file.read())
                    print(f"Uncompressed {filename} to {output_folder}")
                except Exception as e:
                    print(f"Error while uncompressing {filename}: {e}")
                    os.remove(uncompressed_file)  # Remove incomplete output file if exists
            else:
                print(f"Skipping {filename} - File already uncompressed")
        else:
            print(f"Skipping {filename} - Not a .gz file")
#single path usage
if __name__ == "__main__":
 input_file_path = "/lijiakang/dataset/Test_Dataset/Benchmark1/raw_pdb"
 output_folder = "/lijiakang/dataset/Test_Dataset/Benchmark1/raw_pdb_unzip"
 gunzip_all_files(input_file_path,output_folder)
#multipath_usage suitable only if there are further zipped files in son path

#input = "/lijiakang/dataset/Train_dataset/protein_complexes_frompdb/without_0.8_filter"
#output_folder = "/lijiakang/dataset/Train_dataset/protein_complexes_frompdb/without_0.8_unzip"
#for input_file in os.listdir(input):
#    input_file_path = os.path.join(input, input_file)
#    gunzip_all_files(input_file_path, output_folder)

