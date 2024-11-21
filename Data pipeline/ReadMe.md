# Train dataset

## PDB
                                For PDB we have followed the procedures listed below:

                                1.search tag  = protein complex

                                2.resolution refinement ≤9 A

                                3.release date ≤2022.8.1

                                4.single amino acid occupancy frequency ≤0.8 

                                5.MMSEQ2 40% sequence identity cluster and sample  easy-cluster结果是，165492个pdb assemblies， sequence的数量471134个，number of clusters 3960

                                6.Sample using normalized probability accordingly using the MMSEQ2 clustered data.
                                
## PDB-distillation dataset
                                For PDB-distillation dataset we have followed the procedures listed below:

                                1.use the PDB dataset mentioned above, collect its sequences.

                                2.Pick Alphafold2-gap as distillation method

                                3.choose predicted sturctures with mean plddt>70

# Test dataset

## Recent PDB
                                For PDB we have followed the procedures listed below:
                                
                                1. search tag = protein compelx

                                2. resolution refinement ≤9 A

                                3. release date > 2022.8.1

                                4. single amino acid occupancy frequency ≤ 0.8 (9.15done)

                                5. filter out 超过40% 的data， 然后选每一个cluster里面的一个sample来预测 (思路很简单，先cluster，然后根据cluster的representative去跟train data的比较，<40%就拿来test)
## Other dataset                
                                For other test dataset, we first collected them, and we filtered out similarity larger than 40% compared with train dataset.

# Code description (specified for dealling with PDB files)

## Shell.sh
                                It's a shell command that downloads all the PDB files given the pdb entries ID lists(ie. 4UIX)

## run_pdb_to_fasta.sh             
                                It's a shell command that runs parse_pdb.py file, it has equipped and activated the conda environment that is needed.

## unzip.py
                                It's a py file that unzips the pdb files downloaded in .gz format to targeted directory. It's worth noting that if there exists son directory from the input directory, you could use the multiple_path version in the code.

                                 
                                If you have finished the steps mentioned above please use this command in your local terminal to transfer the files to the server. 

                                scp -r -P 30426 * root@172.16.78.10:\lijiakang\dataset\Train_dataset\protein_complexes_frompdb\ini_trainlist

## parse_mmcif/parse_pdb.py 
                                These two are meant to transform .pdb/.cif file into .fasta file. However, if the amino acids are not within the 20 human amino acids, they will be neglected automatically.

## frequence_filter.py
                                It's a py file which filters out .fasta file if single amino acid accounts for more than 80% of the all the chains.

## merge_fasta_files.py         
                                It's a py file which merges all .fasta files in one directory into one .txt file in order to satisfy the input requirements for MMSEQ2.

## mmseqs_cluster.sh
                                It's a shell command that run MMSEQ2.

## Sample_mmseq.py
                                In order to sample according to normalized probability from the cluster result, this code is then created for fulfilling that.

## extract_one_column.py
                                It's used to scratch selected one column along with its full items in a txt file, and then output it to another txt file seperated by "," (ie. you want to scrath pdb entry ID from a pdf file, you will get to use this code.)
                          
