# Protein-Protein-interaction

**Hi! Welcome to open an issue or make a pull request!**

# Motivation
   
- We want to construct a model that predicts the structure of protein complex, which omits the usage of traditional MSA and saves a lot of time.
- it borrows the idea of Foldseek's transforming the 3D interaction between amio acids into token tables.
- And we hope to achieve SOTA or excel SOTA in DockQ and TM-Score.

## Pipeline

There are 3 parts for this problem, which are Data, Pretrained Model, Evoformer+folding trunk

### Data
The data pipeline is crucially important and complicated simultaneously. It is divided into two parts, train set and test set.

#### Train dataset

- Futher training pre-trained model Data.  It takes in sequences and output the embeddings for the sequence, this is where we borrows the idea of the Foldseek's 3Di token tables. The data are mainly collected from Uniref50, PDB, Swiss-Prot, PPI dataset(comPPI, intACT, huMA). Also it needs to filter out sequences that are too alike or credibility score that is below 0.8.
- Protein Complex Prediction Data. it takes in embeddings from pretrained model and outputs 3D structure of the protein complex. One source comes from Protein Data Bank, and it follows the pipeline of the Alphafold2 / Alphafold-Multimer. One source comes from using the PDB data to feed in Alphafold-Multimer/ alphafold2-gap to generate self-distillation dataset and it filters out protein complex with mean plddt below 70. The last source comes from Guojun Zhang's dataset.

#### Test dataset

- CASP15 dataset. CASP (15th Community Wide Experiment on the Critical Assessment of Techniques for Protein Structure Prediction) is a worldly-known contest on predicting structure for protein. And it is held every 2 year.
- Recent PDB data. We have collected PDB data after the timestamp when the training data is last collected, and it used the same criteria as train dataset. And it is worth noting that it filters out data with similarity larger than 40% using mmseqs2.
- Benchmark2(CP17). it is a benchmark consisting of 17 protein complexes and is widely used as test dataset.
- Benchamrk1.  was based on Version 5 of the very established protein-protein docking benchmark (BM5) developed by the Weng group BM5 is a nonredundant and fairly diverse set of protein complexes for testing 
protein–protein docking algorithms. Each entry in BM5 includes the 3D structures of the complex and one or both unbound component proteins. The set includes 40 antibody-antigen, 88 enzyme-containing and 102 ‘‘other type’’ complexes.
- Case study. We intend to pick up several protein complexes that alphafold2-multimer did bad in and study its effectiveness for our model.
