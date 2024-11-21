#! /bin/bash
cd /lijiakang/local_project/

export PATH=/opt/anaconda3/bin:$PATH
source $(which activate) LTEnjoy

cd /lijiakang/dataset/Train_dataset/protein_complexes_frompdb/mmseq_cluster_0.4

mmseqs easy-cluster DB.fasta clusterRes tmp --min-seq-id 0.5 -c 0.4 --cov-mode 1
