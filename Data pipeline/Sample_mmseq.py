from collections import defaultdict
import random

def create_cluster_dictionary(filename): #把cluster的结果做成词典
    clusters = {}

    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        data = line.strip().split('\t')
        chain_assembly = data[0]
        cluster = data[1]
        if chain_assembly not in clusters:
            clusters[chain_assembly] = []
        clusters[chain_assembly].append(cluster)

    return clusters

def count_chains_in_assembly(filename):    #把assembly本身做成词典
    assembly = defaultdict(list)
    with open(filename, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('>'):
            cluster_name, assembly_name = line[1:].split('|')
         #   chain_num = assembly.get(assembly_name, 0) + 1
         #   assembly[assembly_name] = chain_num
            assembly[assembly_name].append(cluster_name)
            i += 1  # Move to the next line
            while i < len(lines) and not lines[i].startswith('>'):
                i += 1  # Skip the chain lines
        else:
            i += 1

    return assembly

def calculate_single_chain_probability(chain_name, clusters):  #计算单条链的概率
    for centroid in clusters.keys():
        for chain in clusters[centroid]:
            if chain_name == chain:
                probability = 1 / len(clusters[centroid])
    return probability

def calculate_and_normalize_probability(assembly,clusters): #把assembly所有链合起来，计算成总概率，并得出归一化分布
    probability = {}
    i = 0
    for assembly_name in assembly.keys():
        i+=1
        print(i)
        probability[assembly_name] = 0
        for chain in assembly[assembly_name]:
            chain_name = chain+'|'+assembly_name
            probability[assembly_name]+= calculate_single_chain_probability(chain_name, clusters)
    probabilities = list(probability.values())
    total_sum = sum(probabilities)
    normalized = {key: value / total_sum for key, value in probability.items()}
    return normalized

def sample_from_distribution(distribution): #根据概率采样
    keys = list(distribution.keys())
    probabilities = list(distribution.values())
    sample = random.choices(keys, weights=probabilities, k=1)
    return sample[0]


# Call the function
filename = '/lijiakang/dataset/Train_dataset/protein_complexes_frompdb/mmseq_cluster_0.4/DB.fasta'  # Replace with your FASTA file path
assembly_list = count_chains_in_assembly(filename)
#assembly looks like this and is a dictionary
#Chain_Assembly: C|1OV3
#Clusters: ['C', 'D']

filename = '/lijiakang/dataset/Train_dataset/protein_complexes_frompdb/mmseq_cluster_0.4/clusterRes_cluster.tsv'  # Replace with your file path
clusters = create_cluster_dictionary(filename)

#clusters looks like this and is a dictionary
#Chain_Assembly: C|1OV3
#Clusters: ['C|1OV3', 'D|1OV3']

Probability_distribution = calculate_and_normalize_probability(assembly_list,clusters)

#采样进行训练譬如100个
samples = []
for _ in range(100):
    sample = sample_from_distribution(Probability_distribution)
    samples.append(sample)