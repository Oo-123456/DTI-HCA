# DTI-HCA

## Requirements

* python 3.8.16
* cuda 11.3
* pytorch 1.12.1
* dgl 1.1.1
* numpy 1.24.3
* sklearn 1.2.2

## Quick start

1. python `main.py` Options are:
   1. `--num_channels: the number of channels, default: 4`
   2. `--num_layers: the maximum length of the learned meta-paths, default: 3`
   3. `--alpha: hyperparameter in loss function, default: 0.4`
   4. `-t: select experimental data, default: 'o'`

## Data description
`data` Folder. This folder contains the DTINet dataset.
* `drug.txt` : list of drug names.
* `protein.txt` : list of protein names.
* `disease.txt` : list of disease names.
* `se.txt` : list of side effect names.
* `drug_dict_map.txt` : a complete ID mapping between drug names and DrugBank ID.
* `protein_dict_map.txt`: a complete ID mapping between protein names and UniProt ID.
* `drug_se.npy` : Drug-SideEffect association matrix.
* `protein_protein.npy` : Protein-Protein interaction matrix.
* `drug_drug.npy` : Drug-Drug interaction matrix.
* `protein_disease.npy` : Protein-Disease association matrix.
* `drug_disease.npy` : Drug-Disease association matrix.
* `protein_drug.npy` : Protein-Drug interaction matrix.
* `drug_protein.npy` : Drug-Protein interaction matrix.
* `drug_protein.npy` : Drug-Protein interaction matrix.
* `drug_protein_indepent.npy` : the cross-validation set.
* `drug_structure.pt` : drug structure features.
* `protein_structure.pt` : protein sequence features.
* `mat_drug_protein_homo_protein_drug.txt` : Drug-Protein interaction matrix, in which DTIs with similar drugs (i.e., drug chemical structure similarities > 0.6) or similar proteins (i.e., protein sequence similarities > 40%) were removed (see the paper). 
* `mat_drug_protein_drug.txt` : Drug-Protein interaction matrix, in which DTIs with drugs sharing similar drug interactions (i.e., Jaccard similarities > 0.6) were removed (see the paper). 
* `mat_drug_protein_sideeffect.txt` : Drug-Protein interaction matrix, in which DTIs with drugs sharing similar side effects (i.e., Jaccard similarities > 0.6) were removed (see the paper). 
* `mat_drug_protein_disease.txt` : Drug-Protein interaction matrix, in which DTIs with drugs or proteins sharing similar diseases (i.e., Jaccard similarities > 0.6) were removed (see the paper). 
* `mat_drug_protein_unique.txt` : Drug-Protein interaction matrix, in which known unique and non-unique DTIs were labelled as 3 and 1, respectively, the corresponding unknown ones were labelled as 2 and 0 (see the paper for the definition of unique). 

`drug_protein_feature` Folder
Run `main.py` to learn drug structure features and protein sequence features.

These files: drug.txt, protein.txt, disease.txt, se.txt, drug_dict_map, protein_dict_map, mat_drug_se.txt, mat_protein_protein.txt, mat_drug_drug.txt, mat_protein_disease.txt, mat_drug_disease.txt, mat_protein_drug.txt, mat_drug_protein.txt, Similarity_Matrix_Proteins.txt, are extracted from https://github.com/luoyunan/DTINet.

These files: mat_drug_protein_homo_protein_drug.txt, mat_drug_protein_drug.txt, mat_drug_protein_sideeffect.txt, mat_drug_protein_disease.txt, mat_drug_protein_unique.txt are extracted from https://github.com/FangpingWan/NeoDTI
