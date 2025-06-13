Summary:
This is a Top-K Similarity Graph Framework (TKSGF) for IoT network intrusion detection. Instead of
relying on physical links, the TKSGF constructs graphs based on Top-K attribute similarity,
ensuring a more meaningful representation of node relationships. We employ GraphSAGE as the 
Graph Neural Network (GNN) model to effectively capture node representations while maintaining scalability. 
Extensive experiments were conducted to analyze the impact of graph directionality (directed vs. undirected), 
different K values, and various GNN architectures and configurations on detection performance. 

Hardware requirements:
- A single NVIDIA GeForce RTX 3070 GPU (or equivalent) with at least 8GB of memory (Recommended).

Software requirements:
- PyCharm: any version should do
- torch version: 2.1.0+cu118 (Recommended)
- matplotlib version: 3.10.1 (Recommended)
- seaborn version: 0.13.2 (Recommended)
- NumPy version: 1.26.4 (Recommended)
- Pandas version: 2.2.3 (Recommended)
- FAISS version: 1.10.0 (Recommended)
- scikit-learn version: 1.6.1 (Recommended)

Instruction:
- Prepare the datasets. They can be downloaded from https://staff.itee.uq.edu.au/marius/NIDS_datasets/
- Update the working path in TKSGF_p1.py
- Run TKSGF_p1.py for graph construction
- Update the working path in TKSGF_p2.py
- Run TKSGF_p2.py for generating classification report
