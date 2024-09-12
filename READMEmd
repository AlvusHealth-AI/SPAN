# SPIN: A 3D Sequence-Pixel Gallery for Analyzing the Disruption of Protein-Protein Interactions in Neuropsychiatric Disorders
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

## Abstract
Developmental disorders are often associated with mutations that disrupt protein interactions and functions. However, analyzing the likelihood and impact of missense mutations on these interactions is challenging due to the lack of annotations and the limited coverage provided by costly experiments. To enable effective AI evaluations, we present the SPIN dataset, comprising 729 protein-protein interaction sets identified from whole-exome sequencing of autism spectrum disorder families. Each set includes 3D structures, amino acid sequences, protein-pathway connection maps, and experimentally validated mutation disruptiveness labels. Our benchmark results show that LLM-based protein language models, integrating sequence and 3D information, outperformed state-of-the-art 3D CNN and GNN models by over 5% in both Area Under the Curve (AUC) and balanced accuracy. For the first time, predicting the disruption of protein interactions caused by missense mutations is presented as a general AI challenge. SPIN is the first dataset validated through wet lab experiments designed for propelling AI development to accelerate the discovery of novel therapeutic targets for complex neuropsychiatric disorders.

![Architecuture](figs/benchmarks.png)

## Dataset
Our dataset contains 729 proteins, sotred in FASTA format. For each protein, the protein structures were predicted from AlphaFold Multimer and stored in PDB format. In total, 376 binding/non-binding interactoins were recorded and stored in `data/protein_interactions.tsv`. Out of the 376 interactions, 85 of the interactions are labeled as binding, while 291 are labeled as non-binding. 

These protein sequences range from a minimum length of 216 to a maximum of 2,811, with an average length of 1,023.79 and a median of 956. The file sizes of the FASTA files range from 234 to 2,835 bytes, with an average size of 1,044.81 bytes and a median size of 977 bytes.

We used three preset AlphaFold Multimer models for protein structure generation, each with 5 different random seeds applied. Thus, 15 predictions for each protein were generated. For structure prediction, only templates dated before 2021-11-01 were used. 10,935 predictions are generated from AlphaFold Multimer for 729 proteins. The entire dataset occupies a total volume of 8.6 TB. The file sizes of PDB files range from 0.116 MB to 3.31 MB, with an average file size of 0.661 MB and a median of 0.595 MB. The file sizes of the MSAs generated from the MSAs construction step range from 0.001 GB to 28.705 GB with mean and median equal to 2.353 GB and 0.597 GB. The intermediate results are also stored in files, including the input features, distograms. The file sizes of the intermediate results range from 0.003 GB to 3.927 GB with an average file size of 2.353 GB and median file size of 0.597 GB. The inference time for predicting protein structures ranges from 0.125 hours to 18.438 hours. The mean and median inference time is 2.181 hours and 1.513 hours.

We stored the entire dataset on Amazon S3. To dowload the dataset, [Amazon CLI]( https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) is recommended. No credentials are required to download our dataset.

```sh
aws s3 sync s3://fastasalphafoldpredictions/SPIN /local/path/for/dataset --no-sign-request
```


## Installation
To run the benchmarks, please install the dependencies via the following command first.
```sh
pip install -r requirements.txt
```

## Benchmarking

We employed four models for benchmarking purpose: 3D CNN, [SpatialCNN](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10966450/), [PPI_GNN](https://www.nature.com/articles/s41598-022-12201-9), [ProtTrans](https://ieeexplore.ieee.org/document/9477085) and [ProLLM](https://arxiv.org/abs/2405.06649).

### 3D CNN
We designed a custom-designed 3D CNN was developed and evaluated on the SPIN dataset. We built a 3D CNN with two parallel layers, each layer accepting inputs from a protein. The outputs from the 3D ResNet blocks were fed to a max-pooling 3D layer and a fully connected layer, converting the 3D features into 1D feature vectors. The 1D vectors from two parallel layers were concatenated and downsampled for PPI predictions.

To train 3D CNN, run the following command:
```sh
python 3DCNN/train.py --datapath /path/to/dataset --batch 32 --epoch 200 --lr 1e-4 --savingPath /path/to/save/results
```

### SpatialCNN
SpatialPPI is a novel 3D CNN model designed to predict PPIs propsoed by Wenxing Hu and Masahito Ohue. SpatialPPI concatenated the protein tensors into a single tensor and fed into a 3D convolutional layer, 3D ResNet blocks with 4 blocks, and a fully connected layer for predictions of the binding/non-binding classes.

To train SpatialCNN, run the following command:
```sh
python SpatialCNN/train.py --datapath /path/to/dataset --model Resnet3D --batch 32 --epoch 200 --lr 1e-4 --savingPath /path/to/save/results
```

### PPI_GNN
Graph Neural Networks (GNNs) excel at capturing and processing structural information, making them particularly well-suited for PPI prediction. By leveraging their ability to model the complex relationships and interactions within protein structures, GNNs provide more accurate predictions of PPIs. PPI_GNN is a notable example that utilizes protein structural information with a graph neural network and sequence features to predict interactions between proteins.    

To train PPI_GNN with GCN backbone, run the following command:
```sh
python PPI_GNN/train.py --datapath /path/to/dataset --model GCN --batch 32 --epoch 200 --lr 1e-3 --dropout 0.2 --savingPath /path/to/save/results    
```

To train PPI_GNN with GAT backbone, run the following command:
```sh
python PPI_GNN/train.py --datapath /path/to/dataset --model GAT --batch 32 --epoch 200 --lr 1e-3 --dropout 0.2 --savingPath /path/to/save/results    
```

### ProtTrans   
Protein Language Models (pLMs) adapt concepts from Natural Language Processing (NLP) by using amino acids as tokens (analogous to words in NLP) and treating entire protein sequences as sentences. ProtTrans is a pLM frequently used as a feature extractor for protein sequences in various sequence-based tasks. we used the T5-based ProtTrans to extract features from the protein sequences. The T5-based ProtTrans is pretrained by the ProtTrans official and can be directly downloaded from Hugging Face. The initial embedding size from ProtTrans for each protein was 1024, which was then gradually downsampled to 512, 256, and 128 by passing the embedding through three fully connected layers. Meanwhile, we employed a predtrained 3D CNN to extract embeddings from 3D structures, with the embedding size for each protein also downsampled to 128. The two embeddings were concatenated and passed through a pair of fully connected layers for PPI.

To train ProtTrans, run the following command:
```sh
python ProtTrans/train.py --datapath /path/to/dataset --batch 32 --epoch 200 --lr 1e-3 --dropout 0.2 --savingPath /path/to/save/results    
```

### ProLLM
ProLLM, short for Protein Chain-of-Thoughts Enhanced LLM for Protein-Protein Interaction Prediction, is the first to use a large language model (LLM), specifically Flan-T5-large, for PPI prediction. ProLLM introduces the Protein Chain of Thought (ProCoT) strategy, which replicates the biological mechanism of signaling pathways as natural language prompts. Additionally, it replaces the natural language embeddings with protein embeddings in the prompts sent to Flan-T5-large by instruction fine-tuning it based on protein knowledge datasets. In our experiment, we generated the embeddings from ProtTrans and the pretrained 3D CNN and concatenated them into one vector for the proteins. The sizes of the embeddings from ProtTrans and 3D CNN are both 512 which makes the size of the concatenated embedding 1024. The concatenated embeddings of the proteins were added to the vocabulary of Flan-T5-large and used to replace the natural language embeddings of Flan-T5-large. Whenever the model encounters the proteins, it employs the more meaningful embeddings of the proteins from the vocabulary instead of the natural language embeddings of the proteins. We built the instruction dataset, more protein details were incorporated, including the functions and pathways information of the proteins. The instruction dataset is formatted in a <question, answer> fashion. The protein functions and pathways information can be found in `data/gene-protein-info.csv`. By fine-tuning Flan-T5-large on the instruction dataset, the protein-domain knowledge is infused into the LLM. In addition to the model fine tuning, We trained the finetuned Flan-T5-large with the protein embedding replacement to answer how the proteins interact with each other.

To fine tune on the instruction dataset
```sh
python ProLLM/finetune.py --datapath /path/to/instruction/dataset --batch 2 --epoch 10 --lr 3e-4 --weight_decay 1e-2 --savingPath /path/to/save/results    
```

To train ProLLM,
```sh
python ProLLM/train.py --datapath /path/to/dataset --weights /path/to/model --batch 2 --epoch 1 --lr 3e-4 --weight_decay 1e-2 --savingPath /path/to/save/results    
```