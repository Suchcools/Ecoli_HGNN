COLI-HGNN
Chinese Academy of Sciences Escherichia coli Knowledge Graph Heterogeneous Graph Neural Network Project

## running environment

* torch 2.1.1 cuda 11.8
* dgl 1.1.2 cuda 11.8
* networkx 2.3
* scikit-learn 1.3.2
* scipy 1.11.3

Abstract
Transcription factors (TFs) are pivotal in gene regulation, influencing RNA polymerase during transcription initiation. Identification and comprehension of TFs offer insights into how organisms regulate gene expression in response to genetic or environmental changes. Traditional TF identification relies on experimental methods, but the surge in newly discovered proteins necessitates more efficient computational approaches. This study proposes the inaugural Escherichia coli TF prediction model based on heterogeneous graphs. Leveraging a deep learning framework, we integrate protein sequences with knowledge graphs, emphasizing protein relations. Graph neural networks (GNNs) excel in learning from graph-structured data, and our goal is to utilize their capabilities for E. coli TF prediction. The model embeds protein sequences, constructs heterogeneous graphs, and employs GNNs for comprehensive representation. This approach enhances prediction accuracy while considering the intricate interactions in E. coli. Model interpretability is explored through explanatory tool analysis. Our study bridges the gap between sequence-based predictions and abundant interaction information in knowledge graphs, offering a holistic understanding of transcriptional regulation in E. coli. The proposed model's feasibility is supported by expertise in protein sequence embedding and deep learning within our research group. This work contributes a novel, interpretable deep learning model for E. coli TF prediction, with potential applications across species and biological processes.
