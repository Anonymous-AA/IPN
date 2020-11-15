# Isometric Propagation Network for Generalized Zero-shot Learning

## Abstract
Zero-shot learning (ZSL) aims to classify images of an unseen class only based on a few attributes describing that class but no access to any training sample. A popular strategy is to learn a mapping between the semantic space of class attributes and the visual space of images based on the seen classes and their data. Thus, an unseen class image can be ideally mapped to its corresponding class attributes. The key challenge is how to align the representations in the two spaces. 
For most ZSL settings, the attributes for each seen/unseen class are only represented by a vector while the seen-class data provide much more information. Thus, the imbalanced supervision from the semantic and the visual space can make the learned mapping easily overfitting to the seen classes. To resolve this problem, we propose Isometric Propagation Network (IPN), which learns to strengthen the relation between classes within each space and align the class dependency in the two spaces. Specifically, IPN learns to propagate the class representations on an auto-generated graph within each space. In contrast to only aligning the resulted static representation, we regularize the two *dynamic* propagation *procedures* to be isometric in terms of the two graphs' edge weights per step by minimizing a consistency loss between them. IPN achieves the state-of-the-art performance on three popular ZSL benchmarks. To evaluate the generalization capability of IPN, we further build two larger benchmarks with more diverse unseen classes, and demonstrate the advantages of IPN on them.


## How to run?

**Prepare data**: follow `exps/prepare-data-step1.py` and `exps/prepare-data-step2.py` to create the cache dataset files.

**Reproduce results on AWA2**: `bash scripts/awa2.sh GPU-ID`.


## Selected parameters
 - gpus             : which gpu(s) to use (GPU-ID)
 - dataset          : the dataset train and evaluate on
 - class_per_it     : the number of classes for episodic training
 - num_shot         : the number of shots for episodic training
 - consistency_coef : the coefficient for consistency loss
 - lr               : the initial learning rate
 - weight_decay     : the weight decay
