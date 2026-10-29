# ConvNeXt ADNI Report - s4803018 (David Pelevin)
## Introduction

This repository focus on the implementation of a ConvNeXt model for identifying Alzheimer's Disease in patients. An ADNI dataset consisting of 2D slices of brain scans with existing train/test splits was provided to train the model. Ideally, the model will quickly and accurately identify the disease, however previous usage of AI models in medical fields revealed drastic overconfidence and overfitting. As a result, the model should be able to identify more uncertain cases that need human review. This facilitates sensible human oversight and accounting for possible errors in the model while still reducing the workload for doctors. ConvNeXt was chosen for its modern image processing design and its use of logits, which can be used to determine the model's confidence.

## Algorithm Breakdown



## Feasibility Review

### User Need, Scope & Acceptance Criteria

This machine learning model is intended to be used to automatically identify Alzheimer’s disease within a patient’s brain scans. Patients with uncertain results should be flagged for human review, allowing doctors and specialists to focus on more ambiguous cases.

The open research question involves testing advanced models against basic models to determine if the improved accuracy is worth the increased resource requirements. Beyond the required 80% test accuracy required by the task, the following criteria were chosen:

-	10% or less false negative Alzheimer’s identification rate. False negatives lead to untreated diseases, and it is preferable to have false positives over false negatives.
-	Identifiably lower confidence level on at least 80% of incorrect outputs compared to correct ones to allow for effective application of expert reviews. 
-	The model should run quickly to process large datasets, so an inference run-time of around 30ms per 256x240px image on a single A100 or V100 GPU will be required. This is based off prior work for 224x224 pixel images using the same GPUs and model type [1].

### Model Choice & Course Concepts

The chosen model for this task is a ConvNeXt model at the Hard Difficulty band. This model combines the computational simplicity of convolutional neural networks (CNN) with the higher accuracy given by Vision Transformer’s self-attention [1].

ConvNeXt models are based on normal CNNs, providing a good point of comparison. The output logits can be used for confidence levels. The ADNI dataset can also be used as given, since ConvNeXt models were originally designed using 2D images of similar sizes. These factors combine to make this a suitable model for the research question.

### Preliminary Feasibility Evidence

Initial testing of the ADNI dataset confirmed that none of the patients were repeated between the training and testing splits or the Alzheimer’s and Normal categories. Responsible usage of this dataset should prevent the overconfidence of some medical machine learning models.

The dataset consists of 256x240px grayscale images that are not limited to specific shades of grey. These images are larger than those used in the initial ConvNeXt design [1], and since the shades are not limited the images should have sufficient detailing for the model to train on.

Broad resource estimates for a ConvNeXt model based on prior work were given earlier [1].

### Risks, Budget & Fallback

The substantial amount of time and GPU resources necessary can be managed using the Rangpur computing cluster’s V100 and A100 GPUs. Next steps involve testing multiple convolution/layer sizes of the ConvNeXt design on the ADNI dataset. If time or space resources are insufficient, a single ConvNeXt model of appropriate accuracy can be compared to the baseline CNN model instead.

### References

[1] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A ConvNet for the 2020s,”
      in Proc. CVPR, 2022, pp. 11976–11986

## Dependencies and Reproducibility

The following libraries must be installed for this code to run:
-   matplotlib
-   torch
-   torchvision
-   pillow
-   sklearn

The results of this report should be reproducible due to the usage of specific seeds in train.py and the sorting of Path.glob outputs.

## Preprocessing and Hyperparameter Decisions

Preprocessing occurs with the following steps:

-   Normalisation
-   Random Horizontal Flips
-   Random crops of the files

** Must finish writing **

## Open Research Dilemna
### Data Hygiene

As alluded to in the Feasibility Review, running an initial data audit on the provided ADNI dataset yields the following:

-   520 unique patients in AD Train
-   556 unique patients in NC Train
-   223 unique patients in AD Test
-   227 unique patients in NC Test
-   0 duplicate patients among the train/test and AD/NC splits

This audit is based on the assumption given by tutors that the filenames follow the format: "patient_id"_"slice_id".jpeg.

To ensure no data leakage occurs, the training validation split was performed on the patient level instead of the slice level. This patient level split is performed in dataset.train_val_split, the outputs of which are used in the dataset.ADNIDataset. Patient IDs of each slice are checked when constructing the dataset and compared with the given split to ensure no data leakage occurs.

### Quantitative Benchmarking Against CNN



### Resource Profiling Table



### Failure Case Autopsy and Low Confidence Suggestions



## Artificial Intelligence Usage Disclosure

Claude Sonnet 5.5 Medium was used to help provide information and advice during the coding process.

Specifically:
-   Identifying how to calculate channel sizes during ConvNeXt stages
-   Debugging a Data Leakage Error that occurred in the Validation set
-   Advice for reducing overfitting (which was mostly ignored due to clashes with the task description)
-   Helped write code in previous demos that was adapted here

AI-suggested code was always checked against PyTorch documentation and ConvNeXt reports/explanations for validity and understanding. AI code was always rewritten with said understanding. The algorithms were also tested before and after changes to verify the desired functionality of said AI code. Code from previous demos were verified at the time, with appropriate adaptions made based on knowledge gained from the time.
