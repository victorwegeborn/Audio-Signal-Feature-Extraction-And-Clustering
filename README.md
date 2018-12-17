Audio Signal Feature Extraction And Clustering
=========

A configurable feature extractor for .wav audio files storing data in buckets on disc for easier
loading of subsets of classes and features during clustering.

Installation
------------
Use python 3.6. Dependencies are listed in requirements.txt


```
pip install -r requirements.txt
```

Extracting features
=========
Features are extracted and stored in buckets on disc according to the subfolders in the audio folder.
Each subfolder is regarded as a class folder which is used to label the data for evaluation.
Any folder within a class folder is merged into the same class.


```
python DataHandler.py -d <audio folder> -f all
```
Keeping features in separate buckets allows for faster testing and evaluation of newly implemented features as well as different permutations of feature sets. The same is true for class selection during clustering.

Currently, these features are available:
- mfcc
- decay
- rolloff
- brightness

After an initail feature extraction it is possible to use the -f flag for extraction of specific features.
This enables faster testing of newly implemented features.

Clustering of data
============
Configure which classes and what features to use during clustering by modifying configuration/congif.py.
By default, all avaliable features are used.

```
python Cluster.py -c config
```

Evaluation metrics
==============
The following metrics are used to evaluate the clusters:
- homogeneity score (homo) - clusters contain only data points which are members of a single class. 1.0 perfect score.
- completeness score (compl) - all the data points that are members of a given class are elements of the same cluster. 1-0 perfect score.
- v-measure score (v-meas) - harmonic mean between homogeneity and completeness. 1.0 perfect score.

Clusters are then relabeled according to initial classes and the accuarcy and a confusion matrix is computed.

Results
=============

A data set consisting of 1175 drum samples were to be identified based using clustering. All drum samples had a frame rate of 44100 and a 24-bit depth.
Bellow is a summary of the distribution of the data per class:

| class | clap | cymbal | fx | hi-hat | kick | perc | rim | snare | tom | (total) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| number of samples | 15 | 63 | 11 | 141 | 277 | 94 | 252 | 174 | 148 | (1175) |
| percentage | 1.28% | 5.36% | 0.94% | 12.00% | 23.57% | 8.00% | 21.45% | 14.81% | 12.60% | (100%)


### Results on a subset of 5 classes

All available features was used in this Experiment.
The final result on a subset of the classes consisting of kick, snare, hi-hat, cymbal, and tom:

```
Algo		time	inertia	  homo	  compl	  v-meas
k-means  	1.23s	371.646	  0.814	  0.803	  0.808

Model accuracy: 0.904
```


<img align="center" width="420" height="315" src="https://github.com/victorwegeborn/Audio-Signal-Feature-Extraction-And-Clustering/blob/master/results/confusion_matrix.png"><img align="center" width="420" height="315" src="https://github.com/victorwegeborn/Audio-Signal-Feature-Extraction-And-Clustering/blob/master/results/confusion_matrix_norm.png">

The unnormalized confusion matrix shows that dispite the skeewed dataset, the clustering works rather well. This can mostly be attributed to the mfcc feature that produces clusters with 0.746 in v-measure score. To further increase the accuracy, some feature that can distinctly separate hi-hats from cymbals must be found as some of the hi-hats are currently clustered together with the cymbals.
