# Predictive-Maintenance-for-Offshore-Oil-Wells-by-Means-of-Deep-Learnin
In this work, we propose a Deep Learning approach for the feature extraction in the offshore oil wells monitoring context, exploiting the public 3W dataset, well-known in literature. The dataset is available at the following link: https://archive.ics.uci.edu/ml/datasets/3W+dataset

We face up a classification task with eight classes, each one related to a particular condition of the machinery. Thanks to the peculiarities of the labels, the proposed framework is valid both for diagnostics and prognostics.

We compare two different approaches in features extraction. The first one is a statistical approach, widely used in the literature related to the considered dataset; the second is based on Convolutional 1D AutoEncoder.

The extracted features are then used as input for several Machine Learning algorithms, namely the Random Forest, Nearest Neighbors, Gaussian Naive Bayes and Quadratic Discriminant Analysis.

The worthiness of the Convolutional AutoEncoder is proved by different experiments on various time horizons.
