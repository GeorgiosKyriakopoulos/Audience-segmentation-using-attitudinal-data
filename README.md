## Audience-segmentation-using-attitudinal-data

The analysis uses 37 Likert-scale variables to segment c.2,000 samples. 
The variables are pre-processed using Factor Analysis. Factor Analysis parameterisation is proposed following a grid-search and cross-fold validation process, which assess the average log-likelihood of samples.
Two clustering algorithms (i.e. k-means; affinity propagation) are tested upon the derived factors. Using a grid search, the tests assess possible solutions with regard to clusters' average silhouette scores.
The solution selected as optimal is then indicatively visualised upon a 3D space, where dimensions represent the top-3 Principal Components of the factors based on which the clusters are constructed.
