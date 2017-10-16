# Import packages

import numpy as np
import pandas as pd
import matplotlib

# Read analysis file

df_1 = pd.read_csv(".../Dataframe_segmentation_v1.csv", index_col="ID")
df_1.shape #(2113, 37)
df_1.dtypes

# Turn data to numeric
df_1.head(0)
columns = ["Q006_01", "Q006_02", "Q006_03", "Q006_04", "Q006_05", "Q006_06", 
            "Q006_07", "Q006_08", "Q006_09", "Q006_10", "Q006_11", "Q007_01", 
            "Q007_02", "Q007_03", "Q007_04", "Q007_05", "Q007_06", "Q007_07", 
            "Q008_01", "Q008_02", "Q008_03", "Q008_04", "Q008_05", "Q008_06", 
            "Q008_07", "Q009", "Q010", "Q015", "Q016_01", "Q016_02", "Q016_03", 
            "Q016_04", "Q016_05", "Q016_06", "Q016_07", "Q016_08", "Q016_09"]

def coerce_df_columns_to_numeric(df, column_list):
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')

coerce_df_columns_to_numeric(df_1, columns)

nan_dim= df_1.shape[0] - df_1.count()
nan_dim.describe()
del columns, nan_dim

# Impute column mean for NaNs
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
df_imp = pd.DataFrame(imp.fit_transform(df_1))
df_imp.columns = df_1.columns
df_imp.index = df_1.index
del df_1

# Standardise
from sklearn import preprocessing
df_scaled = preprocessing.StandardScaler().fit_transform(df_imp)
df_scaled = pd.DataFrame(df_scaled, columns = df_imp.columns)
df_scaled.shape #(2113, 37)
df_scaled.index = df_imp.index
df_scaled.describe()
del df_imp

# Factor analysis
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score

n_components = np.arange(1, 37, 1) 

def select_fa(X):
    fa = FactorAnalysis() # For FA, score is the average log-likelihood of the samples
    fa_scores = []
    for n in n_components:
        fa.n_components = n
        fa_scores.append(np.mean(cross_val_score(fa, X, cv=10)))
    return fa_scores

fa_scores = select_fa(df_scaled)
n_components_fa = n_components[np.argmax(fa_scores)]
print("best n_components by Factor analyis = %d" % n_components_fa)
del fa_scores, n_components 

fa = FactorAnalysis(n_components = n_components_fa, iterated_power=10)
fa
fa.fit(df_scaled)
df_fa = pd.DataFrame(fa.transform(df_scaled), columns=["FA1", "FA2", "FA3", "FA4", "FA5", 
              "FA6", "FA7", "FA8", "FA9", "FA10", "FA11",
              "FA12", "FA13", "FA14"])
df_fa.index = df_scaled.index
del fa_scores, n_components, n_components_fa
df_fa.to_csv("Z:/1.CLIENTS/Wellcome Trust/QUANT/260139752 Longform evaluation/9. Data Processing/3. SPSS/SPSS for segmentation/Segmentation workings_GK/Longform evaluation_FA_v1.csv")


import matplotlib.pyplot as plt
plt.imshow(df_fa.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(df_fa.columns))]
plt.xticks(tick_marks, df_fa.columns, rotation='vertical')
plt.yticks(tick_marks, df_fa.columns)
plt.show()

# KMeans

from sklearn.cluster import KMeans
from sklearn import metrics

for n in np.arange(2, 11, 1):
    km = KMeans(n_clusters=n, random_state=50)
    cluster_labels_km = km.fit_predict(df_fa)
    silhouette_avg_km = metrics.silhouette_score(df_fa, cluster_labels_km)
    print("For n_clusters =", n,"The average K-Means silhouette_score is :", silhouette_avg_km)
del n, cluster_labels_km, silhouette_avg_km

from sklearn.cluster import AffinityPropagation
for n in np.arange(-1000, 0, 100):
    af = AffinityPropagation(preference=n).fit(df_fa)
    centers_ap= af.cluster_centers_indices_
    labels_ap = af.labels_
    n_clusters_ap = len(centers_ap)
    silhouette_avg_ap = metrics.silhouette_score(df_fa, labels_ap)
    print("For n_clusters =", n_clusters_ap,"(preference=", n, "), The average Aff-Prop silhouette_score is :", silhouette_avg_ap)
del n, centers_ap, labels_ap, n_clusters_ap, silhouette_avg_ap

# >>> Based on the trials KMeans with 6 clusters looks good

km = KMeans(n_clusters=6, random_state=50)
cluster_labels_6 = pd.DataFrame(km.fit_predict(df_fa), columns=["Label_6"])
cluster_labels_6.index = df_fa.index

import matplotlib.pyplot as plt
cluster_labels_6[0].hist(bins=6, alpha=0.7)
plt.show()


# Custer visualisation on 
from sklearn.decomposition import PCA
pca = PCA(n_components =3, svd_solver = "full") 
pca
pca.fit(df_fa)
pca_var_ratio = pca.explained_variance_ratio_.cumsum()
df_pca = pd.DataFrame(pca.transform(df_fa), columns=["PC1", "PC2", "PC3"])
df_pca.index = df_fa.index

vis = pd.concat([df_pca, cluster_labels_6], axis=1)
vis.to_csv("Z:/1.CLIENTS/Wellcome Trust/QUANT/260139752 Longform evaluation/9. Data Processing/3. SPSS/SPSS for segmentation/Segmentation workings_GK/Longform evaluation_Clusters_v1.csv")


colors = []
for i in vis.Label_6:
    if i==0:
        colors.append("red")
    elif i==1:
        colors.append("blue")
    elif i==2:
        colors.append("yellow")
    elif i==3:
        colors.append("green")
    elif i==4:
        colors.append("magenta")
    else:
        colors.append("cyan")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.style.use("ggplot") 
     
cluster_fig = plt.figure()
plt.suptitle("3D PC scatter plot showing clusters")
ax = cluster_fig.add_subplot(111, projection="3d")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.scatter(vis.PC1, vis.PC2, vis.PC3, c=colors, marker="x")
plt.show()


