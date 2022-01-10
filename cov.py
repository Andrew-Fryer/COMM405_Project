
# load data
import pandas as pd
df = pd.read_csv('2021q3_fin_sec_data.csv')

# normalize numeric values
df_num = df.select_dtypes(include='number')
df_norm = (df_num - df_num.mean()) / (df_num.max() - df_num.min())
df[df_norm.columns] = df_norm

# cluster data
from sklearn.cluster import KMeans
num_clusters = 20
kmeans = KMeans(n_clusters=num_clusters).fit(df_norm.values)
print('cluster centers:', kmeans.cluster_centers_)

