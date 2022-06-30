import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('penguins_size.csv')

df.isnull().sum()

df.drop(df[df['body_mass_g'].isnull()].index, axis=0, inplace=True)
df['sex'] = df['sex'].fillna('MALE')
df.drop(df[df['sex'] == '.'].index, inplace=True)

df_new = df.drop(["species", "island", "flipper_length_mm", "body_mass_g", "sex"], axis=1)
array_df = np.array(df_new)

scale = MinMaxScaler()
scaled_array = scale.fit_transform(array_df)

dbscan = DBSCAN(eps=0.05, min_samples=3)
dbscan.fit(scaled_array)

labels = dbscan.labels_
len_label = len(labels)
clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Terdapat " + str(clusters) + " Cluster yang terbentuk")
for i in range(0, 20):
    print("Jenis Spesies: " + str(df.values[i, 0]) + ", Asal Wilayal: " + str(df.values[i, 1]) + ", Gender: " + str(df.values[i, 6]) +
          ", Cluster: " + str(dbscan.labels_[i]))
    print("---------------------------------------------------------------")

df["cluster"] = dbscan.labels_
output = plt.scatter(scaled_array[:, 0], scaled_array[:, 1], s=100, c=df.cluster, marker="o", alpha=1, )
plt.title("Hasil Clustering dengan DBSCAN")
plt.colorbar(output)
plt.show()
