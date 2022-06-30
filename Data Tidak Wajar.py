import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("penguins_size.csv")

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
n_raw = len(labels)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("terdapat: " + str(n_clusters) + " Cluster yang terbentuk")

a = int(input("Masukkan Nomor Kluster yang ingin anda lihat(0-" + str(n_clusters - 1) +
              ", tekan -1 jika ingin melihat data tak wajar): "))
b = 0
for i in range(0, n_raw):
    if dbscan.labels_[i] == a:
        print("Jeni Spesies: " + str(df.values[i, 0]) + ", Asal Wilayah: " + str(df.values[i, 1]) + ", Gender: " + str(
            df.values[i, 6]))
        print("---------------------------------------------------------------")
        b += 1

if(a >= 0):
    print("total banyak pinguin di cluster " + str(a) + " sebanyak: " + str(b))
else:
    print("total banyak pinguin dengan data tidak wajar sebanyak: " + str(b))