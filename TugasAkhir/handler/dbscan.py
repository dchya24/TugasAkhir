import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def init():
    df = pd.read_csv(os.path.dirname(__file__) + "/penguins_size.csv")

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

    return {
        "dbscan": dbscan,
        "driver": df,
        "scaled_array": scaled_array
    }
    # print("Terdapat " + str(n_clusters_) + " cluster yang terbentuk")


def generatePicture():
    data = init()
    driver = data["driver"]
    dbscan = data["dbscan"]
    scaled_array = data["scaled_array"]

    driver["kluster"] = dbscan.labels_

    labels = dbscan.labels_
    n_raw = len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    output = plt.scatter(scaled_array[:, 0], scaled_array[:, 1], s=100, c=driver.kluster, marker="o", alpha=1)

    penguins = []
    no = 1
    for i in range(0, n_raw):
        penguins.append({
            "no": no,
            "spesies": str(driver.values[i, 0]),
            "asal": str(driver.values[i, 1]),
            "berat": str(driver.values[i, 5]),
            "gender": str(driver.values[i, 6]),
            "cluster": dbscan.labels_[i]
        })
        no += 1

    plt.title("Hasil DBSCAN")
    plt.colorbar(output)
    plt.savefig(os.path.join(BASE_PATH, "static", "data.png"))
    plt.close()

    return {
        "n_clusters_": n_clusters,
        "penguins": penguins
    }

def searchClusterData(cluster):
    data = init()
    driver = data["driver"]
    dbscan = data["dbscan"]
    scaled_array = data["scaled_array"]

    labels = dbscan.labels_
    n_raw = len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    penguins = []
    no = 1
    for i in range(0, n_raw):
        if dbscan.labels_[i] == cluster:
            penguins.append({
                "no": no,
                "spesies": str(driver.values[i, 0]),
                "asal": str(driver.values[i, 1]),
                "berat": str(driver.values[i, 5]),
                "gender": str(driver.values[i, 6])
            })
            no += 1

    return penguins