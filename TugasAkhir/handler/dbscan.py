import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


def init():
    driver = pd.read_csv(os.path.dirname(__file__) +"/go_track_tracks.csv")
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    driver_x = driver.drop(
        ['linha', 'rating', 'time', 'car_or_bus', 'rating_weather', 'rating_bus', 'id', 'id_android'], axis=1)

    x_array = np.array(driver_x)

    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x_array)

    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan.fit(x_scaled)

    labels = dbscan.labels_
    n_raw = len(labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print("Terdapat " + str(n_clusters_) + " cluster yang terbentuk")

    anomali_data = []
    for i in range(0, n_raw):
        if (dbscan.labels_[i] == -1):
            anomali_data.append({
                "nomor_id": driver.values[i, 1],
                "id_android": driver.values[i, 1]
            })

    driver["kluster"] = dbscan.labels_
    output = plt.scatter(x_scaled[:, 0], x_scaled[:, 1], s=100, c=driver.kluster, marker="o", alpha=1)

    plt.title("Hasil DBSCAN")
    plt.colorbar(output)
    plt.savefig(os.path.join(BASE_PATH, "static", "data.png"))
    plt.close()

    return {
        "anomali_data": anomali_data,
        "n_clusters_": n_clusters_
    }
