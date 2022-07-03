from django.shortcuts import render
from . import dbscan

def index(request):
    if request.method == "GET":
        data = dbscan.generatePicture()

        return render(request, "dashboard.html", {
            "clusters_count": data["n_clusters_"],
            "clusters_range": range(data["n_clusters_"])
        })

def search(request):
    if request.method == "POST":
        cluster = request.POST['cluster']

        penguins = dbscan.searchClusterData(int(cluster))

        return render(request, "pencarian.html", {
            "cluster": cluster,
            "penguins": penguins
        })