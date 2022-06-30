from django.shortcuts import render
from . import dbscan

def index(request):
    if request.method == "GET":
        data = dbscan.init()

        return render(request, "dashboard.html", {
            "clusters_count": str(data["n_clusters_"]),
            "anomali_data": data["anomali_data"]
        })