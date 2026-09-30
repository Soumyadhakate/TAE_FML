# Spectral Clustering for Pattern Discovery

A Flask-based interactive web application for discovering hidden patterns using Spectral Clustering.

## Features
- CSV / XLSX / XLS / JSON upload
- Dataset preview and statistics
- Numerical + categorical preprocessing
- Standardization, imputation and one-hot encoding
- Spectral Clustering with RBF / nearest-neighbor affinity
- PCA visualization
- Similarity matrix heatmap
- Graph-Laplacian eigenvalue visualization
- Cluster distribution
- Silhouette, Davies-Bouldin and Calinski-Harabasz metrics
- Comparison with K-Means and DBSCAN
- Large-dataset sampling safeguard
- Download clustered CSV

## Run

Python 3.10+ recommended.

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## Notes
Spectral Clustering can become computationally expensive as the number of samples grows because graph/similarity operations scale with dataset size. This project automatically samples datasets above 5,000 rows.
