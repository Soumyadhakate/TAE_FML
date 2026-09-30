
import os, uuid, json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DATA = {}

def read_dataset(path, ext):
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext == ".json":
        return pd.read_json(path)
    raise ValueError("Unsupported file format.")

def prepare_dataframe(df, selected_cols=None):
    if selected_cols:
        df = df[selected_cols].copy()
    else:
        df = df.copy()

    # Drop obvious index/id columns unless explicitly selected
    auto_drop = []
    for c in df.columns:
        name = str(c).strip().lower()
        if name in {"id", "index", "unnamed: 0"}:
            auto_drop.append(c)
    if auto_drop and not selected_cols:
        df = df.drop(columns=auto_drop)

    if df.empty:
        raise ValueError("No usable columns found.")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols))

    if not transformers:
        raise ValueError("No usable numerical/categorical features found.")

    pre = ColumnTransformer(transformers)
    X = pre.fit_transform(df)

    if X.shape[1] > 30:
        n = min(10, X.shape[1], X.shape[0] - 1)
        X = PCA(n_components=max(2, n), random_state=42).fit_transform(X)

    return X, df, num_cols, cat_cols

def fig_json(fig):
    return fig.to_json()

def metric_value(fn, X, labels):
    try:
        if len(set(labels)) < 2 or len(set(labels)) >= len(X):
            return None
        return float(fn(X, labels))
    except Exception:
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Please select a dataset."), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in [".csv", ".xlsx", ".xls", ".json"]:
        return jsonify(error="Supported formats: CSV, XLSX, XLS, JSON."), 400

    token = uuid.uuid4().hex
    path = os.path.join(UPLOAD_DIR, token + ext)
    f.save(path)

    try:
        df = read_dataset(path, ext)
        DATA[token] = {"path": path, "ext": ext, "df": df}
        preview = df.head(8).replace({np.nan: None}).to_dict(orient="records")
        info = {
            "token": token,
            "filename": f.filename,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": [str(c) for c in df.columns],
            "numeric_columns": [str(c) for c in df.select_dtypes(include=np.number).columns],
            "categorical_columns": [str(c) for c in df.select_dtypes(exclude=np.number).columns],
            "missing_values": int(df.isna().sum().sum()),
            "preview": preview
        }
        return jsonify(info)
    except Exception as e:
        return jsonify(error=str(e)), 400

@app.post("/api/cluster")
def cluster():
    payload = request.get_json(force=True)
    token = payload.get("token")
    if token not in DATA:
        return jsonify(error="Dataset session expired. Upload again."), 400

    raw = DATA[token]["df"]
    cols = payload.get("columns") or None
    try:
        X, used_df, num_cols, cat_cols = prepare_dataframe(raw, cols)

        if len(X) > 5000:
            sample_n = int(payload.get("sample_n") or 5000)
            sample_n = max(500, min(sample_n, len(X)))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), size=sample_n, replace=False)
            X = X[idx]
            used_df = used_df.iloc[idx].copy()

        k = int(payload.get("clusters", 3))
        k = max(2, min(k, min(10, len(X)-1)))
        affinity = payload.get("affinity", "nearest_neighbors")
        gamma = float(payload.get("gamma", 1.0))
        neighbors = int(payload.get("neighbors", 10))
        neighbors = max(2, min(neighbors, len(X)-1))

        model = SpectralClustering(
            n_clusters=k,
            affinity=affinity,
            gamma=gamma,
            n_neighbors=neighbors,
            assign_labels="kmeans",
            random_state=42
        )
        labels = model.fit_predict(X)

        pca = PCA(n_components=2, random_state=42)
        xy = pca.fit_transform(X)

        plot_df = pd.DataFrame({"PC1": xy[:,0], "PC2": xy[:,1], "Cluster": labels.astype(str)})
        scatter = px.scatter(
            plot_df, x="PC1", y="PC2", color="Cluster",
            title="Spectral Clustering — PCA Projection",
            hover_data=["Cluster"], template="plotly_white"
        )
        scatter.update_layout(margin=dict(l=20,r=20,t=50,b=20))

        counts = pd.Series(labels).value_counts().sort_index()
        bar = px.bar(
            x=[f"Cluster {i}" for i in counts.index], y=counts.values,
            labels={"x":"Cluster", "y":"Samples"},
            title="Cluster Distribution", template="plotly_white"
        )

        # Eigenvalues from normalized graph Laplacian
        if affinity == "nearest_neighbors":
            A = kneighbors_graph(X, n_neighbors=neighbors, mode="connectivity",
                                  include_self=True).toarray()
            A = np.maximum(A, A.T)
        else:
            D = pairwise_distances(X)
            A = np.exp(-(gamma * (D ** 2)))
            np.fill_diagonal(A, 0)
        deg = A.sum(axis=1)
        inv = np.zeros_like(deg, dtype=float)
        inv[deg > 0] = 1 / np.sqrt(deg[deg > 0])
        L = np.eye(len(A)) - (inv[:,None] * A * inv[None,:])
        vals = np.linalg.eigvalsh(L)[:min(15, len(L))]
        eig = px.line(x=list(range(1, len(vals)+1)), y=vals, markers=True,
                      labels={"x":"Eigenvalue index", "y":"Eigenvalue"},
                      title="Smallest Graph-Laplacian Eigenvalues", template="plotly_white")

        # Heatmap only on a manageable subset
        hs = min(80, len(A))
        heat = px.imshow(A[:hs,:hs], title=f"Similarity Matrix (first {hs} samples)",
                         color_continuous_scale="Viridis", template="plotly_white")

        sil = metric_value(silhouette_score, X, labels)
        db = metric_value(davies_bouldin_score, X, labels)
        ch = metric_value(calinski_harabasz_score, X, labels)

        # Compare other methods on same X
        comparisons = []
        for name, pred in [
            ("Spectral", labels),
            ("K-Means", KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)),
            ("DBSCAN", DBSCAN(eps=0.8, min_samples=5).fit_predict(X))
        ]:
            valid = pred != -1
            unique = len(set(pred[valid]))
            score = None
            if unique >= 2 and valid.sum() > unique:
                score = metric_value(silhouette_score, X[valid], pred[valid])
            comparisons.append({"algorithm": name, "clusters": int(unique), "silhouette": score})

        insights = [
            f"The dataset was grouped into {k} clusters.",
            f"Cluster sizes are: " + ", ".join([f"Cluster {i}: {int(c)} samples" for i,c in counts.items()]) + ".",
            f"The 2D PCA view explains {float(pca.explained_variance_ratio_.sum()*100):.1f}% of variance in the displayed components.",
        ]
        if sil is not None:
            insights.append(f"Silhouette score: {sil:.3f}. Higher values generally indicate better-separated clusters.")
        if len(X) > 5000:
            insights.append("A representative sample was used because the original dataset was large.")

        out = used_df.copy()
        out["Cluster"] = labels
        result_path = os.path.join(RESULT_DIR, token + "_clustered.csv")
        out.to_csv(result_path, index=False)

        return jsonify({
            "rows_used": len(X),
            "features_used": int(X.shape[1]),
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "clusters": k,
            "metrics": {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch},
            "insights": insights,
            "scatter": fig_json(scatter),
            "distribution": fig_json(bar),
            "eigenvalues": fig_json(eig),
            "heatmap": fig_json(heat),
            "comparison": comparisons,
            "download": f"/download/{token}"
        })
    except Exception as e:
        return jsonify(error=str(e)), 400

@app.get("/download/<token>")
def download(token):
    path = os.path.join(RESULT_DIR, token + "_clustered.csv")
    if not os.path.exists(path):
        return "Result not found", 404
    return send_file(path, as_attachment=True, download_name="spectral_clustered_dataset.csv")

@app.post("/api/sample")
def sample():
    token = request.json.get("token")
    if token not in DATA:
        return jsonify(error="Invalid session."), 400
    df = DATA[token]["df"]
    n = min(5, len(df))
    return jsonify(df.sample(n=n, random_state=42).replace({np.nan: None}).to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
