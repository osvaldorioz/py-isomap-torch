from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import torch
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import isomap_module as im
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/isomap")
def calculo(num_samples: int, num_features: int, num_components: int, dimension: int):
    output_file_1 = 'dispersion_isomap.png'
    output_file_2 = 'heatmap_isomap.png'

    # Generar datos de prueba
    np.random.seed(42)
    X = np.random.rand(num_samples, num_features).astype(np.float32)
    X_torch = torch.tensor(X)

    # Ejecutar Isomap
    k = num_components
    d = dimension
    Y, G_floyd = im.isomap(X_torch, k, d)
    Y = Y.numpy()
    G_floyd = G_floyd.numpy()

    # Grafico de dispersion
    plt.figure(figsize=(8, 6))
    plt.scatter(Y[:, 0], Y[:, 1], c=np.arange(Y.shape[0]), cmap='viridis')
    plt.colorbar(label="Índice del punto")
    plt.title("Isomap - Proyección en 2D")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    #plt.show()
    plt.savefig(output_file_1)

    # Heatmap de distancias geodesicas
    plt.figure(figsize=(8, 6))
    sns.heatmap(G_floyd, cmap="coolwarm", square=True)
    plt.title("Heatmap de Distancias Geodésicas")
    #plt.show()
    plt.savefig(output_file_2)
    
    plt.close()
    
    j1 = {
        "Grafica de dispersion": output_file_1,
        "Grafica Heatmap": output_file_2
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/isomap-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)