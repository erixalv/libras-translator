import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import numpy as np
from src.modelo.treino import treinar_epocas_fixas
import json

# Load dataset
dados = np.load('data/processed/dataset_final.npz', allow_pickle=True)
X = dados['X_train']
y = dados['y_train']
if 'sinalizador_train' in dados:
    sinalizadores = dados['sinalizador_train']
else:
    sinalizadores = np.full(len(y), 'desconhecido')

# Load classes
with open('vocabulario.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
classes = vocab['sinais']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DIR_SAIDA = 'data/processed'

# Train for 50 epochs
treinar_epocas_fixas(X, y, sinalizadores, len(classes), device, 50, DIR_SAIDA, classes)
print("Finished fast training")
