import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import numpy as np
import joblib
from src.modelo.arquiteturas import ClassificadorLSTM
import json

# Load dataset
dados = np.load('data/processed/dataset_final.npz', allow_pickle=True)
X_test = dados['X_test']
y_test = dados['y_test']

# Load classes
with open('vocabulario.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
classes = vocab['sinais']

# Load scaler and transform X_test
scaler = joblib.load('data/processed/scaler.pkl')
T = X_test.shape[1]
F = X_test.shape[2]
X_flat = X_test.reshape(-1, F)
X_flat = scaler.transform(X_flat)
X_test = X_flat.reshape(-1, T, F)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = ClassificadorLSTM(n_features=X_test.shape[-1], n_classes=len(classes)).to(device)
checkpoint = torch.load('data/processed/modelo_melhor.pt', map_location=device)
modelo.load_state_dict(checkpoint['state_dict'])
modelo.eval()

# Test
with torch.no_grad():
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    out = modelo(X_t)
    preds = out.argmax(dim=1).cpu().numpy()
    
acc = (preds == y_test).mean()
print(f'Acurácia no dataset de teste: {acc * 100:.2f}%')
