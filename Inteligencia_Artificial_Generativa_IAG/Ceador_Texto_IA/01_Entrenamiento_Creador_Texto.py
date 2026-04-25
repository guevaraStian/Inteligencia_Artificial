# El siguiente software sirve para entrenar el software con la base de datos
# coge una base de datos y entrena el software con esta informacion
# Usando la librerias torch, tqdm y sus datos
# En el terminal de comandos importar librerias con el comando PIP 
# pip install torch tqdm
# pip 25.1.1
# Python 3.13.1
import torch
from torch.utils.data import DataLoader
from dataset import TextDataset
from Model import TextGenModel
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

Entrenamiento_data = TextDataset("data/Entrenamiento.txt")
valid_data = TextDataset("data/valid.txt")

Entrenamiento_loader = DataLoader(Entrenamiento_data, batch_size=64, shuffle=True)

Model = TextGenModel(Entrenamiento_data.vocab_size).to(device)
optimizer = torch.optim.Adam(Model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

EPOCHS = 3

for epoch in range(EPOCHS):
    Model.train()
    pbar = tqdm(Entrenamiento_loader)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits, _ = Model(x)
        loss = criterion(logits.reshape(-1, Entrenamiento_data.vocab_size), y.reshape(-1))

        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    torch.save(Model.state_dict(), "model.pth")

print("Entrenamiento completo.")