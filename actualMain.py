from decoderOnly import *
import torchvision
import torchvision.datasets as datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import *

batch_size = 32
num_epochs = 1
learning_rate = 3e-4
batch_size = 32

## DEFINING MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(trgVocabSize=256).to(device)
mnist_trainset = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ])),
  batch_size=batch_size, shuffle=True)


## DEFINING TRAINING OBJECTS/PARAMETERS
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)
criterion = nn.CrossEntropyLoss()

## TRAINING LOOP

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
    
    model.train()
    for i, (image, targets) in enumerate(mnist_trainset):
        #print(image.shape)
        image = torch.tensor(image,dtype=torch.int64)
        N, c, h, w = image.shape

        #flatten
        image = image.reshape(N,-1).to(device)
        #print(image)

        src = image[:,:-1]

        #flattens
        trg = image[:,1:].reshape(-1)

        output = model(src)
        output = output.reshape(-1, output.shape[2])

        optimizer.zero_grad()

        loss = criterion(output, trg)
        print(loss)
        #print(output.shape)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        if i == 20:
            break


predicted = translate_sentence(model, device)
print(predicted)
plt.imshow(predicted)
plt.show()