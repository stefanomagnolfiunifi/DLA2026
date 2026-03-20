import torch
from torch import nn
from torchvision.models import get_model

def train_epoch(model, data_loader, criterion, optimizer, device):
    
    model.train()
    correct = 0
    epoch_train_loss = 0

    for images,labels in data_loader:
        images,labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)

        # Retrieve index of highest logit
        _, predictions = torch.max(output, 1)
        correct += torch.sum(predictions == labels).item()
        
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()

        # loss.item() is the mean loss of the batch. It is multiplied for batch size to avoid last batch issue
        epoch_train_loss += loss.item() * images.size(0) 

    epoch_train_acc = correct/len(data_loader.dataset)
    epoch_train_loss = epoch_train_loss/len(data_loader.dataset)
    return epoch_train_loss, epoch_train_acc

def evaluate_model(model, data_loader, criterion, device):

    model.eval()
    epoch_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            loss = criterion(output,labels)
            
            epoch_loss += loss.item() * images.size(0)

            # Retrieve index of highest logit
            _, predictions = torch.max(output, 1)
            correct += torch.sum(predictions == labels).item()
    
    epoch_loss = epoch_loss/len(data_loader.dataset)
    epoch_acc = correct/len(data_loader.dataset)

    return epoch_loss, epoch_acc

def create_resnet18_MLP(n_outputs):

    model = get_model('resnet18', weights='DEFAULT')

    n_inputs_fc = model.fc.in_features
 
    mlp = nn.Sequential(
        nn.Linear(n_inputs_fc, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, n_outputs)
    )

    model.fc = mlp

    return model

