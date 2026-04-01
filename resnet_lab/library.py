import torch
from torch import nn
from torchvision.models import get_model
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score

def train_epoch(model, data_loader, criterion, optimizer, device):
    
    model.train()
    model.to(device)
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
    model.to(device)
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


def replace_head_with_identity(model):

    # ResNet
    if hasattr(model, 'fc'):
        model.fc = nn.Identity()
    # ViT
    elif hasattr(model, 'heads'):
        model.heads == nn.Identity()
    # EfficientNet
    elif hasattr(model, 'classifier'):
        model.classifier = nn.Identity()
    
    return model


@torch.no_grad()
def extract_features(model, data_loader, device):

    model = replace_head_with_identity(model)
    model.eval()
    model.to(device)

    features = []
    labels = []

    for images, cls in data_loader:
        images = images.to(device)

        outputs = model(images)

        features.append(outputs.cpu())
        labels.append(cls.cpu())

    return torch.cat(features), torch.cat(labels)

''' Returns cosine similarity between query and gallery matrixes. Cosine similarity is calculated efficiently using dot product on normalized matrixes. '''
def query_gallery(gallery_features, query_features, normalize = True):
    
    if normalize:
        # Normalize the rows of the matrixes with L2 norm
        query_features = F.normalize(query_features, p=2, dim=1)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)
    
    # Matrix Multiplication. Result is a matrix (n_query, n_gallery)
    similarity_matrix = torch.mm(query_features, gallery_features.t())
    
    return similarity_matrix


''' Returns mean average precision for a similiraty matrix given query and gallery labels'''
def evaluate_retrieval(similarity_matrix, gallery_labels, query_labels):

    # Conversion in numpy for scikit-learn
    sim_matrix = similarity_matrix.cpu().numpy()
    q_labels = query_labels.cpu().numpy()
    g_labels = gallery_labels.cpu().numpy()
    
    num_queries = sim_matrix.shape[0]
    ap_scores = []
    
    ap_per_class = [[] for i in range (0,43)]
    for i in range(num_queries):
        
        current_query_class = q_labels[i]
        
        # Find relevant images (images with same class as the query one)
        y_true = (g_labels == current_query_class).astype(int)
        
        # Similarity values for i-th test image
        y_scores = sim_matrix[i]
        
        ap = average_precision_score(y_true, y_scores)
        ap_per_class[current_query_class].append(ap)
        ap_scores.append(ap)
        
    # mean average precision
    mAP = np.mean(ap_scores)
    
    mAP_per_class = []
    for list in ap_per_class:
        mAP_per_class.append(np.mean(list))

    mAP_per_class = [(i , round(mAP_per_class[i].item(), 4)) for i in range (0, len(mAP_per_class))]
    return mAP, mAP_per_class


def NMC_train(train_features_matrix, train_labels):

    unique_classes = torch.unique(train_labels)

    # Inizialize final matrix
    num_classes = len(unique_classes)
    feature_size = train_features_matrix.shape[1]

    mean_matrix = torch.zeros((num_classes, feature_size))    

    for i,cls in enumerate(unique_classes):

        # Retrieve only the features of the class cls
        class_features = train_features_matrix[train_labels==cls]

        mean_matrix[i] = torch.mean(class_features, dim=0)
    
    return mean_matrix


def NMC_test(mean_matrix, test_feature_matrix):

    similarity_matrix = torch.matmul(test_feature_matrix, mean_matrix.t())

    return torch.argmax(similarity_matrix, dim=1)