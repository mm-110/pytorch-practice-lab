from sklearn.decomposition import PCA
import torch
import timm
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
model = model.eval().to(device)

# Load and process data
transform = transforms.Compose([
    transforms.Resize(224),  # Ridimensiona le immagini a 224x224
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.MNIST(root='./data', 
                          train=True, 
                          download=True, 
                          transform=transform)
testset = datasets.MNIST(root='./data', 
                         train=False, 
                         download=True, 
                         transform=transform)

# Create a random subset of the training dataset
subset_indices = random.sample(range(len(trainset)), len(trainset) // 2)
train_subset = torch.utils.data.Subset(trainset, subset_indices)

tranloader = torch.utils.data.DataLoader(train_subset, 
                                         batch_size=2, 
                                         shuffle=True, 
                                         num_workers=2)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=2, 
                                         shuffle=False, 
                                         num_workers=2)

class_names = trainset.classes
print(class_names)

# Define class
class ViTFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(ViTFeatureExtractor, self).__init__()
        self.model = model
        self.model.head = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        return x
    
feature_extractor = ViTFeatureExtractor(model).to(device)

def get_embeddings(dataloader, model):
    embeddings = []
    labels = []
    with torch.no_grad():
        for i, (images, label) in enumerate(dataloader):
            print(f'Iter no: {i+1} / {len(dataloader)}', end='\r')
            images = images.to(device)
            # label = label.to(device)  # Sposta le etichette sul dispositivo

            output = model(images)

            embeddings.append(output)
            labels.append(label)

            torch.cuda.empty_cache()

    return torch.cat(embeddings), torch.cat(labels)

if 'embeddings.pth' in os.listdir():
    checkpoint = torch.load('embeddings.pth')
    embeddings = checkpoint['embeddings']
    labels = checkpoint['labels']
else:
    embeddings, labels = get_embeddings(tranloader, feature_extractor)
    torch.save({'embeddings': embeddings, 'labels': labels}, 'embeddings.pth')

print(f'Embeddings shape: {embeddings.shape}')

# Esegui il clustering sugli embedding
num_clusters = len(class_names)  # Numero di cluster desiderato
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings.cpu().numpy())

print(type(clusters))

# Crea un dizionario per mappare i cluster alle etichette
cluster_labels = defaultdict(list)
for idx, cluster in enumerate(clusters):
    cluster_labels[cluster].append(labels[idx].item())

# Stampa le etichette per ogni cluster
for cluster, label_list in cluster_labels.items():
    label_count = Counter(label_list)
    total_labels = len(label_list)
    print(f"Cluster {cluster}:")
    for label, count in label_count.items():
        percentage = (count / total_labels) * 100
        print(f"  Etichetta {label}: {percentage:.2f}%")

predominant_labels = {}
for cluster, label_list in cluster_labels.items():
    label_count = Counter(label_list)
    predominant_label, max_count = label_count.most_common(1)[0]
    predominant_labels[cluster] = (predominant_label, max_count / len(label_list) * 100)

# Stampa le etichette predominanti per ciascun cluster
for cluster, (label, percentage) in predominant_labels.items():
    print(f"Cluster {cluster}: Etichetta predominante {label} con {percentage:.2f}%")

# Verifica se la stessa etichetta è predominante su più cluster
label_clusters = defaultdict(list)
for cluster, (label, _) in predominant_labels.items():
    label_clusters[label].append(cluster)

for label, clusters in label_clusters.items():
    if len(clusters) > 1:
        print(f"Etichetta {label} è predominante in più cluster: {clusters}")