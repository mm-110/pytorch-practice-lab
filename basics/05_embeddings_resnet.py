from sklearn.decomposition import PCA
import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict, Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = model.eval().to(device)

# Load and process data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', 
                           train=True, 
                           download=True, 
                           transform=transform)
testset = datasets.CIFAR10(root='./data', 
                           train=False, 
                           download=True, 
                           transform=transform)
tranloader = torch.utils.data.DataLoader(trainset, 
                                         batch_size=4, 
                                         shuffle=True, 
                                         num_workers=2)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=4, 
                                         shuffle=False, 
                                         num_workers=2)

class_names = trainset.classes
print(class_names)

# Define class
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)
    
feature_extractor = FeatureExtractor(model).to(device)

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

    return torch.cat(embeddings), torch.cat(labels)

if 'embeddings.pth' in os.listdir():
    checkpoint = torch.load('embeddings.pth')
    embeddings = checkpoint['embeddings']
    labels = checkpoint['labels']
else:
    embeddings, labels = get_embeddings(tranloader, feature_extractor)
    torch.save({'embeddings': embeddings, 'labels': labels}, 'embeddings.pth')

print(f'Embeddings shape: {embeddings.shape}')

# tsne = TSNE(n_components=2, random_state=42)
# pca = PCA(n_components=2)
# embeddings_2d = pca.fit_transform(embeddings.cpu().numpy())

# Esegui il clustering sugli embedding
num_clusters = 10  # Numero di cluster desiderato
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