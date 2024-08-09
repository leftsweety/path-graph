import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import os
import networkx as nx
import cv2
import matplotlib.pyplot as plts

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).float()
    return correct.sum() / len(correct)

def save_model_pki(model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

class CustomDataset(Dataset):
    def __init__(self, dataframe, labels):
        self.dataframe = dataframe
        self.labels = labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data = torch.tensor(self.dataframe.iloc[idx].values, dtype=torch.float)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float)  # Ensure label is a float tensor
        return data, label

import matplotlib.pyplot as plt
import seaborn as sns

def plot_cross_validation_results(results_df):
    plt.figure(figsize=(12, 8))
    # metrics = ['train_loss', 'train_acc', 'train_f1', 'train_auc', 'val_loss', 'val_acc', 'val_f1', 'val_auc']
    metrics = ['train_loss', 'train_acc', 'train_f1', 'train_auc', 'val_loss', 'val_acc', 'val_f1', 'val_auc']

    results_melted = results_df.melt(id_vars=['fold'], value_vars=metrics, var_name='metric', value_name='value')
    sns.boxplot(x='metric', y='value', data=results_melted)
    plt.xticks(rotation=45)
    plt.title('Cross-Validation Results')
    plt.show()

def plot_graph(data, coords):
    G = nx.Graph()
    # Add nodes with their coordinates
    for i in range(data.x.shape[0]):
        G.add_node(i, pos=(coords[i][0], coords[i][1]))

    # Add edges
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)

    # Get positions for all nodes
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=20, node_color='blue', with_labels=False)
    plt.show()

import networkx as nx
import matplotlib.pyplot as plt

def plot_graph_highlight(data, coords, highlight_nodes=[]):
    G = nx.Graph()
    # Add nodes with their coordinates
    for i in range(data.x.shape[0]):
        G.add_node(i, pos=(coords[i][0], coords[i][1]))

    # Add edges
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)

    # Get positions for all nodes
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the graph
    plt.figure(figsize=(8, 8))
    node_colors = ['red' if i in highlight_nodes else 'blue' for i in range(data.x.shape[0])]
    nx.draw(G, pos, node_size=20, node_color=node_colors, with_labels=False)
    plt.show()

def show_image_by_path(image_name):
    image_path=None
    # image_path = image_dir_path+'225322/16592x_74177y.jpg'  # Replace with the path to your image
    image_path=image_name
    if not os.path.exists(image_path):
        print(f"Error: The file at path '{image_path}' does not exist.")
    else:
        # Step 2: Read the image using OpenCV
        image = cv2.imread(image_path)
        
        # Check if the image was loaded correctly
        if image is None:
            print(f"Error: Failed to load the image at path '{image_path}'.")
        else:
            # Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Step 3: Display the image using Matplotlib
            plt.figure(figsize=(8, 6))
            plt.imshow(image_rgb)
            plt.title('Image Display')
            plt.axis('off')  # Hide axis
            plt.show()