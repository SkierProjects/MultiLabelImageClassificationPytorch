import csv
import torch
from imclaslib.config import Config
import imclaslib.files.pathutils as pathutils
import imclaslib.dataset.datasetutils as datasetutils
# Define edge weights for different types of edges
EDGE_WEIGHTS_DICT = {
    'Usually mutually exclusive': -1.0,
    'Somewhat common Together': 0.5,
    'Very common together': 0.8,
    'Parent': 1.0,
    'Somewhat Mutually Exclusive': -0.5,
}
config = Config("default_config.yml")
# Function to read the CSV and generate edge indexes and edge weights
def generate_graph_edges(csv_filename):
    label_id_dict = datasetutils.get_tag_to_index_mapping(config)
    # Create a mapping from label IDs to numerical indices
    label_indices = {label_id: idx for idx, label_id in enumerate(label_id_dict.values())}
    
    source_nodes = []
    target_nodes = []
    edge_weights_list = []
    
    with open(csv_filename, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',')
        
        for row in csv_reader:
            # Get the indices from the label IDs
            from_idx = label_indices.get(label_id_dict.get(row['From Name']))
            to_idx = label_indices.get(label_id_dict.get(row['To Name']))
            
            # Ignore if the label is not found in the dictionary
            if from_idx is None or to_idx is None:
                continue
            
            # Add the edge to the lists
            source_nodes.append(from_idx)
            target_nodes.append(to_idx)
            
            # Get the edge weight based on edge type
            weight = EDGE_WEIGHTS_DICT.get(row['Edge Type'], 1.0)  # Default weight is 1.0 if not found
            edge_weights_list.append(weight)
            
            # If the edge is not of type 'Parent' and is not directional, add the reverse edge
            if row['Edge Type'] != 'Parent':
                source_nodes.append(to_idx)
                target_nodes.append(from_idx)
                edge_weights_list.append(weight)
                
    # Convert lists to PyTorch tensors
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_weights = torch.tensor(edge_weights_list, dtype=torch.float32)
    
    return edge_index, edge_weights

# Replace 'graph_commons.csv' with the actual path to your CSV file
edge_index, edge_weights = generate_graph_edges(pathutils.get_graph_path(config))

# Print the edge index and weights in the desired format
print('edge_index =', edge_index)
print('edge_weights =', edge_weights)