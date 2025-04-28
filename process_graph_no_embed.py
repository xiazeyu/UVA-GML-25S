import torch
import pandas as pd
import numpy as np
# SentenceTransformer is no longer needed if we don't encode text features
# from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
import time # For timing, optional

# --- Encoders are no longer needed ---
# class SequenceEncoder: ...
# class GenresEncoder: ...

# --- Load your data (adjust paths if needed) ---
print("Loading edge data...")
try:
    edges = torch.load('dataset/metadata_graph.pt')
    print(f"Loaded edges for {len(edges)} source ASINs.")
except FileNotFoundError as e:
    print(f"Error loading edge data: {e}. Cannot proceed without edges.")
    exit() # Exit if edges are essential

# --- Review embeddings/features are not used for node features anymore ---
# print("Loading review features...")
# review_features = {}
# try:
#     review_features |= torch.load('dataset/Digital_Music_features.pt')
#     # ... other feature files
#     print(f"Loaded review features for {len(review_features)} ASINs.")
# except FileNotFoundError as e:
#     print(f"Warning: Skipping review features loading due to error: {e}.")
#     review_features = {}

# print("Loading node review embeddings...")
# node_review_embeddings = {}
# try:
#     node_review_embeddings |= torch.load('dataset/Digital_Music_embeddings.pt')
#     # ... other embedding files
#     print(f"Loaded node review embeddings for {len(node_review_embeddings)} ASINs.")
# except FileNotFoundError as e:
#     print(f"Warning: Skipping node review embeddings loading due to error: {e}.")
#     node_review_embeddings = {}


# --- Define load_node function ---
# Removed review_features, node_review_embeddings arguments as they are not used
def load_node_metadata():
    print("Loading node metadata (parquet)...")
    try:
        df = pd.read_parquet('dataset/metadata_node.parquet')
    except FileNotFoundError:
        print("Error: metadata_node.parquet not found!")
        return None, None # Indicate failure

    print("Processing node metadata to get mapping...")
    # Keep processing needed for identifying unique nodes (ASINs)
    if 'salesRank' in df.columns:
        df = df.drop(columns=['salesRank'])

    # Ensure 'asin' column exists or is the index
    if 'asin' not in df.columns and df.index.name != 'asin':
        print("Error: 'asin' column or index not found in metadata!")
        return None, None
    if 'asin' in df.columns:
         # Check for NaNs/duplicates in ASIN before setting index
         df = df.dropna(subset=['asin'])
         df = df.drop_duplicates(subset=['asin'])
         df = df.set_index('asin')

    print(f"Identified {len(df)} unique nodes from metadata.")

    # Create mapping from ASIN to integer index
    mapping = {index: i for i, index in enumerate(df.index)}
    print(f"Created mapping for {len(mapping)} nodes.")

    # --- Feature Encoding Section REMOVED ---
    # No need to initialize encoders or process features

    # --- Integrate Review Embeddings and Features Section REMOVED ---
    # No need to load or append these features

    # --- Concatenate all features REMOVED ---
    # No 'xs' list, no torch.cat

    # Return the number of nodes and the mapping
    num_nodes = len(df.index)
    print(f"Node metadata processing complete. Found {num_nodes} nodes.")
    return num_nodes, mapping

# --- Load nodes and create initial Data object ---
# Call the modified function
num_nodes, mapping = load_node_metadata()

# Check if loading was successful before proceeding
if num_nodes is None or mapping is None:
    print("Node metadata loading failed. Exiting.")
    exit() # Or handle error appropriately

# Create the PyG Data object WITHOUT features 'x'
# Initialize with num_nodes instead
data = Data(num_nodes=num_nodes)
print("\nInitial PyG Data object created (without node features):")
print(data)

# --- Process Edges (This part remains largely the same) ---
print("\nProcessing edges to create edge_index...")
start_time = time.time()

source_indices = []
target_indices = []
edges_processed = 0
edges_added = 0

# Define the edge types you want to include
edge_types_to_include = ['also_bought', 'bought_together'] # Adjust as needed

for source_asin, connections in edges.items():
    # Check if the source node exists in our mapping
    if source_asin in mapping:
        source_idx = mapping[source_asin]

        # Iterate through the desired connection types
        for edge_type in edge_types_to_include:
            if edge_type in connections:
                # Iterate through the target ASINs for this connection type
                for target_asin in connections[edge_type]:
                    # Check if the target node also exists in our mapping
                    if target_asin in mapping:
                        target_idx = mapping[target_asin]
                        source_indices.append(source_idx)
                        target_indices.append(target_idx)
                        edges_added += 1
                    # else: # Optional: Log nodes present in edges but not in metadata
                    #     print(f"Warning: Target node {target_asin} not found in mapping.")

    edges_processed += 1
    if edges_processed % 100000 == 0: # Print progress periodically
        print(f"  Processed {edges_processed}/{len(edges)} source nodes for edges...")
# else: # Optional: Log nodes present in edges but not in metadata
#     print(f"Warning: Source node {source_asin} not found in mapping.")


# Create the edge_index tensor
# Ensure the lists are not empty before creating the tensor
if source_indices and target_indices:
    # Ensure graph is undirected (add reverse edges) if necessary for your model
    # For now, just creating the directed edges found
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    print(f"Created edge_index with shape: {edge_index.shape}")
else:
    # Handle the case with no edges: create an empty tensor with the correct shape
    print("Warning: No valid edges found between nodes in the metadata mapping.")
    edge_index = torch.empty((2, 0), dtype=torch.long)


# Add edge_index to the Data object
data.edge_index = edge_index

end_time = time.time()
print(f"\nEdge processing complete in {end_time - start_time:.2f} seconds.")
print(f"Total edges added (matching edge types and mapped nodes): {edges_added}") # This count is for one direction if graph is undirected

# Print the final Data object structure
print("\nFinal PyG Data object (without node features):")
print(data)

# You can optionally save the data object
print("\nSaving Data object...")
# Update filename to reflect no features
torch.save(data, 'dataset/processed_graph_data_no_embed.pt')
print("Data object saved as 'processed_graph_data_no_embed.pt'.")