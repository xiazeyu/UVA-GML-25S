import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
import time # For timing, optional

# --- Assume your classes (SequenceEncoder, GenresEncoder) are defined here ---
class SequenceEncoder:
    def __init__(self, model_name='all-mpnet-base-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Initialized SequenceEncoder with model {model_name} on device {self.device}")

    @torch.no_grad()
    def __call__(self, df):
        # Ensure input is a list of strings
        values = df.astype(str).tolist()
        print(f"Encoding {len(values)} items with SequenceEncoder...")
        x = self.model.encode(values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        print("Encoding complete.")
        return x.cpu()

class GenresEncoder:
    def __init__(self, sep='|'):
        self.sep = sep
        self.mapping = None # Store mapping here

    def __call__(self, df):
        print("Encoding genres...")
        # Ensure input is Series of strings
        values = df.astype(str).tolist()
        
        # Build genre set and mapping only if not already built
        if self.mapping is None:
             genres = set(g for col in values for g in col.split(self.sep) if g) # Added 'if g' to handle empty strings
             self.mapping = {genre: i for i, genre in enumerate(genres)}
             print(f"Built genre mapping with {len(self.mapping)} unique genres.")

        x = torch.zeros(len(df), len(self.mapping))
        for i, col in enumerate(values):
            # Check if col is not NaN and is a string before splitting
            if pd.notna(col) and isinstance(col, str):
                 for genre in col.split(self.sep):
                     if genre in self.mapping: # Check if genre exists in mapping
                         x[i, self.mapping[genre]] = 1
        print("Genre encoding complete.")
        return x

# --- Load your data (adjust paths if needed) ---
print("Loading edge data...")
edges = torch.load('dataset/metadata_graph.pt')
print(f"Loaded edges for {len(edges)} source ASINs.")

print("Loading review features...")
review_features = {}


print("Loading node review embeddings...")
node_review_embeddings = {}


# --- Define load_node function ---
def load_node(review_features, node_review_embeddings): # Pass loaded data as arguments
    print("Loading node metadata (parquet)...")
    try:
        df = pd.read_parquet('dataset/metadata_node.parquet')
    except FileNotFoundError:
        print("Error: metadata_node.parquet not found!")
        return None, None # Indicate failure

    print("Processing node metadata...")
    # Handle potential missing columns gracefully
    if 'salesRank' in df.columns:
        df = df.drop(columns=['salesRank'])
    
    # Ensure 'categories' column exists and process it
    if 'categories' in df.columns:
        df['categories'] = df['categories'].apply(lambda x: '|'.join(x[0]) if isinstance(x, (np.ndarray, list)) and len(x) > 0 and isinstance(x[0], (list, tuple)) else '')
    else:
        df['categories'] = '' # Add empty column if it doesn't exist

    # Fill NaN values safely
    df['title'] = df['title'].fillna('')
    df['brand'] = df['brand'].fillna('')
    if 'price' not in df.columns:
        df['price'] = 0.0 # Add price column with default if missing
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0) # Ensure numeric, handle errors

    # Set index *after* processing potential NaNs in 'asin' if it were not the index
    # If 'asin' is already the index, this is fine. If not, ensure 'asin' exists.
    if 'asin' not in df.columns and df.index.name != 'asin':
        print("Error: 'asin' column or index not found in metadata!")
        return None, None
    if 'asin' in df.columns:
         # Check for NaNs/duplicates in ASIN before setting index
         df = df.dropna(subset=['asin'])
         df = df.drop_duplicates(subset=['asin'])
         df = df.set_index('asin')

    print(f"Processed metadata for {len(df)} unique nodes.")

    # Create mapping from ASIN to integer index
    mapping = {index: i for i, index in enumerate(df.index)}
    print(f"Created mapping for {len(mapping)} nodes.")

    # --- Feature Encoding ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Check if necessary columns exist before initializing encoders
    encoders = {}
    if 'title' in df.columns:
        encoders['title'] = SequenceEncoder(device=device)
    if 'brand' in df.columns:
         encoders['brand'] = SequenceEncoder(device=device) # Reusing the same type of encoder
    if 'categories' in df.columns:
         encoders['categories'] = GenresEncoder()

    xs = []
    # Process features only if the column and encoder exist
    for col, encoder in encoders.items():
        if col in df:
            print(f"Encoding column: {col}")
            xs.append(encoder(df[col]))
        else:
             print(f"Warning: Column '{col}' not found in DataFrame, skipping encoding.")


    # Add price feature
    if 'price' in df.columns:
        print("Adding price feature...")
        price_tensor = torch.tensor(df['price'].values, dtype=torch.float).unsqueeze(1)
        xs.append(price_tensor)
    else:
        print("Warning: 'price' column not found, skipping price feature.")


    # --- Integrate Review Embeddings and Features ---
    # Check if the dictionaries are populated
    has_node_embeddings = bool(node_review_embeddings)
    has_review_features = bool(review_features)

    # Prepare tensors for embeddings and features, handling missing ASINs
    default_embedding_dim = 768 # Example dimension, adjust if known
    default_feature_dim = 128 # Example dimension, adjust if known

    if has_node_embeddings:
        print("Adding node review embeddings...")
        # Find the dimension from the first available embedding
        try:
             first_emb_key = next(iter(node_review_embeddings))
             default_embedding_dim = node_review_embeddings[first_emb_key].shape[0]
        except StopIteration:
             print("Warning: node_review_embeddings dict is empty, cannot determine dimension.")
             has_node_embeddings = False # Treat as if no embeddings

        if has_node_embeddings:
            node_embedding_list = []
            found_count = 0
            for asin in df.index:
                 if asin in node_review_embeddings:
                     node_embedding_list.append(node_review_embeddings[asin])
                     found_count += 1
                 else:
                     # Use a zero tensor for missing embeddings
                     node_embedding_list.append(torch.zeros(default_embedding_dim, dtype=torch.float))
            print(f"Found node embeddings for {found_count}/{len(df.index)} ASINs.")
            if node_embedding_list: # Only append if list is not empty
                 xs.append(torch.stack(node_embedding_list))


    if has_review_features:
        print("Adding review features...")
         # Find the dimension from the first available feature
        try:
             first_feat_key = next(iter(review_features))
             default_feature_dim = review_features[first_feat_key].shape[0]
        except StopIteration:
             print("Warning: review_features dict is empty, cannot determine dimension.")
             has_review_features = False # Treat as if no features

        if has_review_features:
            review_feature_list = []
            found_count = 0
            for asin in df.index:
                 if asin in review_features:
                     review_feature_list.append(review_features[asin])
                     found_count += 1
                 else:
                    # Use a zero tensor for missing features
                     review_feature_list.append(torch.zeros(default_feature_dim, dtype=torch.float))
            print(f"Found review features for {found_count}/{len(df.index)} ASINs.")
            if review_feature_list: # Only append if list is not empty
                 xs.append(torch.stack(review_feature_list))

    # Concatenate all features
    if not xs:
        print("Error: No features were generated!")
        return None, None

    print("Concatenating all node features...")
    x = torch.cat(xs, dim=-1)
    print(f"Final node feature matrix shape: {x.shape}")

    return x, mapping

# --- Load nodes and create initial Data object ---
x, mapping = load_node(review_features, node_review_embeddings)

# Check if loading was successful before proceeding
if x is None or mapping is None:
    print("Node loading failed. Exiting.")
    exit() # Or handle error appropriately

# Create the PyG Data object
data = Data(x=x)
print("\nInitial PyG Data object created:")
print(data)

# --- Process Edges ---
print("\nProcessing edges to create edge_index...")
start_time = time.time()

source_indices = []
target_indices = []
edges_processed = 0
edges_added = 0

# Define the edge types you want to include
edge_types_to_include = ['also_bought', 'bought_together']

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

    edges_processed += 1
    if edges_processed % 50000 == 0: # Print progress periodically
        print(f"  Processed {edges_processed}/{len(edges)} source nodes...")

# Create the edge_index tensor
# Ensure the lists are not empty before creating the tensor
if source_indices and target_indices:
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
else:
    # Handle the case with no edges: create an empty tensor with the correct shape
    edge_index = torch.empty((2, 0), dtype=torch.long)


# Add edge_index to the Data object
data.edge_index = edge_index

end_time = time.time()
print(f"\nEdge processing complete in {end_time - start_time:.2f} seconds.")
print(f"Total edges added ('also_bought' + 'bought_together'): {edges_added}")

# Print the final Data object structure
print("\nFinal PyG Data object:")
print(data)

# You can optionally save the data object
print("\nSaving Data object...")
torch.save(data, 'dataset/processed_graph_data_no_review.pt')
print("Data object saved.")