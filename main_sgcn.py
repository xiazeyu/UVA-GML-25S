import os.path as osp

import torch
import torch.nn.functional as F # Import F for activation functions if needed
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm  # Import tqdm

# Import SummaryWriter from torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter

# Import necessary PyG components
from torch_geometric.nn import ChebConv # Import ChebConv instead of GCNConv
from torch_geometric.utils import negative_sampling, to_undirected
from torch_geometric.transforms import RandomLinkSplit, NormalizeFeatures

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # Check for MPS availability (Apple Silicon GPU)
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# --- TensorBoard Setup ---
# Create a SummaryWriter instance. Logs will be saved in the 'runs/link_prediction_chebconv' directory.
writer = SummaryWriter(log_dir='runs/link_prediction_chebconv') # Changed log dir name
print(f"TensorBoard logs will be saved in: {writer.log_dir}")

# --- Data Loading and Preprocessing ---
# NOTE: Ensure your 'dataset/processed_graph_data.pt' exists and contains a PyG Data object
# If not, you might need to load a dataset like Planetoid first:
# from torch_geometric.datasets import Planetoid
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# dataset = Planetoid(path, name='Cora') # Or 'CiteSeer', 'PubMed'
# data = dataset[0]
# torch.save(data, 'dataset/processed_graph_data.pt') # Example saving
data_path = 'dataset/processed_graph_data_no_llm.pt'

try:
    print(f"Loading preprocessed data from {data_path}")
    writer.add_text('Data Info', f"Loading data from {data_path}")
    data = torch.load(data_path)
except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}")
    print("Please ensure the file exists or generate it.")
    # Example: Load Cora if file not found
    print("Loading Cora dataset as an example...")
    from torch_geometric.datasets import Planetoid
    path = osp.join(osp.dirname(osp.realpath('.')), 'data', 'Planetoid') # Adjusted path for notebook/script context
    dataset_cora = Planetoid(path, name='Cora')
    data = dataset_cora[0]
    # Preprocess Cora
    data.edge_index = to_undirected(data.edge_index) # Ensure undirected
    # Note: Link prediction often doesn't need node features normalized in the same way as node classification
    # data = NormalizeFeatures()(data) # Optional: Normalize features if desired

data = data.to(device)


# --- Data Splitting ---
# Ensure data has edge_index and potentially x
if not hasattr(data, 'edge_index'):
     raise ValueError("Data object must have 'edge_index'.")
if not hasattr(data, 'x'):
     print("Warning: Data object has no node features 'x'. Using identity matrix.")
     data.x = torch.eye(data.num_nodes, device=device) # Use identity features if none exist


transform = RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    neg_sampling_ratio=1.0, # Sample 1 negative edge for each positive edge
    is_undirected=True,
    add_negative_train_samples=False, # Negative samples will be generated dynamically
    split_labels=True, # Necessary for newer PyG versions to get edge_label and edge_label_index
)
train_data, val_data, test_data = transform(data) # Use the loaded 'data' object
print("\n--- Data Splits ---")
print("Train Data:", train_data)
print("Validation Data:", val_data)
print("Test Data:", test_data)
print("-" * 20)

# --- Model Definition ---
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=2): # Added K parameter
        super().__init__()
        # Use ChebConv layers
        # K is the filter size (order of Chebyshev polynomial)
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        self.conv2 = ChebConv(hidden_channels, out_channels, K=K)

    def encode(self, x, edge_index):
        # Apply ReLU activation
        # Note: ChebConv does not require graph normalization like GCNConv typically does (e.g., add_self_loops, norm)
        # as normalization is handled internally based on eigenvalues.
        x = self.conv1(x, edge_index).relu()
        # Consider adding dropout: x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        # Dot product decoder remains the same
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        # Predict all potential links (adjacency matrix)
        prob_adj = z @ z.t()
        # Return indices where probability is positive (or above a threshold)
        # Note: This can be computationally expensive for large graphs
        return (prob_adj > 0).nonzero(as_tuple=False).t()


# --- Initialization ---
# Adjust K as needed (e.g., 2, 3). Higher K increases computational cost.
model = Net(data.num_features, 256, 128, K=2).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss() # Suitable for binary classification with logits

# --- Training Function ---
def train(epoch):
    model.train()
    optimizer.zero_grad()

    # Encode node features using ChebConv
    # Pass train_data.edge_index which contains the message passing edges
    z = model.encode(train_data.x, train_data.edge_index)

    # Dynamic negative sampling for each epoch
    # Use the message passing edges (train_data.edge_index) to avoid sampling validation/test edges
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), # Sample same number as positive supervision edges
        method='sparse' # Efficient method for sparse graphs
    )

    # Combine positive supervision edges and negative edges for loss calculation
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    # Create corresponding labels (1 for positive, 0 for negative)
    edge_label = torch.cat([
        train_data.edge_label, # Should be all 1s for positive supervision edges
        train_data.edge_label.new_zeros(neg_edge_index.size(1)) # 0s for negative edges
    ], dim=0)

    # Get predictions (logits) and calculate loss
    out = model.decode(z, edge_label_index).view(-1) # Ensure output is flat
    loss = criterion(out, edge_label)

    # Backpropagate and update weights
    loss.backward()
    optimizer.step()

    # Log training loss to TensorBoard
    writer.add_scalar('Loss/train', loss.item(), epoch)

    return loss.item() # Return scalar value of loss


# --- Testing Function ---
@torch.no_grad() # Disable gradient calculations for evaluation
def test(data_split, epoch, split_name): # Renamed 'data' to 'data_split' to avoid confusion
    model.eval() # Set model to evaluation mode

    # Encode node features using the message passing edges from the specific split
    z = model.encode(data_split.x, data_split.edge_index)

    # Decode using the supervision edges for the specific split
    out = model.decode(z, data_split.edge_label_index).view(-1).sigmoid() # Apply sigmoid for probabilities

    # Calculate ROC AUC score using the supervision labels
    auc = roc_auc_score(data_split.edge_label.cpu().numpy(), out.cpu().numpy())
    # Calculate Average Precision score
    ap = average_precision_score(data_split.edge_label.cpu().numpy(), out.cpu().numpy())

    # Log validation/test AUC/AP to TensorBoard
    writer.add_scalar(f'AUC/{split_name}', auc, epoch)
    writer.add_scalar(f'AP/{split_name}', ap, epoch)

    return auc, ap # Return both AUC and AP


# --- Training Loop ---
best_val_auc = final_test_auc = final_test_ap = 0
num_epochs = 1000 # Define number of epochs

# Wrap the range with tqdm for a progress bar
print("\n--- Starting Training ---")
pbar = tqdm(range(1, num_epochs + 1), desc="Training Progress")
for epoch in pbar:
    loss = train(epoch)
    val_auc, val_ap = test(val_data, epoch, 'val')
    test_auc, test_ap = test(test_data, epoch, 'test') # Test on test set in each epoch

    # Track best validation score and corresponding test score
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
        final_test_ap = test_ap # Store corresponding test AP
        # Optional: Save the best model checkpoint
        # torch.save(model.state_dict(), 'best_model_chebconv.pt')
        # print(f"\nEpoch {epoch:03d}: New best validation AUC: {best_val_auc:.4f}") # Print only when improved


    # Update tqdm progress bar description with current metrics
    pbar.set_postfix({'Loss': f'{loss:.4f}', 'Val AUC': f'{val_auc:.4f}', 'Test AUC': f'{test_auc:.4f}'})

    # Print metrics less frequently to avoid clutter
    # if epoch % 50 == 0 or epoch == 1: # Print every 50 epochs and the first epoch
    #     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')

# --- Final Results ---
print("\n" + "-" * 20)
print(f'Training finished.')
print(f'Best Validation AUC: {best_val_auc:.4f}')
print(f'Final Test AUC (at best validation AUC): {final_test_auc:.4f}')
print(f'Final Test AP (at best validation AUC): {final_test_ap:.4f}')

# --- Close TensorBoard Writer ---
writer.close()
print(f"TensorBoard writer closed. Logs saved in {writer.log_dir}")

# --- Optional: Get Final Predictions ---
# Load the best model if saved
# model.load_state_dict(torch.load('best_model_chebconv.pt'))
model.eval()
with torch.no_grad():
    # Encode using the test data's message passing edges
    z = model.encode(test_data.x, test_data.edge_index)
    # Decode using the test data's supervision edges
    final_test_scores = model.decode(z, test_data.edge_label_index).view(-1).sigmoid()
    print(f"\nFinal Test Scores (first 10): {final_test_scores[:10].cpu().numpy()}")
    print(f"Corresponding True Labels (first 10): {test_data.edge_label[:10].cpu().numpy()}")
    # final_edge_index = model.decode_all(z) # Predict all links
    # print(f"\nPredicted final edge index shape: {final_edge_index.shape}")

