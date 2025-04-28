import os.path as osp

import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm  # Import tqdm

# Import SummaryWriter from torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.datasets import Planetoid # Although imported, not used directly if loading preprocessed data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit, ToUndirected, NormalizeFeatures

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
# Create a SummaryWriter instance. Logs will be saved in the 'runs/link_prediction_gcn' directory.
# You can change the directory name as needed.
writer = SummaryWriter()
print(f"TensorBoard logs will be saved in: {writer.log_dir}")

# --- Data Loading and Preprocessing ---
data_path = 'dataset/processed_graph_data.pt'
print(f"Loading preprocessed data from {data_path}")
writer.add_text('Data Info', f"Loading data from {data_path}")
dataset = torch.load(data_path)

# Apply necessary transforms
dataset = ToUndirected()(dataset)
dataset = NormalizeFeatures()(dataset)
dataset = dataset.to(device)


# --- Data Splitting ---
transform = RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    neg_sampling_ratio=1.0, # Sample 1 negative edge for each positive edge
    is_undirected=True,
    add_negative_train_samples=False, # Negative samples will be generated dynamically in training loop
)
train_data, val_data, test_data = transform(dataset)
print("\n--- Data Splits ---")
print("Train Data:", train_data)
print("Validation Data:", val_data)
print("Test Data:", test_data)
print("-" * 20)

# --- Model Definition ---
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Use He initialization for GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        # Apply ReLU activation and dropout for potentially better performance
        x = self.conv1(x, edge_index).relu()
        # Consider adding dropout: x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        # Dot product decoder
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        # Predict all potential links (adjacency matrix)
        prob_adj = z @ z.t()
        # Return indices where probability is positive (or above a threshold)
        # Note: This can be computationally expensive for large graphs
        return (prob_adj > 0).nonzero(as_tuple=False).t()


# --- Initialization ---
model = Net(dataset.num_features, 256, 128).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss() # Suitable for binary classification with logits

# --- Training Function ---
def train(epoch):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # Dynamic negative sampling for each epoch
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), # Sample same number as positive edges
        method='sparse' # Efficient method for sparse graphs
    )

    # Combine positive and negative edges for loss calculation
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    # Create corresponding labels (1 for positive, 0 for negative)
    edge_label = torch.cat([
        train_data.edge_label, # Should be all 1s for positive edges
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
def test(data, epoch, split_name):
    model.eval() # Set model to evaluation mode
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid() # Apply sigmoid for probabilities

    # Calculate ROC AUC score
    auc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    # Calculate Average Precision score
    ap = average_precision_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

    # Log validation/test AUC to TensorBoard
    writer.add_scalar(f'AUC/{split_name}', auc, epoch)
    writer.add_scalar(f'AP/{split_name}', ap, epoch)

    return auc


# --- Training Loop ---
best_val_auc = final_test_auc = 0
num_epochs = 1000 # Define number of epochs

# Wrap the range with tqdm for a progress bar
print("\n--- Starting Training ---")
for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress"):
    loss = train(epoch)
    val_auc = test(val_data, epoch, 'val')
    test_auc = test(test_data, epoch, 'test')

    # Track best validation score and corresponding test score
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
        # Optional: Save the best model checkpoint
        # torch.save(model.state_dict(), 'best_model.pt')
        print(f"\nEpoch {epoch:03d}: New best validation AUC: {best_val_auc:.4f}")


    # Update tqdm progress bar description with current metrics (optional)
    # pbar.set_postfix({'Loss': loss:.4f, 'Val AUC': val_auc:.4f, 'Test AUC': test_auc:.4f})

    # Print metrics every few epochs or keep as is
    if epoch % 10 == 0 or epoch == 1: # Print every 10 epochs and the first epoch
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')

# --- Final Results ---
print("-" * 20)
print(f'Training finished.')
print(f'Best Validation AUC: {best_val_auc:.4f}')
print(f'Final Test AUC (corresponding to best validation): {final_test_auc:.4f}')

# --- Close TensorBoard Writer ---
writer.close()
print(f"TensorBoard writer closed. Logs saved in {writer.log_dir}")

# --- Optional: Get Final Predictions ---
# Load the best model if saved
# model.load_state_dict(torch.load('best_model.pt'))
model.eval()
with torch.no_grad():
    z = model.encode(test_data.x, test_data.edge_index)
    # final_edge_index = model.decode_all(z) # Predict all links
    # print(f"\nPredicted final edge index shape: {final_edge_index.shape}")
    # Alternatively, get scores for test edges:
    final_test_scores = model.decode(z, test_data.edge_label_index).view(-1).sigmoid()
    print(f"\nFinal Test Scores (first 10): {final_test_scores[:10]}")
    print(f"Corresponding True Labels (first 10): {test_data.edge_label[:10]}")