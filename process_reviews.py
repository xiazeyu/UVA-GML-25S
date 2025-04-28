import pandas as pd
import gzip
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm # Ensure tqdm is imported
import argparse
import os
import sys

# --- Helper Function to Parse Gzip JSON ---
def parse(path):
  """Generator to parse gzipped JSON lines."""
  try:
    with gzip.open(path, 'rb') as g:
      for l in g:
        # Use bytes.decode() and then eval (use json.loads for safety if data source isn't trusted)
        try:
            yield eval(l.decode('utf-8'))
        except (SyntaxError, NameError, TypeError) as e:
            print(f"Warning: Skipping line due to parsing error: {e} - Line content (first 100 chars): {l[:100]}...", file=sys.stderr)
            continue # Skip problematic lines
  except FileNotFoundError:
    print(f"Error: Input file not found at {path}", file=sys.stderr)
    sys.exit(1)
  except Exception as e:
    print(f"Error reading or parsing file {path}: {e}", file=sys.stderr)
    sys.exit(1)

# --- Helper Function to Load DataFrame ---
def getDF(path):
  """Loads data from parsed gzip JSON into a Pandas DataFrame with tqdm progress."""
  print(f"Loading data from {path}...")
  i = 0
  df_dict = {}
  # Use tqdm to iterate over the generator from parse()
  # Since we don't know the total lines beforehand without reading the file twice,
  # tqdm will show iteration count and rate without a percentage bar.
  # Extract filename for clearer description.
  filename = os.path.basename(path)
  parser = parse(path)
  # Wrap the parser generator with tqdm
  for d in tqdm(parser, desc=f"Parsing {filename}", unit=" lines"):
    df_dict[i] = d
    i += 1

  if not df_dict:
      print(f"Error: No data successfully loaded from {path}. File might be empty or corrupted.", file=sys.stderr)
      sys.exit(1)

  print(f"Loaded {len(df_dict)} records.")
  print("Creating DataFrame...")
  # DataFrame creation itself can take time, add print statement
  df = pd.DataFrame.from_dict(df_dict, orient='index')
  print("DataFrame created.")
  return df

# --- Main Processing Function ---
def process_category(category_name):
    """
    Processes a given category: loads data, generates embeddings, calculates features,
    and saves the results.
    """
    # --- Define File Paths ---
    input_path = f'dataset/reviews_{category_name}_5.json.gz'
    embeddings_output_path = f'dataset/{category_name}_embeddings.pt'
    features_output_path = f'dataset/{category_name}_features.pt'
    output_dir = 'dataset'

    # --- Ensure Output Directory Exists ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' ensured.")

    # --- Load Data (Now with tqdm inside getDF) ---
    df = getDF(input_path)

    # --- Check for required columns ---
    required_columns = ['asin', 'summary', 'reviewText', 'overall', 'helpful']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        print(f"Error: Missing required columns in {input_path}: {missing}", file=sys.stderr)
        sys.exit(1)
    # Drop rows where 'asin' is missing, as it's crucial for grouping
    initial_rows = len(df)
    df.dropna(subset=['asin'], inplace=True)
    if len(df) < initial_rows:
        print(f"Warning: Dropped {initial_rows - len(df)} rows with missing 'asin'.")


    # ========== Step 1: Generate Sentence Embeddings ==========
    print("\n--- Generating Sentence Embeddings ---")

    # --- Load Model ---
    print("Loading SentenceTransformer model (all-mpnet-base-v2)...")
    # Consider adding try-except for model loading if network issues are possible
    try:
        model = SentenceTransformer('all-mpnet-base-v2')
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}", file=sys.stderr)
        print("Please ensure you have an internet connection or the model is cached.", file=sys.stderr)
        sys.exit(1)
    print("Model loaded.")

    # --- Prepare Text ---
    print("Preparing text for embedding...")
    df['text_to_embed'] = df['summary'].fillna('') + ". " + df['reviewText'].fillna('')
    texts_to_embed = df['text_to_embed'].tolist()
    print(f"Prepared {len(texts_to_embed)} review texts for embedding.")

    # --- Encode Text in Batches (already has tqdm) ---
    batch_size = 1024 # Adjust based on your GPU/CPU memory
    all_embeddings = []
    print("Starting embedding generation...")
    # This loop already uses tqdm
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Generating Embeddings", unit=" batch"):
        batch = texts_to_embed[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)

    embeddings_array = np.array(all_embeddings)
    print(f"Generated embeddings of shape: {embeddings_array.shape}")

    # --- Aggregate Embeddings per Product (ASIN) ---
    print("Aggregating embeddings per product (ASIN)...")
    product_embeddings = {}
    # Add embeddings to DataFrame for easier grouping
    # Check if embeddings_array length matches df length, handle potential mismatch
    if len(embeddings_array) != len(df):
         print(f"Warning: Mismatch between number of embeddings ({len(embeddings_array)}) and DataFrame rows ({len(df)}). This might indicate an issue.", file=sys.stderr)
         # Decide how to handle: error out, or try to proceed cautiously?
         # For now, we'll try to proceed but this needs investigation if it occurs.
         # Option: align based on index if possible, but risky if order changed.
         # Safest might be erroring:
         # print("Error: Embedding count mismatch. Exiting.", file=sys.stderr)
         # sys.exit(1)
         # Let's add as is, but be aware grouping might fail or be incorrect
         min_len = min(len(embeddings_array), len(df))
         df = df.iloc[:min_len].copy() # Take subset matching embeddings count
         embeddings_array = embeddings_array[:min_len]
         print(f"Adjusted DataFrame and embeddings to minimum length: {min_len}")

    df['embedding_list'] = list(embeddings_array)

    # Group by product ID and calculate mean embeddings
    print("Grouping embeddings by ASIN...")
    # This groupby itself can be heavy, print statement indicates start
    product_groups = df.groupby('asin')['embedding_list'].apply(list)
    print(f"Found {len(product_groups)} unique products (ASINs).")

    embedding_dim = model.get_sentence_embedding_dimension()
    # This loop already uses tqdm
    for asin, embedding_list in tqdm(product_groups.items(), desc="Aggregating Embeddings", unit=" product"):
        if embedding_list and len(embedding_list) > 0:
            # Filter out potential Nones or vectors of incorrect shape
            valid_embeddings = [emb for emb in embedding_list if isinstance(emb, np.ndarray) and emb.shape == (embedding_dim,)]
            if valid_embeddings:
                 product_embeddings[asin] = torch.tensor(np.mean(np.vstack(valid_embeddings), axis=0), dtype=torch.float32)
            else:
                 # print(f"Warning: No valid embeddings found for ASIN {asin} after filtering.")
                 product_embeddings[asin] = torch.zeros(embedding_dim, dtype=torch.float32) # Assign zero vector
        else:
            # Handle products with no reviews or where embedding list was empty/problematic
            # print(f"Warning: No embeddings found for ASIN {asin}.")
            product_embeddings[asin] = torch.zeros(embedding_dim, dtype=torch.float32)

    print(f"Aggregated embeddings for {len(product_embeddings)} products.")

    # --- Save Embeddings ---
    print(f"Saving aggregated product embeddings to {embeddings_output_path}...")
    torch.save(product_embeddings, embeddings_output_path)
    print("Embeddings saved successfully.")


    # ========== Step 2: Calculate and Save Features ==========
    print("\n--- Calculating Product Features ---")

    # --- Calculate Average Rating ---
    print("Calculating average overall rating...")
    # This groupby can be heavy
    product_avg_rating = df.groupby('asin')['overall'].mean().reset_index()
    product_avg_rating = product_avg_rating.rename(columns={'overall': 'avg_overall_rating'})
    print("Average rating calculation complete.")

    # --- Calculate Helpfulness Features ---
    print("Calculating helpfulness features...")
    # Extract helpfulness components per review (vectorized apply, usually fast)
    print("Extracting helpful votes...")
    helpful_values = df['helpful'].apply(lambda x: x if isinstance(x, list) and len(x) == 2 else [0, 0]).tolist()
    helpful_arr = np.array(helpful_values, dtype=np.int32)
    df['helpful_votes'] = helpful_arr[:, 0]
    df['total_votes'] = helpful_arr[:, 1]

    # Calculate per-review helpfulness ratio (vectorized, usually fast)
    print("Calculating per-review helpfulness ratio...")
    # Use np.divide for safe division
    with np.errstate(divide='ignore', invalid='ignore'): # Suppress division by zero warnings
        df['helpful_ratio'] = np.divide(df['helpful_votes'], df['total_votes'])
    # Handle NaN (from 0/0) and inf (from x/0 where x>0)
    df['helpful_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['helpful_ratio'].fillna(0.5, inplace=True) # Fill NaN with neutral 0.5

    # Aggregate to product level
    print("Aggregating helpfulness data per product...")
    # Calculate product-level sums (groupby can be heavy)
    product_helpful_sums = df.groupby('asin')[['helpful_votes', 'total_votes']].sum().reset_index()

    # Calculate product-level helpfulness ratio (vectorized, usually fast)
    with np.errstate(divide='ignore', invalid='ignore'):
        product_helpful_sums['product_level_helpful_ratio'] = np.divide(
            product_helpful_sums['helpful_votes'], product_helpful_sums['total_votes']
        )
    product_helpful_sums['product_level_helpful_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    product_helpful_sums['product_level_helpful_ratio'].fillna(0.5, inplace=True) # Fill NaN with neutral 0.5

    # Calculate average helpfulness ratio per product (groupby can be heavy)
    product_avg_helpful_ratio = df.groupby('asin')['helpful_ratio'].mean().reset_index()
    product_avg_helpful_ratio = product_avg_helpful_ratio.rename(columns={'helpful_ratio': 'avg_helpful_ratio'})
    print("Helpfulness feature aggregation complete.")

    # Combine helpfulness features (merge can be heavy)
    print("Merging helpfulness features...")
    product_helpful_features = pd.merge(
        product_helpful_sums[['asin', 'helpful_votes', 'total_votes', 'product_level_helpful_ratio']],
        product_avg_helpful_ratio,
        on='asin',
        how='left' # Keep all products from sums, even if avg ratio is missing (shouldn't happen with groupby mean)
    )
    print("Helpfulness features merged.")

    # --- Combine All Features ---
    print("Combining all calculated features...")
    # Merge can be heavy
    product_features_final = pd.merge(product_avg_rating, product_helpful_features, on='asin', how='left') # Start with avg rating, add helpfulness
    print("All features merged.")

    # Fill any potential NaNs resulting from merges or calculations
    print("Filling potential NaN values in final features...")
    product_features_final = product_features_final.fillna({
        'avg_overall_rating': 0.0, # Assign a default rating (e.g., 0 or neutral 3.0?)
        'helpful_votes': 0,
        'total_votes': 0,
        'product_level_helpful_ratio': 0.5, # Neutral ratio
        'avg_helpful_ratio': 0.5 # Neutral ratio
    })
    print("NaN filling complete.")
    print("Final Product Features Head:")
    print(product_features_final.head())

    # --- Convert Features to Torch Dictionary ---
    features_dict_torch = {}
    print("\nConverting features DataFrame to asin: tensor dictionary...")
    feature_cols = [
        'avg_overall_rating',
        'avg_helpful_ratio',
        'helpful_votes',
        'total_votes',
        'product_level_helpful_ratio'
    ]
    # Ensure columns exist before iterating
    missing_feature_cols = [col for col in feature_cols if col not in product_features_final.columns]
    if missing_feature_cols:
        print(f"Error: Missing expected feature columns for conversion: {missing_feature_cols}", file=sys.stderr)
        sys.exit(1)

    # This loop already uses tqdm
    for index, row in tqdm(product_features_final.iterrows(), total=product_features_final.shape[0], desc="Converting Features", unit=" product"):
        asin = row['asin']
        # Ensure features are in the correct order and are float32
        features_tensor = torch.tensor([row[col] for col in feature_cols], dtype=torch.float32)
        features_dict_torch[asin] = features_tensor

    # --- Save Features ---
    print(f"Saving product features dictionary to {features_output_path}...")
    torch.save(features_dict_torch, features_output_path)
    print("Features saved successfully.")
    print(f"\nProcessing for category '{category_name}' complete.")


# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Amazon review data to generate product embeddings and features.")
    parser.add_argument("category_name",
                        type=str,
                        help="The name of the category to process (e.g., 'Books', 'Video_Games'). "
                             "Expects input file dataset/reviews_{category_name}_5.json.gz")

    args = parser.parse_args()

    # --- Run Processing ---
    process_category(args.category_name)