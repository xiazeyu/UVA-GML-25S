import torch
import gzip
import tqdm

asin_set = set()

for category in [
    'Digital_Music',
    'Video_Games',
    'Pet_Supplies',
]:
    data = torch.load(f'dataset/{category}_embeddings.pt')
    asin_set = asin_set | set(data.keys())
    print(f'Number of ASINs in {category}: {len(data)}')

# Your provided parse function (using eval as requested)
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    try:
      # If your data is standard JSON per line, this is safer:
      # yield json.loads(l)
      yield eval(l) # Using eval as in your original code
    except (SyntaxError, NameError, Exception) as e:
      # Optional: log or print errors for lines that fail to parse
      # print(f"Warning: Skipping line due to error: {e} - Line: {l[:100]}...")
      continue # Skip malformed lines

# Function to construct the graph dictionary
def construct_graph(path, required_asin_set):
  """
  Constructs a graph dictionary from product metadata.

  Args:
    path (str): The path to the gzipped metadata file.
    required_asin_set (set): A set of ASINs that must be included as nodes
                              in the graph. Edges will also be filtered to
                              only connect nodes within this set.

  Returns:
    dict: A dictionary representing the graph. Keys are ASINs from
          required_asin_set. Values are dictionaries containing lists
          of related ASINs ('also_bought', 'also_viewed', 'bought_together')
          that are also present in required_asin_set.
          Example:
          {
              'ASIN1': {
                  'also_bought': ['ASIN_B1', 'ASIN_B2'], # ASIN_B1, ASIN_B2 are in required_asin_set
                  'also_viewed': ['ASIN_V1'],          # ASIN_V1 is in required_asin_set
                  'bought_together': []
              },
              'ASIN2': { ... }
          }
  """
  graph = {}
  print(f"Starting graph construction from: {path}")
  print(f"Filtering for {len(required_asin_set)} required ASINs.")

  # Iterate through the file using the parse function and tqdm for progress
  for product_data in tqdm.tqdm(parse(path), desc="Processing metadata"):

    # Safely get the ASIN for the current product
    current_asin = product_data.get('asin')

    # Only process if the ASIN is valid and in our required set
    if current_asin and current_asin in required_asin_set:

      # Ensure the node exists in the graph dictionary
      if current_asin not in graph:
           graph[current_asin] = {
              'also_bought': [],
              'also_viewed': [],
              'bought_together': []
          }

      # Safely get the 'related' dictionary (default to empty if missing)
      related_data = product_data.get('related', {})

      # --- Filter 'also_bought' ---
      also_bought_list = related_data.get('also_bought', [])
      # Keep only ASINs that are in our required set
      filtered_also_bought = [
          asin for asin in also_bought_list if asin in required_asin_set
      ]
      graph[current_asin]['also_bought'] = filtered_also_bought

      # --- Filter 'also_viewed' ---
      also_viewed_list = related_data.get('also_viewed', [])
      # Keep only ASINs that are in our required set
      filtered_also_viewed = [
          asin for asin in also_viewed_list if asin in required_asin_set
      ]
      graph[current_asin]['also_viewed'] = filtered_also_viewed

      # --- Filter 'bought_together' ---
      # Note: 'bought_together' often has only one item, but treat as a list
      bought_together_list = related_data.get('bought_together', [])
      # Keep only ASINs that are in our required set
      filtered_bought_together = [
          asin for asin in bought_together_list if asin in required_asin_set
      ]
      graph[current_asin]['bought_together'] = filtered_bought_together

      # Optional: Clean up nodes that ended up with no connections
      # after filtering, though usually you want to keep all nodes
      # from the required_asin_set.
      if not graph[current_asin]['also_bought'] and \
         not graph[current_asin]['also_viewed'] and \
         not graph[current_asin]['bought_together']:
          # Decide if you want to remove isolated nodes from the required set
          # del graph[current_asin] # Uncomment to remove
          pass # Keep isolated nodes by default

  print(f"Graph construction complete. Graph contains {len(graph)} nodes.")
  return graph

# --- Example Usage ---

# 2. Specify the correct path to your data file
file_path = 'dataset/metadata.json.gz' # Make sure this path is correct

# 3. Call the function to construct the graph
# Add error handling in case the file doesn't exist
try:
    my_graph = construct_graph(file_path, asin_set)

    # 4. Now 'my_graph' holds the desired dictionary. You can inspect it:
    print(f"\nGraph constructed with {len(my_graph)} nodes.")

    # Print details for a specific node if it exists in the graph
    example_asin = 'B000034DMT'
    if example_asin in my_graph:
        print(f"\nDetails for node '{example_asin}':")
        import pprint
        pprint.pprint(my_graph[example_asin])
    else:
        # This might happen if the example ASIN wasn't in the asin_set
        # or wasn't found in the file
        print(f"\nNode '{example_asin}' not found in the constructed graph.")

    # Example of checking another node
    example_asin_2 = 'B000AYEIWW'
    if example_asin_2 in my_graph:
        print(f"\nDetails for node '{example_asin_2}':")
        import pprint
        pprint.pprint(my_graph[example_asin_2])


except FileNotFoundError:
    print(f"\nError: The file was not found at '{file_path}'. Please check the path.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
