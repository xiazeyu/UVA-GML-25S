import torch
import gzip
import tqdm
import pandas as pd
import pprint # For pretty printing example output
import os # Added for directory creation

# --- Step 1: Load ASINs from embedding files ---
# This part assumes you have the torch files and path structure as before.
# If asin_set is already loaded in your environment, you can skip this.
asin_set = set()
categories_to_load = [
    'Digital_Music',
    'Video_Games',
    'Pet_Supplies',
]
print("Loading required ASINs...")
for category in categories_to_load:
    file_path_embeddings = f'dataset/{category}_embeddings.pt' # Make sure this path is correct
    try:
        # Ensure you have torch installed: pip install torch
        data = torch.load(file_path_embeddings)
        category_asins = set(data.keys())
        asin_set.update(category_asins) # Use update for set union
        print(f'  Loaded {len(category_asins)} ASINs from {category}.')
    except FileNotFoundError:
        print(f"  Warning: Embedding file not found at '{file_path_embeddings}'. Skipping this category.")
    except ImportError:
        print("  Error: PyTorch library not found. Please install it (`pip install torch`) to load .pt files.")
        # Handle error - perhaps exit or load ASINs differently
        asin_set = set() # Ensure asin_set is empty if loading fails
        break # Stop trying to load more files
    except Exception as e:
        print(f"  Warning: Error loading file '{file_path_embeddings}': {e}. Skipping this category.")

print(f"Total unique required ASINs loaded: {len(asin_set)}")
if not asin_set:
    print("\nError: No ASINs were loaded. Cannot build the DataFrame. Please check embedding file paths or install PyTorch.")
    # Exit or handle this error appropriately in a real script
    # exit() # Or raise an exception


# --- Step 2: Define the parsing function ---
def parse(path):
  """Generator to parse lines from a gzipped file using eval."""
  try:
      with gzip.open(path, 'rb') as g:
          for l in g:
              try:
                  # Using eval as per original request.
                  # Consider json.loads(l) if the file contains valid JSON per line for safety.
                  yield eval(l)
              except (SyntaxError, NameError, ValueError, Exception) as e:
                  # Optional: log or print errors for lines that fail to parse
                  # print(f"Warning: Skipping line due to eval error: {e} - Line: {l[:100]}...")
                  continue # Skip malformed lines
  except FileNotFoundError:
      print(f"\nError: Metadata file not found at '{path}'. Cannot parse.")
      raise # Re-raise the exception to be caught by the main try-except block
  except Exception as e:
      print(f"\nError opening or reading gzip file '{path}': {e}")
      raise # Re-raise


# --- Step 3: Define the DataFrame construction function ---
def create_metadata_dataframe(path, required_asin_set):
    """
    Constructs a Pandas DataFrame containing metadata for specified ASINs.

    Args:
      path (str): The path to the gzipped metadata file.
      required_asin_set (set): A set of ASINs whose metadata should be included.

    Returns:
      pandas.DataFrame: A DataFrame with columns 'asin', 'title', 'price',
                        'salesRank', 'brand', 'categories'. Returns an empty
                        DataFrame if the input set is empty or file cannot be read.
    """
    if not required_asin_set:
        print("Warning: The required_asin_set is empty. Returning an empty DataFrame.")
        return pd.DataFrame(columns=['asin', 'title', 'price', 'salesRank', 'brand', 'categories'])

    data_list = [] # List to hold dictionaries, each representing a row
    print(f"\nStarting DataFrame construction from: {path}")
    print(f"Filtering for {len(required_asin_set)} required ASINs.")

    try:
        product_iterator = parse(path)
        for product_data in tqdm.tqdm(product_iterator, desc="Processing metadata for DataFrame"):
            current_asin = product_data.get('asin')

            # Only process if the ASIN is valid and in our required set
            if current_asin and current_asin in required_asin_set:
                # Extract required fields with defaults
                title = product_data.get('title', None)
                price = product_data.get('price', None)
                # Ensure price is float if found
                if price is not None:
                    try:
                        price = float(price)
                    except (ValueError, TypeError):
                        price = None # Set to None if conversion fails

                sales_rank = product_data.get('salesRank', {}) # Default to empty dict
                brand = product_data.get('brand', None)
                categories = product_data.get('categories', []) # Default to empty list

                # Append data as a dictionary to the list
                data_list.append({
                    'asin': current_asin,
                    'title': title,
                    'price': price,
                    'salesRank': sales_rank, # Note: Storing dicts/lists might require specific handling in Parquet
                    'brand': brand,
                    'categories': categories # Note: Storing dicts/lists might require specific handling in Parquet
                })

    except StopIteration:
        print("Warning: Parsing stopped unexpectedly.")
    except FileNotFoundError:
        # Error message already printed by parse()
        print("Returning an empty DataFrame due to file not found.")
        return pd.DataFrame(columns=['asin', 'title', 'price', 'salesRank', 'brand', 'categories'])
    except Exception as e:
        print(f"An error occurred during DataFrame construction: {e}")
        print("Returning potentially incomplete or empty DataFrame.")
        # Fall through to create DataFrame from potentially partial data_list

    # Convert the list of dictionaries to a DataFrame
    if not data_list:
        print("No matching ASINs found in the metadata file. Returning an empty DataFrame.")
        return pd.DataFrame(columns=['asin', 'title', 'price', 'salesRank', 'brand', 'categories'])
    else:
        df = pd.DataFrame(data_list)
        # Convert complex types to string for better Parquet compatibility if needed
        # df['salesRank'] = df['salesRank'].astype(str)
        # df['categories'] = df['categories'].astype(str)
        print(f"\nDataFrame construction complete. DataFrame has {len(df)} rows.")
        return df

# --- Step 4: Execute DataFrame Construction ---

# Specify the correct path to your metadata file
metadata_file_path = 'dataset/metadata.json.gz' # Make sure this path is correct

# Call the function to construct the DataFrame
# Ensure pandas is installed: pip install pandas
metadata_df = pd.DataFrame() # Initialize empty DataFrame
try:
    # Only proceed if asin_set was successfully populated
    if asin_set:
        metadata_df = create_metadata_dataframe(metadata_file_path, asin_set)
    else:
        print("Skipping DataFrame construction as the required ASIN set is empty.")

    # --- Step 5: Inspect the Result ---
    if not metadata_df.empty:
        print(f"\nDataFrame created successfully with {len(metadata_df)} rows and {len(metadata_df.columns)} columns.")

        # Display the first few rows
        print("\nDataFrame Head:")
        print(metadata_df.head())

        # Display DataFrame info (columns, data types, non-null counts)
        print("\nDataFrame Info:")
        metadata_df.info()

        # Example: Access data for a specific ASIN if it exists
        example_asin_check = 'B000034DMT' # Example ASIN
        if example_asin_check in metadata_df['asin'].values:
            print(f"\nData for ASIN '{example_asin_check}':")
            # Use .loc for label-based indexing if 'asin' becomes the index,
            # or boolean indexing if 'asin' remains a column.
            # Assuming 'asin' is a column:
            print(metadata_df[metadata_df['asin'] == example_asin_check].iloc[0])
        else:
             print(f"\nASIN '{example_asin_check}' not found in the DataFrame.")

    elif asin_set: # If asin_set was not empty, but DataFrame is, it indicates an issue
        print("\nDataFrame construction resulted in an empty DataFrame, possibly due to no matching ASINs found or an error.")
    # If asin_set was empty, the earlier message covers it.


    # --- Step 6: Save DataFrame to Parquet ---
    if not metadata_df.empty:
        output_directory = 'dataset'
        output_filename = 'metadata_node.parquet'
        output_path = os.path.join(output_directory, output_filename)

        print(f"\nAttempting to save DataFrame to Parquet file: {output_path}")

        try:
            # Ensure the output directory exists
            os.makedirs(output_directory, exist_ok=True)
            print(f"Directory '{output_directory}' ensured.")

            # Save the DataFrame to Parquet format
            # index=False prevents pandas from writing the DataFrame index as a column
            metadata_df.to_parquet(output_path, index=False, engine='pyarrow') # Or engine='fastparquet'
            print(f"DataFrame successfully saved to {output_path}")

        except ImportError:
            print("\nError: Could not save to Parquet.")
            print("Please install 'pyarrow' or 'fastparquet' engine.")
            print("Suggestion: pip install pyarrow")
        except Exception as e:
            print(f"\nError: Failed to save DataFrame to {output_path}")
            print(f"An error occurred: {e}")
    else:
        print("\nSkipping saving DataFrame as it is empty.")


except ImportError as e:
    # Catch ImportError for pandas or torch if they occurred earlier
     if 'pandas' in str(e):
         print("\nError: Pandas library not found. Please install it (`pip install pandas`)")
     elif 'torch' in str(e):
         # Message already printed in Step 1, but can add more info here if needed
         pass
     else:
        print(f"\nAn import error occurred: {e}")

except Exception as e:
    print(f"\nAn unexpected error occurred during the overall process: {e}")

