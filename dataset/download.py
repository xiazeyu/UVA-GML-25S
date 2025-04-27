"""
https://snap.stanford.edu/data/amazon0302.html
https://snap.stanford.edu/data/amazon0312.html
https://snap.stanford.edu/data/amazon0505.html
https://snap.stanford.edu/data/amazon0601.html
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
https://snap.stanford.edu/data/amazon-meta.html
"""

import requests
from tqdm import tqdm
import gzip
import shutil
import os

def download_file(url, destination):
    try:
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(destination, 'wb') as file, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=destination
        ) as progress_bar:
            for data in response.iter_content(block_size):
                file.write(data)
                progress_bar.update(len(data))
        
        print(f"Downloaded to {destination}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

def decompress_file(source, destination):
    try:
        print(f"Decompressing {source}...")
        with gzip.open(source, 'rb') as f_in:
            with open(destination, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Decompressed to {destination}")
    except Exception as e:
        print(f"Error decompressing {source}: {e}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    urls = [
        "https://snap.stanford.edu/data/amazon0302.txt.gz",
        "https://snap.stanford.edu/data/amazon0312.txt.gz",
        "https://snap.stanford.edu/data/amazon0505.txt.gz",
        "https://snap.stanford.edu/data/amazon0601.txt.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz",
        "https://snap.stanford.edu/data/bigdata/amazon/amazon-meta.txt.gz"
    ]

    for url in urls:
        filename = url.split("/")[-1]
        destination = os.path.join(script_dir, filename)
        download_file(url, destination)
        
        # Decompress the file
        if filename.endswith('.gz'):
            decompressed_filename = filename[:-3]  # Remove .gz extension
            decompressed_destination = os.path.join(script_dir, decompressed_filename)
            decompress_file(destination, decompressed_destination)
            os.remove(destination)  # Remove the compressed file if decompression is successful

if __name__ == "__main__":
    main()
