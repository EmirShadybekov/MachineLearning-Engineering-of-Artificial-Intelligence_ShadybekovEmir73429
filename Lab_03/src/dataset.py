"""
Fashion-MNIST dataset loader for Lab 3.
Transforms: ToTensor(), Normalize((0.2860,), (0.3530,))
Splits: 60k train -> 55k train, 5k validation. 10k test.
Batch size: 128.
"""

import os
import urllib.request
import gzip
import shutil
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def download_fashion_mnist_files(raw_dir: str):
    """Ensure raw FashionMNIST files are downloaded reliably with User-Agent header."""
    os.makedirs(raw_dir, exist_ok=True)
    
    base_urls = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
    ]
    
    resources = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    for filename in resources:
        dest_path = os.path.join(raw_dir, filename)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            continue
            
        print(f"Downloading {filename}...")
        downloaded = False
        for base_url in base_urls:
            url = base_url + filename
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=20) as response, open(dest_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                if os.path.getsize(dest_path) > 1000:
                    downloaded = True
                    break
            except Exception as e:
                print(f"Failed from {url}: {e}, trying next mirror...")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                    
        if not downloaded:
            raise RuntimeError(f"Could not download {filename} from any mirror.")

    # Extract all .gz files if not already extracted
    for filename in resources:
        dest_path = os.path.join(raw_dir, filename)
        unzipped_path = os.path.join(raw_dir, filename[:-3])
        if not os.path.exists(unzipped_path):
            print(f"Extracting {filename} -> {unzipped_path}...")
            with gzip.open(dest_path, 'rb') as f_in, open(unzipped_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def get_dataloaders(root_data_dir: str = None, batch_size: int = 128, seed: int = 42):
    if root_data_dir is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        root_data_dir = os.path.abspath(os.path.join(curr_dir, "..", "..", "data"))
        
    os.makedirs(root_data_dir, exist_ok=True)
    raw_dir = os.path.join(root_data_dir, "FashionMNIST", "raw")
    download_fashion_mnist_files(raw_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    full_train = datasets.FashionMNIST(root=root_data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.FashionMNIST(root=root_data_dir, train=False, download=False, transform=transform)

    # Split official 60k into 55k train and 5k validation
    gen = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train, [55000, 5000], generator=gen)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
