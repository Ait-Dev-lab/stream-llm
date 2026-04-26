# -*- coding: utf-8 -*-
"""colab/split_shards.ipynb
Run in Colab with T4 GPU. Converts GGUF -> per-layer .bin shards.
"""

!pip install gguf numpy > /dev/null 2>&1

import numpy as np
import os
import json
import hashlib
from gguf import GGUFReader

# ==========================================
# CONFIG
# ==========================================
MODEL_URL = "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
OUTPUT_DIR = "/content/shards/"
SHARD_SIZE_MB = 100  # Target size per shard
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# DOWNLOAD MODEL (if not already in Drive)
# ==========================================
print("Downloading model...")
!wget -q -O /content/model.gguf {MODEL_URL}
print(f"Model downloaded: {os.path.getsize('/content/model.gguf') / 1e9:.2f} GB")

# ==========================================
# READ GGUF
# ==========================================
reader = GGUFReader("/content/model.gguf")

# Build shard index
shards = []
current_shard = {"layers": [], "size_bytes": 0}
shard_id = 0

for tensor in reader.tensors:
    tensor_name = tensor.name
    tensor_size = tensor.nbytes
    
    # Determine layer from name
    if "blk." in tensor_name:
        layer_num = int(tensor_name.split("blk.")[1].split(".")[0])
    elif "token_embd" in tensor_name or "output" in tensor_name:
        layer_num = -1  # Embedding / output
    else:
        layer_num = -2  # Other (norm, etc.)
    
    if current_shard["size_bytes"] + tensor_size > SHARD_SIZE_MB * 1e6 and current_shard["layers"]:
        # Close current shard
        shards.append(current_shard)
        current_shard = {"layers": [], "size_bytes": 0}
        shard_id += 1
    
    current_shard["layers"].append({
        "name": tensor_name,
        "layer": layer_num,
        "nbytes": tensor_size,
        "dtype": str(tensor.tensor_type),
        "shape": list(tensor.shape)
    })
    current_shard["size_bytes"] += tensor_size

if current_shard["layers"]:
    shards.append(current_shard)

print(f"Total shards: {len(shards)}")

# ==========================================
# SPLIT AND SAVE
# ==========================================
shard_manifest = {"total_layers": 28, "shards": []}

for i, shard in enumerate(shards):
    shard_data = b""
    for t in shard["layers"]:
        tensor_data = reader.get_tensor(t["name"])
        shard_data += tensor_data.tobytes()
    
    filename = f"shard_{i}.bin"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, "wb") as f:
        f.write(shard_data)
    
    checksum = hashlib.sha256(shard_data).hexdigest()
    
    shard_manifest["shards"].append({
        "id": i,
        "filename": filename,
        "layers": [l["name"] for l in shard["layers"]],
        "size_bytes": len(shard_data),
        "checksum": f"sha256:{checksum}"
    })
    
    print(f"Shard {i}: {filename} ({len(shard_data)/1e6:.1f} MB)")

# Save manifest
with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
    json.dump(shard_manifest, f, indent=2)

print("\nDone! All shards in", OUTPUT_DIR)
print(f"Total shards: {len(shards)}")
print(f"Manifest: {os.path.join(OUTPUT_DIR, 'model_config.json')}")

# ==========================================
# UPLOAD INSTRUCTIONS
# ==========================================
print("\n" + "="*60)
print("NEXT: Upload shards to your CDN")
print("="*60)
print("""
Option 1: Cloudflare R2 (10GB free)
Option 2: Backblaze B2 (10GB free) 
Option 3: GitHub Releases (free, but slower)
Option 4: Hugging Face Spaces (free, static hosting)
""")
