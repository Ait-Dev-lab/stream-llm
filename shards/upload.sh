#!/bin/bash
# shards/upload.sh
# Upload generated .bin shards to Cloudflare R2 (free 10GB)
# Run this after split_shards.py finishes

# Install rclone if not present
# curl https://rclone.org/install.sh | sudo bash

# Configure R2 endpoint
export RCLONE_CONFIG_R2_TYPE=s3
export RCLONE_CONFIG_R2_ACCESS_KEY_ID=your_access_key
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=your_secret_key
export RCLONE_CONFIG_R2_ENDPOINT=https://your-account.r2.cloudflarestorage.com

echo "Uploading shards to R2 bucket: stream-llm-shards..."

for file in /content/shards/shard_*.bin; do
    echo "Uploading $(basename $file)..."
    rclone copy "$file" r2:stream-llm-shards/ --progress
done

echo "Uploading model_config.json..."
rclone copy /content/shards/model_config.json r2:stream-llm-shards/

echo "\nDone! Files available at:"
echo "https://your-cdn.com/shards/"
