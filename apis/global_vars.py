import socket 
NODE = socket.gethostname().split(".")[0]

# CLIP API
CLIP_URL = f"http://{NODE}:8090"  # assumes you call the script on the same node
CLIP_CACHE_FILE = f"cache/cache_clip"
