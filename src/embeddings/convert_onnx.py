import os
from pathlib import Path

from django.conf import settings
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# Configure Django settings to access your project's configuration
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django  # noqa


django.setup()

# --- Configuration ---
# The name of the model you want to convert (should match your settings)
model_name = settings.HUGGINGFACE_EMBEDDING_MODEL_NAME

# The directory where your models are cached
cache_dir = settings.EMBEDDING_MODELS_CACHE_DIR
# --- End Configuration ---
try:
    import onnxruntime  # noqa
except ImportError as e:
    print(f"Error of import onnxruntime: {e}")
    exit()

print(f"Loading model: {model_name}")
model = SentenceTransformer(model_name, cache_folder=str(cache_dir))

# The output path for the ONNX model will be determined by the transformers library,
# but it will be within your cache_dir.
# We need to find the exact path.

model_path = snapshot_download(repo_id=model_name, cache_dir=cache_dir)

output_path = Path(model_path) / "onnx" / "model.onnx"

if output_path.exists():
    print(f"ONNX model already exists at: {output_path}")
else:
    print(f"Converting model to ONNX and saving to: {output_path}")
    # The conversion is implicitly handled by the library when you save it this way,
    # but we are making it explicit.
    # For many models, you might need to use `optimum.onnxruntime` for conversion.
    # A simple way to trigger conversion and caching is to load it with onnx backend once.

    print("Creating a dummy SentenceTransformer with ONNX backend to trigger conversion.")
    try:
        SentenceTransformer(model_name, cache_folder=str(cache_dir), backend="onnx")
        print("Model converted and cached successfully.")
    except Exception as e:
        print(f"An error occurred during ONNX conversion: {e}")
        print("Please ensure you have `optimum[onnxruntime]` installed (`pip install optimum[onnxruntime]`).")
