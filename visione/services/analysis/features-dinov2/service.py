import argparse
import functools
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import flask
import PIL.Image
import requests
import torch
import torchvision.transforms as T
from werkzeug.utils import secure_filename

# Configuration
APP_NAME = "dinov2-feature-extractor"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DINOV2_MODEL_NAME = "dinov2_vitb14"  # Smallest DINOv2 ViT model
REQUEST_TIMEOUT_SECONDS = 30  # Timeout for downloading images from URLs
CACHE_SIZE_IMAGE = int(os.environ.get("CACHE_SIZE_IMAGE", 256)) # LRU Cache size for image embeddings

# Image preprocessing parameters
IMAGE_PREPROCESS_RESIZE = 256
IMAGE_PREPROCESS_CROP_SIZE = 224
IMAGE_PREPROCESS_INTERPOLATION = T.InterpolationMode.BICUBIC
IMAGE_PREPROCESS_MEAN = (0.485, 0.456, 0.406) # ImageNet mean
IMAGE_PREPROCESS_STD = (0.229, 0.224, 0.225)  # ImageNet std

# Logging setup
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(APP_NAME)


class DINOv2FeatureExtractor:
    """Encapsulates DINOv2 model loading and feature extraction logic."""

    def __init__(self, model_name: str = DEFAULT_DINOV2_MODEL_NAME, device_str: str = DEFAULT_DEVICE) -> None:
        self.device = torch.device(device_str)
        self.model_name = model_name
        logger.info(f"Initializing DINOv2FeatureExtractor with model '{self.model_name}' on device: {self.device}")

        try:
            # Load DINOv2 model from PyTorch Hub
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Ensure model is in evaluation mode
            logger.info(f"DINOv2 model '{self.model_name}' loaded successfully and set to eval mode on {self.device}.")

            # Define image preprocessing transform
            self.preprocess = T.Compose([
                T.Resize(IMAGE_PREPROCESS_RESIZE, interpolation=IMAGE_PREPROCESS_INTERPOLATION),
                T.CenterCrop(IMAGE_PREPROCESS_CROP_SIZE),
                T.ToTensor(),
                T.Normalize(mean=IMAGE_PREPROCESS_MEAN, std=IMAGE_PREPROCESS_STD),
            ])
            logger.info("DINOv2 image preprocessing pipeline initialized.")

        except Exception as e:
            logger.error(f"Error during DINOv2 model loading ('{self.model_name}'): {e}", exc_info=True)
            raise RuntimeError(f"Failed to load DINOv2 model '{self.model_name}': {e}") from e

    def _load_and_preprocess_image(self, image_source: Union[Path, str, PIL.Image.Image]) -> torch.Tensor:
        """Loads an image from various sources and preprocesses it."""
        logger.debug(f"Loading and preprocessing image from source type: {type(image_source)}")
        try:
            if isinstance(image_source, PIL.Image.Image):
                image = image_source.convert("RGB") # Ensure 3 channels
            elif isinstance(image_source, (str, Path)):
                image = PIL.Image.open(image_source).convert("RGB")
            else:
                raise TypeError(f"Unsupported image_source type: {type(image_source)}")
            
            preprocessed_image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            logger.debug(f"Image preprocessed to tensor shape: {preprocessed_image_tensor.shape}")
            return preprocessed_image_tensor
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_source}")
            raise
        except PIL.UnidentifiedImageError:
            logger.error(f"Cannot identify image file (corrupted or unsupported format): {image_source}")
            raise
        except Exception as e:
            logger.error(f"Error loading/preprocessing image {image_source}: {e}", exc_info=True)
            raise RuntimeError(f"Image loading/preprocessing failed: {e}") from e

    @functools.lru_cache(maxsize=CACHE_SIZE_IMAGE)
    def get_image_embedding(self, image_identifier: str, image_loader_func: Callable[[], PIL.Image.Image]) -> List[float]:
        """
        Extracts embedding for an image, using a loader function and caching by identifier.
        The image_identifier should uniquely represent the image (e.g., URL or stable file path).
        The image_loader_func is a callable that loads and returns a PIL.Image.Image object.
        """
        logger.info(f"Request to embed image identified by: '{image_identifier}' (Cache info: {self.get_image_embedding.cache_info()})")
        try:
            pil_image = image_loader_func()  # Execute the loader to get PIL image
            if not isinstance(pil_image, PIL.Image.Image):
                 raise ValueError(f"Image loader for '{image_identifier}' did not return a PIL.Image object.")

            preprocessed_image_tensor = self._load_and_preprocess_image(pil_image)
            
            with torch.no_grad():
                features_dict = self.model(preprocessed_image_tensor)
                # DINOv2 typically provides class token ('x_norm_clstoken') or patch tokens.
                # Using class token as the global image descriptor.
                image_features = features_dict.get('x_norm_clstoken') 
                if image_features is None:
                    logger.error(f"Key 'x_norm_clstoken' not found in DINOv2 model output for {image_identifier}. Available keys: {list(features_dict.keys())}")
                    raise RuntimeError("Could not extract class token feature from DINOv2 model.")

            logger.debug(f"Image features extracted for '{image_identifier}' with shape: {image_features.shape}")
            return image_features.squeeze().cpu().tolist() # Squeeze to remove batch dim, convert to list
        
        except Exception as e:
            logger.error(f"Error extracting image embedding for '{image_identifier}': {e}", exc_info=True)
            # Attempt to clear cache for this specific key if it caused an error,
            # though lru_cache doesn't directly support targeted eviction.
            # A full clear might be too aggressive. For now, rely on cache expiry or size limit.
            # If this identifier is known to be problematic, manual cache intervention might be needed
            # or a more sophisticated caching mechanism.
            raise RuntimeError(f"Image embedding extraction failed for '{image_identifier}': {e}") from e


# Global extractor instance
dinov2_extractor: Optional[DINOv2FeatureExtractor] = None

# Globals to store startup configuration for the extractor
_startup_model_name_dinov2: str = DEFAULT_DINOV2_MODEL_NAME
_startup_device_dinov2: str = DEFAULT_DEVICE

def get_extractor(model_name: str = DEFAULT_DINOV2_MODEL_NAME, device: str = DEFAULT_DEVICE) -> DINOv2FeatureExtractor:
    """Initializes and returns the global DINOv2 feature extractor instance.
       Re-initializes if model_name or device changes.
    """
    global dinov2_extractor, _startup_model_name_dinov2, _startup_device_dinov2

    # Update startup globals if different from current call, relevant for direct calls outside Flask context
    _startup_model_name_dinov2 = model_name
    _startup_device_dinov2 = device
    
    if dinov2_extractor is None or \
       dinov2_extractor.model_name != model_name or \
       str(dinov2_extractor.device) != device:
        logger.info(
            f"DINOv2 extractor: Instance not found or configuration changed. "
            f"(Re)initializing with model '{model_name}' on device '{device}'..."
        )
        try:
            dinov2_extractor = DINOv2FeatureExtractor(model_name=model_name, device_str=device)
        except RuntimeError as e:
            logger.critical(f"DINOv2 extractor: Failed to initialize: {e}", exc_info=True)
            dinov2_extractor = None # Ensure it's None on failure
            raise # Re-raise to signal critical failure
    
    if dinov2_extractor is None: # Should be caught by re-raise, but as a safeguard
        raise RuntimeError("DINOv2 extractor could not be initialized and is None.")
        
    return dinov2_extractor


# Flask application
app = flask.Flask(APP_NAME)

@app.before_first_request
def initialize_model_before_first_request() -> None:
    """Ensure the model is loaded before the first request, using startup settings."""
    logger.info("Flask: Attempting to pre-load DINOv2 model before first request...")
    try:
        # Use the globals that would have been set by __main__ or defaults
        get_extractor(model_name=_startup_model_name_dinov2, device=_startup_device_dinov2)
        logger.info(f"Flask: DINOv2 model '{_startup_model_name_dinov2}' pre-loading complete on '{_startup_device_dinov2}'.")
    except Exception as e:
        logger.critical(f"Flask: DINOv2 model could not be pre-loaded. Service might be impaired. Error: {e}", exc_info=True)
        # Application will still start; requests requiring the model will likely fail.

@app.route("/ping", methods=["GET"])
def ping() -> flask.Response:
    """Health check endpoint."""
    return flask.jsonify({"status": "ok", "message": "pong from DINOv2 extractor"})


@app.route("/get-image-feature", methods=["POST"])
def get_image_feature_endpoint() -> flask.Response:
    """Extracts features from an input image (URL or uploaded file)."""
    try:
        # Use the globally configured (and potentially pre-loaded) extractor
        extractor = get_extractor(model_name=_startup_model_name_dinov2, device=_startup_device_dinov2)
    except RuntimeError as e: 
        logger.error(f"DINOv2 image feature extraction unavailable due to model init failure: {e}")
        return flask.jsonify({"error": f"Service unavailable: DINOv2 model not initialized ({str(e)})"}), 503

    temp_file_to_delete: Optional[str] = None
    image_identifier: Optional[str] = None
    source_description: str = "unknown"
    
    try:
        if "image_url" in flask.request.form:
            image_url = flask.request.form["image_url"]
            if not isinstance(image_url, str) or not image_url.strip():
                 return flask.jsonify({"error": "Invalid or empty 'image_url' in form data"}), 400
            
            image_identifier = image_url # Use URL as cache key
            source_description = f"url: {image_url}"
            logger.info(f"Processing /get-image-feature for {source_description}")
            
            def load_from_url_closure() -> PIL.Image.Image:
                # This closure will be called by the cached function if not in cache
                nonlocal temp_file_to_delete # Allows modification of outer scope var
                try:
                    response = requests.get(image_url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS)
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    
                    # PIL can often read from a BytesIO stream directly
                    # However, saving to temp file can be more robust for various formats/libraries
                    # and provides a stable path if needed by other parts of a system.
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_f: # Suffix helps some libs
                        temp_file_to_delete = tmp_f.name
                    
                    with open(temp_file_to_delete, "wb") as f_out:
                        for chunk in response.iter_content(chunk_size=8192): # 8KB chunks
                            f_out.write(chunk)
                    logger.debug(f"Image from URL {image_url} downloaded to {temp_file_to_delete}")
                    return PIL.Image.open(temp_file_to_delete) # PIL.Image.open handles RGB conversion if needed later
                except requests.exceptions.RequestException as req_e:
                    logger.error(f"Error downloading image from URL {image_url}: {req_e}", exc_info=True)
                    raise RuntimeError(f"Failed to download image from URL '{image_url}': {str(req_e)}") from req_e
                except PIL.UnidentifiedImageError as pil_e:
                    logger.error(f"Cannot identify image from URL {image_url} (downloaded to {temp_file_to_delete}): {pil_e}")
                    # Clean up temp file if download was successful but image is bad
                    if temp_file_to_delete and os.path.exists(temp_file_to_delete):
                        os.unlink(temp_file_to_delete)
                        temp_file_to_delete = None # Avoid double deletion
                    raise RuntimeError(f"Invalid image format from URL '{image_url}': {str(pil_e)}") from pil_e
            
            image_embedding = extractor.get_image_embedding(image_identifier, load_from_url_closure)

        elif "file" in flask.request.files:
            image_file = flask.request.files["file"]
            if not image_file or not image_file.filename: # Check if filename is present and not empty
                return flask.jsonify({"error": "No selected file or empty filename in 'file' part"}), 400
            
            filename = secure_filename(image_file.filename)
            source_description = f"file: {filename}"
            logger.info(f"Processing /get-image-feature for {source_description}")
            
            # For uploaded files, using a temporary file path as cache identifier is simple
            # but won't cache identical content if uploaded multiple times (new temp path).
            # A content hash would be more robust for caching identical uploads.
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_f:
                temp_file_to_delete = tmp_f.name # Mark for cleanup
            image_file.seek(0) # Ensure stream is at the beginning
            image_file.save(temp_file_to_delete)
            image_identifier = temp_file_to_delete # Use temp file path as cache key

            def load_from_temp_path_closure() -> PIL.Image.Image:
                try:
                    return PIL.Image.open(temp_file_to_delete)
                except PIL.UnidentifiedImageError as pil_e:
                    logger.error(f"Cannot identify uploaded image file {filename} (saved to {temp_file_to_delete}): {pil_e}")
                    raise RuntimeError(f"Invalid uploaded image format for file '{filename}': {str(pil_e)}") from pil_e
            
            image_embedding = extractor.get_image_embedding(image_identifier, load_from_temp_path_closure)
        else:
            return flask.jsonify({"error": "Missing 'image_url' (form data) or 'file' (multipart) part in request"}), 400

        return flask.jsonify({
            "source_description": source_description,
            "embedding": image_embedding
        })

    except ValueError as ve: # E.g., from image_loader not returning PIL.Image
        logger.warning(f"ValueError during image feature extraction for '{source_description}': {ve}")
        return flask.jsonify({"error": str(ve)}), 400
    except RuntimeError as rte: # Catch errors from embedding extraction or image loading/downloading
        logger.error(f"Runtime error in /get-image-feature for '{source_description}': {rte}", exc_info=True)
        return flask.jsonify({"error": str(rte)}), 500
    except Exception as e: # Catch-all for unexpected errors
        logger.error(f"Unexpected error in /get-image-feature for '{source_description}': {e}", exc_info=True)
        return flask.jsonify({"error": "Failed to extract image features due to an unexpected server error"}), 500
    finally:
        # Cleanup: Ensure temporary file is deleted if one was created
        if temp_file_to_delete and os.path.exists(temp_file_to_delete):
            try:
                os.unlink(temp_file_to_delete)
                logger.debug(f"Temporary file {temp_file_to_delete} deleted successfully.")
            except Exception as e_del:
                logger.error(f"Error deleting temporary file {temp_file_to_delete}: {e_del}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv2 Feature Extractor Service")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the service to."
    )
    parser.add_argument(
        "--port", type=int, default=os.environ.get("FLASK_RUN_PORT", 8080), help="Port to bind the service to."
    )
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE, help=f"Device to run model on (e.g., 'cuda', 'cpu'). Default: {DEFAULT_DEVICE}"
    )
    parser.add_argument(
        "--model-name", type=str, default=DEFAULT_DINOV2_MODEL_NAME, 
        help=f"DINOv2 model name from torch.hub (e.g., 'dinov2_vitb14', 'dinov2_vitl14'). Default: {DEFAULT_DINOV2_MODEL_NAME}"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run Flask in debug mode (disables model pre-loading in __main__)."
    )
    
    cli_args = parser.parse_args()

    # Update globals that before_first_request will use for pre-loading
    _startup_model_name_dinov2 = cli_args.model_name
    _startup_device_dinov2 = cli_args.device

    if not cli_args.debug: 
        # Pre-load model if not in debug mode (Flask reloader can cause issues with multiple loads)
        try:
            logger.info(f"Pre-loading DINOv2 model '{cli_args.model_name}' on device: {cli_args.device} from __main__...")
            get_extractor(model_name=cli_args.model_name, device=cli_args.device)
            logger.info("DINOv2 model pre-loaded successfully via __main__.")
        except Exception as e:
            logger.critical(
                f"Failed to pre-load DINOv2 model during startup in __main__. "
                f"The service might not function correctly. Error: {e}", exc_info=True
            )
            # Depending on operational requirements, one might choose to exit(1) here.
            # For now, let Flask start; requests will fail if the model is unusable.

    app.run(host=cli_args.host, port=cli_args.port, debug=cli_args.debug)
