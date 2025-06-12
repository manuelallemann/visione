import argparse
import functools
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import flask
import PIL.Image
import requests
import torch
import torchvision.transforms
from werkzeug.utils import secure_filename

# Attempt to import CLIP from openai-clip or clip package
try:
    import clip
except ImportError:
    # Try alternative import if the primary one fails (e.g. older package name)
    try:
        from clip import clip  # type: ignore[no-redef]
    except ImportError as e:
        logging.critical(f"Failed to import CLIP library. Please ensure 'openai-clip' is installed: {e}")
        clip = None # type: ignore[assignment]


# Configuration
APP_NAME = "clip-feature-extractor"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CLIP_MODEL_NAME = "ViT-B/32"
REQUEST_TIMEOUT_SECONDS = 30 # Timeout for downloading images from URLs
CACHE_SIZE_TEXT = int(os.environ.get("CACHE_SIZE_TEXT", 1024))
CACHE_SIZE_IMAGE = int(os.environ.get("CACHE_SIZE_IMAGE", 256))


# Logging setup
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(APP_NAME)


class CLIPFeatureExtractor:
    """Encapsulates CLIP model loading and feature extraction logic."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL_NAME, device_str: str = DEFAULT_DEVICE) -> None:
        self.device = torch.device(device_str)
        self.model_name = model_name
        logger.info(f"Initializing CLIPFeatureExtractor with model '{self.model_name}' on device: {self.device}")

        if clip is None:
            msg = "CLIP library not imported. Extractor cannot function."
            logger.error(msg)
            raise RuntimeError(msg)

        try:
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval() # Ensure model is in evaluation mode
            logger.info(f"CLIP model '{self.model_name}' loaded and set to evaluation mode on {self.device}.")
        except Exception as e:
            logger.error(f"Error during CLIP model loading: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load CLIP model '{self.model_name}': {e}") from e

    @functools.lru_cache(maxsize=CACHE_SIZE_TEXT)
    def get_text_embedding(self, text: str) -> List[float]:
        """Extracts embedding for a single text string, with caching."""
        logger.debug(f"Request to embed text: '{text[:50]}...'")
        if not text.strip():
            logger.warning("Received empty text for embedding.")
            raise ValueError("Text input cannot be empty.")
        try:
            with torch.no_grad():
                tokenized_text = clip.tokenize([text]).to(self.device)
                text_features = self.model.encode_text(tokenized_text)
                text_features /= text_features.norm(dim=-1, keepdim=True) # Normalize features
            
            logger.debug(f"Text features extracted with shape: {text_features.shape}")
            return text_features.squeeze().cpu().tolist()
        except Exception as e:
            logger.error(f"Error extracting text embedding for '{text[:50]}...': {e}", exc_info=True)
            raise RuntimeError(f"Text embedding extraction failed: {e}") from e

    def _load_and_preprocess_image(self, image_source: Union[Path, str, PIL.Image.Image]) -> torch.Tensor:
        """Loads an image from various sources and preprocesses it."""
        try:
            if isinstance(image_source, PIL.Image.Image):
                image = image_source
            elif isinstance(image_source, (str, Path)):
                image = PIL.Image.open(image_source).convert("RGB")
            else:
                raise TypeError(f"Unsupported image_source type: {type(image_source)}")
            
            preprocessed_image = self.preprocess(image).unsqueeze(0).to(self.device)
            return preprocessed_image
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_source}")
            raise
        except PIL.UnidentifiedImageError:
            logger.error(f"Cannot identify image file: {image_source}")
            raise
        except Exception as e:
            logger.error(f"Error loading/preprocessing image {image_source}: {e}", exc_info=True)
            raise RuntimeError(f"Image loading/preprocessing failed: {e}") from e

    @functools.lru_cache(maxsize=CACHE_SIZE_IMAGE)
    def get_image_embedding(self, image_identifier: str, image_loader_func) -> List[float]:
        """
        Extracts embedding for an image, using a loader function and caching by identifier.
        The image_identifier should uniquely represent the image (e.g., URL or stable file path).
        The image_loader_func is a callable that loads and returns a PIL.Image.Image object.
        """
        logger.debug(f"Request to embed image identified by: '{image_identifier}'")
        try:
            pil_image = image_loader_func() # Execute the loader to get PIL image
            if not isinstance(pil_image, PIL.Image.Image):
                 raise ValueError(f"Image loader for '{image_identifier}' did not return a PIL.Image object.")

            preprocessed_image = self._load_and_preprocess_image(pil_image)
            
            with torch.no_grad():
                image_features = self.model.encode_image(preprocessed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True) # Normalize features

            logger.debug(f"Image features extracted for '{image_identifier}' with shape: {image_features.shape}")
            return image_features.squeeze().cpu().tolist()
        except Exception as e:
            logger.error(f"Error extracting image embedding for '{image_identifier}': {e}", exc_info=True)
            # Avoid caching failures
            if image_identifier in self.get_image_embedding.cache_info(): # type: ignore[attr-defined]
                self.get_image_embedding.cache_clear() # type: ignore[attr-defined]
            raise RuntimeError(f"Image embedding extraction failed for '{image_identifier}': {e}") from e


# Global extractor instance
clip_extractor: Optional[CLIPFeatureExtractor] = None

def get_extractor(model_name: str = DEFAULT_CLIP_MODEL_NAME, device: str = DEFAULT_DEVICE) -> CLIPFeatureExtractor:
    """Initializes and returns the global feature extractor instance."""
    global clip_extractor
    if clip_extractor is None or clip_extractor.model_name != model_name or str(clip_extractor.device) != device:
        logger.info(f"CLIP extractor instance not found or configuration changed, (re)initializing with model {model_name} on {device}...")
        try:
            clip_extractor = CLIPFeatureExtractor(model_name=model_name, device_str=device)
        except RuntimeError as e:
            logger.critical(f"Failed to initialize CLIPFeatureExtractor: {e}", exc_info=True)
            clip_extractor = None # Ensure it stays None on failure
            raise # Re-raise to make it clear at startup if critical
    
    if clip_extractor is None: # Should not be reached if re-raise happens, but as a safeguard
        raise RuntimeError("CLIP extractor could not be initialized.")
        
    return clip_extractor


# Flask application
app = flask.Flask(APP_NAME)

# Determine model and device at startup based on args for pre-loading
# This will be done in the main block
_startup_model_name = DEFAULT_CLIP_MODEL_NAME
_startup_device = DEFAULT_DEVICE

@app.before_first_request
def initialize_model_before_first_request():
    """Ensure the model is loaded before the first request, using startup settings."""
    try:
        logger.info(f"Flask: Attempting to pre-load model '{_startup_model_name}' on device '{_startup_device}' before first request.")
        get_extractor(model_name=_startup_model_name, device=_startup_device)
        logger.info("Flask: CLIP model pre-loading complete via before_first_request.")
    except Exception as e:
        logger.critical(f"Flask: Model could not be pre-loaded via before_first_request. {e}", exc_info=True)
        # The application will still start, but requests will likely fail if the model is needed.

@app.route("/ping", methods=["GET"])
def ping() -> flask.Response:
    """Health check endpoint."""
    return flask.jsonify({"status": "ok", "message": "pong"})


@app.route("/get-text-feature", methods=["POST"])
def get_text_feature_endpoint() -> flask.Response:
    """Extracts feature embedding from input text."""
    try:
        # Use startup config unless dynamic config per request is needed
        extractor = get_extractor(model_name=_startup_model_name, device=_startup_device)
    except RuntimeError as e: 
        logger.error(f"Text feature extraction unavailable: {e}")
        return flask.jsonify({"error": f"Service unavailable: {str(e)}"}), 503

    if not flask.request.is_json:
        return flask.jsonify({"error": "Request must be JSON"}), 400
    
    data = flask.request.get_json()
    if not data or "text" not in data:
        return flask.jsonify({"error": "Missing 'text' field in JSON payload"}), 400
    
    text = data["text"]
    if not isinstance(text, str):
        return flask.jsonify({"error": "'text' field must be a string"}), 400

    try:
        logger.info(f"Processing /get-text-feature for text: '{text[:50]}...'")
        text_embedding = extractor.get_text_embedding(text)
        return flask.jsonify({"text": text, "embedding": text_embedding})
    except ValueError as ve: 
        logger.warning(f"ValueError in text feature extraction: {ve}")
        return flask.jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in /get-text-feature: {e}", exc_info=True)
        return flask.jsonify({"error": "Failed to extract text features"}), 500


@app.route("/get-image-feature", methods=["POST"])
def get_image_feature_endpoint() -> flask.Response:
    """Extracts features from an input image (URL or uploaded file)."""
    try:
        extractor = get_extractor(model_name=_startup_model_name, device=_startup_device)
    except RuntimeError as e: 
        logger.error(f"Image feature extraction unavailable: {e}")
        return flask.jsonify({"error": f"Service unavailable: {str(e)}"}), 503

    temp_file_to_delete: Optional[str] = None
    image_identifier: Optional[str] = None
    
    try:
        pil_image: Optional[PIL.Image.Image] = None

        if "image_url" in flask.request.form:
            image_url = flask.request.form["image_url"]
            if not isinstance(image_url, str) or not image_url.strip():
                 return flask.jsonify({"error": "Invalid 'image_url'"}), 400
            image_identifier = image_url
            logger.info(f"Processing /get-image-feature for URL: {image_url}")
            
            def load_from_url():
                nonlocal temp_file_to_delete # To ensure cleanup
                try:
                    response = requests.get(image_url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS)
                    response.raise_for_status()
                    
                    # Create a non-deleted temporary file to store image
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_f:
                        temp_file_to_delete = tmp_f.name
                    
                    with open(temp_file_to_delete, "wb") as f_out:
                        for chunk in response.iter_content(chunk_size=8192):
                            f_out.write(chunk)
                    logger.debug(f"Image from URL {image_url} downloaded to {temp_file_to_delete}")
                    return PIL.Image.open(temp_file_to_delete).convert("RGB")
                except requests.exceptions.RequestException as req_e:
                    logger.error(f"Error downloading image from URL {image_url}: {req_e}", exc_info=True)
                    raise RuntimeError(f"Failed to download image from URL: {str(req_e)}") from req_e
                except PIL.UnidentifiedImageError as pil_e:
                    logger.error(f"Cannot identify image from URL {image_url} (downloaded to {temp_file_to_delete}): {pil_e}")
                    raise RuntimeError(f"Invalid image format from URL: {str(pil_e)}") from pil_e

            image_embedding = extractor.get_image_embedding(image_identifier, load_from_url)

        elif "file" in flask.request.files:
            image_file = flask.request.files["file"]
            if not image_file or image_file.filename == "":
                return flask.jsonify({"error": "No selected file or empty filename"}), 400
            
            filename = secure_filename(image_file.filename if image_file.filename else "uploaded_image")
            # For uploaded files, the identifier for caching is tricky.
            # Hashing the file content is robust but can be slow.
            # Using filename is not unique. For now, no caching for direct uploads or use a temp identifier.
            image_identifier = f"uploaded:{filename}:{os.urandom(4).hex()}" # Temp identifier for logging
            logger.info(f"Processing /get-image-feature for uploaded file: {filename}")
            
            def load_from_file_storage():
                try:
                    return PIL.Image.open(image_file.stream).convert("RGB")
                except PIL.UnidentifiedImageError as pil_e:
                    logger.error(f"Cannot identify uploaded image file {filename}: {pil_e}")
                    raise RuntimeError(f"Invalid uploaded image format: {str(pil_e)}") from pil_e

            # Caching for file uploads is complex. If we want to cache based on content,
            # we'd need to read the stream, hash it, then use the hash as image_identifier.
            # For simplicity, we can bypass lru_cache for direct uploads or implement hashing.
            # Here, we'll call a non-cached path for direct uploads.
            # A more advanced approach would be to save it to a temp file and use its path as identifier.
            
            # Let's save to a temp file to make it consistent with URL processing for the extractor
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_f:
                temp_file_to_delete = tmp_f.name # Mark for cleanup
            image_file.seek(0) # Reset stream pointer
            image_file.save(temp_file_to_delete)
            image_identifier = temp_file_to_delete # Use temp path as identifier for caching

            def load_from_temp_path():
                 return PIL.Image.open(temp_file_to_delete).convert("RGB")

            image_embedding = extractor.get_image_embedding(image_identifier, load_from_temp_path)

        else:
            return flask.jsonify({"error": "Missing 'image_url' (form data) or 'file' (multipart) part in request"}), 400

        return flask.jsonify({
            "source": flask.request.form.get("image_url") or \
                      (flask.request.files["file"].filename if "file" in flask.request.files else "uploaded_file"),
            "embedding": image_embedding
        })

    except ValueError as ve: 
        logger.warning(f"ValueError in image feature extraction: {ve}")
        return flask.jsonify({"error": str(ve)}), 400
    except RuntimeError as rte: # Catch errors from embedding extraction or image loading
        logger.error(f"Runtime error in /get-image-feature: {rte}", exc_info=True)
        return flask.jsonify({"error": str(rte)}), 500
    except Exception as e:
        logger.error(f"Unexpected error in /get-image-feature: {e}", exc_info=True)
        return flask.jsonify({"error": "Failed to extract image features due to an unexpected error"}), 500
    finally:
        if temp_file_to_delete and os.path.exists(temp_file_to_delete):
            try:
                os.unlink(temp_file_to_delete)
                logger.debug(f"Temporary file {temp_file_to_delete} deleted.")
            except Exception as e_del:
                logger.error(f"Error deleting temporary file {temp_file_to_delete}: {e_del}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Feature Extractor Service")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the service to."
    )
    parser.add_argument(
        "--port", type=int, default=os.environ.get("FLASK_RUN_PORT", 8080), help="Port to bind the service to."
    )
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE, help="Device to run model on (e.g., 'cuda', 'cpu')."
    )
    parser.add_argument(
        "--model-name", type=str, default=DEFAULT_CLIP_MODEL_NAME, help=f"CLIP model name (e.g., {', '.join(clip.available_models() if clip else [])})."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run Flask in debug mode (disables model pre-loading)."
    )
    
    cli_args = parser.parse_args()

    # Update globals that before_first_request might use
    _startup_model_name = cli_args.model_name
    _startup_device = cli_args.device

    if not cli_args.debug: 
        try:
            logger.info(f"Pre-loading CLIP model '{cli_args.model_name}' on device: {cli_args.device}...")
            get_extractor(model_name=cli_args.model_name, device=cli_args.device) # Initialize model at startup
            logger.info("CLIP model pre-loaded successfully via __main__.")
        except Exception as e:
            logger.critical(f"Failed to pre-load model during startup in __main__: {e}", exc_info=True)
            # Depending on policy, might exit or allow Flask to start and fail on first request.
            # For a critical service, exiting might be preferable: exit(1)

    app.run(host=cli_args.host, port=cli_args.port, debug=cli_args.debug)
