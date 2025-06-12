import argparse
import functools
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import av
import flask
import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from werkzeug.utils import secure_filename

# Attempt to import from the cloned CLIP2Video repository
# These paths are based on the Dockerfile structure and typical library organization
try:
    from CLIP2Video.modules.modeling import CLIP2Video
    from CLIP2Video.modules.tokenization_clip import SimpleTokenizer
except ImportError as e:
    logging.error(
        f"Failed to import from CLIP2Video library. Ensure it's in PYTHONPATH: {e}"
    )
    # Fallback for core Torch classes if CLIP2Video specific ones fail
    CLIP2Video = None
    SimpleTokenizer = None


# Configuration
APP_NAME = "clip2video-feature-extractor"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_PATH = "/usr/src/app/checkpoint/ViT-B-32.pt"
# Assumed name for the file downloaded by gdown id 1bTOlOOqbZD0DOTJlCN74a2LktCurOjUg
CLIP2VIDEO_CHECKPOINT_PATH = (
    "/usr/src/app/checkpoint/clip2video_msrvtt_vitb32.pt"
)

# Video processing parameters (typical for CLIP2Video models)
MAX_FRAMES_PER_VIDEO = 12  # Number of frames to sample from each video
VIDEO_INPUT_RESOLUTION = 224 # Resolution for CLIP ViT-B/32

# Logging setup
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(APP_NAME)


# Simplified Args class to mimic argparse.Namespace for model compatibility
class ModelArgs:
    def __init__(selfself, **kwargs: Any) -> None:
        # Default values inspired by CLIP2Video/main.py and common sense for inference
        selfself.video_header_type = "SpaceTimeTransformer" # Common default
        selfself.clip_model_name = "ViT-B/32"
        selfself.cache_dir = "/usr/src/app/checkpoint" # For CLIP model
        selfself.use_control_token = False
        selfself.manager_type = "text_prompt"
        selfself.manager_config = "{}" # Empty JSON
        selfself.drl_sample_num = 1
        selfself.use_loose_type = False
        selfself.interaction_type = "no" # No interaction for feature extraction
        selfself.max_frames = MAX_FRAMES_PER_VIDEO
        selfself.slice_framepos = 2 # Middle frame sampling strategy
        selfself.fps_extraction = 1 # Default, actual sampling might be uniform

        # Required for model internal CLIP loading
        selfself.pretrained_clip_name = "ViT-B/32" # Matching clip_model_name
        selfself.clip_archive = CLIP_MODEL_PATH # Path to ViT-B-32.pt

        # Update with any provided kwargs
        for key, value in kwargs.items():
            setattr(selfself, key, value)

        # Derived attributes that CLIP2Video model might expect from CLIP's state_dict
        # These would typically be loaded from the CLIP model's state_dict if not passed
        # For ViT-B/32:
        selfself.input_resolution = VIDEO_INPUT_RESOLUTION
        selfself.video_embed_dim = 512 # Output dimension of ViT-B/32 CLIP


class CLIP2VideoFeatureExtractor:
    """Encapsulates CLIP2Video model loading and feature extraction logic."""

    def __init__(self, device_str: str = DEFAULT_DEVICE) -> None:
        self.device = torch.device(device_str)
        logger.info(f"Initializing CLIP2VideoFeatureExtractor on device: {self.device}")

        if CLIP2Video is None or SimpleTokenizer is None:
            msg = "CLIP2Video library components not imported. Extractor cannot function."
            logger.error(msg)
            raise RuntimeError(msg)

        try:
            self.args = ModelArgs()
            self.tokenizer = SimpleTokenizer()
            
            # Load CLIP state dict to get necessary parameters if model expects them
            # clip_state_dict = torch.load(CLIP_MODEL_PATH, map_location='cpu')['state_dict']
            # self.args.input_resolution = clip_state_dict["input_resolution"].item()
            # self.args.video_embed_dim = clip_state_dict["ln_final.weight"].shape[0]
            # logger.info(f"CLIP params: resolution={self.args.input_resolution}, embed_dim={self.args.video_embed_dim}")


            self.model = CLIP2Video(self.args, self.tokenizer)
            
            logger.info(f"Loading CLIP2Video checkpoint from: {CLIP2VIDEO_CHECKPOINT_PATH}")
            # The checkpoint might contain the full model or just weights to be loaded.
            # The original `load_model` in CLIP2Video/main.py loads it like this:
            checkpoint = torch.load(CLIP2VIDEO_CHECKPOINT_PATH, map_location='cpu')
            
            # Check if checkpoint is a state_dict or a full model structure
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint: # Common pattern
                model_state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                 model_state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict): # Assume it's the state_dict directly
                model_state_dict = checkpoint
            else: # If it's not a dict, it might be the model itself (less common for .pt)
                # This case needs specific handling if true. For now, assume state_dict.
                raise ValueError("Unsupported checkpoint format for CLIP2Video model.")

            # Adjust keys if necessary (e.g. if saved with DataParallel, keys might have 'module.' prefix)
            new_state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v  # remove `module.`
                else:
                    new_state_dict[k] = v
            
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

            self.model.to(self.device)
            self.model.eval()
            logger.info("CLIP2Video model loaded and set to evaluation mode.")

            # Image preprocessing transform
            self.image_transform = transforms.Compose([
                transforms.Resize(VIDEO_INPUT_RESOLUTION, interpolation=Image.BICUBIC),
                transforms.CenterCrop(VIDEO_INPUT_RESOLUTION),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073), # CLIP specific
                    (0.26862954, 0.26130258, 0.27577711)  # CLIP specific
                ),
            ])
            logger.info("Image transform pipeline initialized.")

        except Exception as e:
            logger.error(f"Error during CLIP2VideoExtractor initialization: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize CLIP2VideoExtractor: {e}") from e

    def _load_video_frames_av(
        self, video_path: Union[str, Path], num_frames_to_sample: int
    ) -> Optional[torch.Tensor]:
        """Loads, samples, and preprocesses frames from a video using PyAV."""
        video_path_str = str(video_path)
        logger.debug(f"Loading frames from {video_path_str}, aiming for {num_frames_to_sample} frames.")
        try:
            container = av.open(video_path_str)
        except av.AVError as e:
            logger.error(f"PyAV Error opening video {video_path_str}: {e}")
            return None
        
        video_stream = None
        for stream in container.streams:
            if stream.type == 'video':
                video_stream = stream
                break
        if video_stream is None:
            logger.error(f"No video stream found in {video_path_str}")
            container.close()
            return None

        total_frames_in_video = video_stream.frames
        if total_frames_in_video == 0: # Some streams might not report frames count correctly
            # Fallback: decode a few frames to estimate or iterate
            total_frames_in_video = sum(1 for _ in container.decode(video_stream))
            container.seek(-1) # Reset stream
            if total_frames_in_video == 0:
                 logger.error(f"Video stream in {video_path_str} has no frames.")
                 container.close()
                 return None


        indices = np.linspace(
            0, total_frames_in_video - 1, num_frames_to_sample, dtype=int
        )
        
        frames_pil = []
        frames_processed_count = 0
        
        # Efficient frame seeking if possible, otherwise iterate
        # PyAV's frame iteration is generally efficient
        current_frame_idx = -1
        target_indices_set = set(indices)
        
        container.seek(0) # Ensure we start from the beginning
        for frame_idx, frame in enumerate(container.decode(video_stream)):
            if frame_idx in target_indices_set:
                try:
                    img = frame.to_image() # PIL Image
                    frames_pil.append(img)
                    frames_processed_count +=1
                    if frames_processed_count == num_frames_to_sample:
                        break
                except Exception as e:
                    logger.warning(f"Could not decode frame {frame_idx} from {video_path_str}: {e}")
        
        container.close()

        if not frames_pil:
            logger.error(f"No frames could be extracted from {video_path_str}.")
            return None
        
        # If fewer frames were extracted than requested, pad by duplicating last frame
        # This is a common strategy, though others exist (e.g., sampling with replacement)
        while len(frames_pil) < num_frames_to_sample and len(frames_pil) > 0:
            frames_pil.append(frames_pil[-1])
            logger.debug(f"Padding frames for {video_path_str}, now have {len(frames_pil)}")
        
        if not frames_pil: # Should not happen if initial check passed, but as safeguard
            logger.error(f"Frame list became empty after padding for {video_path_str}")
            return None

        frame_tensors = [self.image_transform(img) for img in frames_pil]
        video_tensor = torch.stack(frame_tensors, dim=0) # Shape: (num_frames, C, H, W)
        logger.debug(f"Successfully loaded and processed {video_tensor.shape[0]} frames for {video_path_str}.")
        return video_tensor

    @functools.lru_cache(maxsize=256)
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Extracts embedding for a single text string."""
        logger.debug(f"Request to embed text: '{text[:50]}...'")
        if not text.strip():
            logger.warning("Received empty text for embedding.")
            raise ValueError("Text input cannot be empty.")
        try:
            with torch.no_grad():
                # Tokenize and prepare text input as per CLIP2Video model's requirement
                # The SimpleTokenizer typically returns integer IDs.
                # The model's get_text_feat might expect these IDs directly or a tokenized structure.
                # Based on CLIP2Video.modules.modeling.CLIP2Video.get_text_feat, it takes text_data which is tokenized.
                
                # tokenized_text = self.tokenizer(text).to(self.device) # This is for original CLIP
                # For SimpleTokenizer, it's usually:
                sot_token = self.tokenizer.encoder["<|startoftext|>"]
                eot_token = self.tokenizer.encoder["<|endoftext|>"]
                tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
                
                # The model expects a batch, so wrap in a list and convert to tensor
                text_input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
                
                text_features = self.model.get_text_feat(text_input_ids)
            
            logger.debug(f"Text features extracted with shape: {text_features.shape}")
            return text_features.squeeze().cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting text embedding for '{text[:50]}...': {e}", exc_info=True)
            raise RuntimeError(f"Text embedding extraction failed: {e}")


    def get_video_features(self, video_path: Union[str, Path]) -> np.ndarray:
        """Extracts features for a single video file."""
        logger.debug(f"Request to extract features from video: {video_path}")
        
        video_frames_tensor = self._load_video_frames_av(
            video_path, num_frames_to_sample=self.args.max_frames
        )

        if video_frames_tensor is None:
            msg = f"Could not load frames from video: {video_path}"
            logger.error(msg)
            raise ValueError(msg)

        # Model expects batch: (batch_size, num_frames, C, H, W)
        video_frames_tensor = video_frames_tensor.unsqueeze(0).to(self.device)
        logger.debug(f"Video tensor prepared for model with shape: {video_frames_tensor.shape}")

        try:
            with torch.no_grad():
                video_features = self.model.get_video_feat(video_frames_tensor)
            
            logger.debug(f"Video features extracted with shape: {video_features.shape}")
            # Assuming the model returns one feature vector per video in the batch
            return video_features.squeeze().cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting video features for {video_path}: {e}", exc_info=True)
            raise RuntimeError(f"Video feature extraction failed: {e}")


# Global extractor instance
clip2video_extractor: Optional[CLIP2VideoFeatureExtractor] = None

def get_extractor() -> CLIP2VideoFeatureExtractor:
    """Initializes and returns the global feature extractor instance."""
    global clip2video_extractor
    if clip2video_extractor is None:
        logger.info("CLIP2Video extractor instance not found, initializing...")
        try:
            clip2video_extractor = CLIP2VideoFeatureExtractor(device_str=DEFAULT_DEVICE)
        except RuntimeError as e:
            # Log critical error and ensure extractor remains None
            # The app can start, but feature extraction endpoints will fail.
            logger.critical(f"Failed to initialize CLIP2VideoFeatureExtractor: {e}", exc_info=True)
            clip2video_extractor = None # Ensure it stays None on failure
            raise # Re-raise to make it clear at startup if critical
    
    if clip2video_extractor is None: # Check again if initialization failed and was caught by a higher level
        raise RuntimeError("CLIP2Video extractor could not be initialized.")
        
    return clip2video_extractor


# Flask application
app = flask.Flask(APP_NAME)

@app.before_first_request
def initialize_model():
    """Ensure the model is loaded before the first request."""
    try:
        get_extractor()
        logger.info("CLIP2Video model pre-loading complete.")
    except Exception as e:
        logger.critical(f"Application startup failed: Model could not be loaded. {e}", exc_info=True)
        # Depending on deployment, might want to exit or let Flask handle it
        # For now, subsequent requests will fail if extractor is None

@app.route("/ping", methods=["GET"])
def ping() -> flask.Response:
    """Health check endpoint."""
    return flask.jsonify({"status": "ok", "message": "pong"})


@app.route("/get-text-feature", methods=["POST"])
def get_text_feature_endpoint() -> flask.Response:
    """Extracts feature embedding from input text."""
    try:
        extractor = get_extractor() # Ensures model is loaded
    except RuntimeError as e: # Model loading failed
        logger.error(f"Text feature extraction unavailable: {e}")
        return flask.jsonify({"error": f"Service unavailable: {str(e)}"}), 503

    if not flask.request.is_json:
        return flask.jsonify({"error": "Request must be JSON"}), 400
    
    data = flask.request.get_json()
    text = data.get("text")

    if not text or not isinstance(text, str):
        return flask.jsonify({"error": "Missing or invalid 'text' field in JSON payload"}), 400

    try:
        logger.info(f"Processing /get-text-feature for text: '{text[:50]}...'")
        text_embedding = extractor.get_text_embedding(text)
        return flask.jsonify({"text": text, "embedding": text_embedding.tolist()})
    except ValueError as ve: # e.g. empty text
        logger.warning(f"ValueError in text feature extraction: {ve}")
        return flask.jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in /get-text-feature: {e}", exc_info=True)
        return flask.jsonify({"error": "Failed to extract text features"}), 500


@app.route("/get-video-feature", methods=["POST"])
def get_video_feature_endpoint() -> flask.Response:
    """Extracts features from an input video (URL or uploaded file)."""
    try:
        extractor = get_extractor() # Ensures model is loaded
    except RuntimeError as e: # Model loading failed
        logger.error(f"Video feature extraction unavailable: {e}")
        return flask.jsonify({"error": f"Service unavailable: {str(e)}"}), 503

    video_path_to_process: Optional[str] = None
    temp_file_to_delete: Optional[str] = None

    try:
        if "video_url" in flask.request.form:
            video_url = flask.request.form["video_url"]
            logger.info(f"Processing /get-video-feature for URL: {video_url}")
            
            # Download video from URL to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_f: # Suffix helps some libs
                temp_file_to_delete = tmp_f.name
            
            logger.debug(f"Downloading video from {video_url} to {temp_file_to_delete}")
            response = requests.get(video_url, stream=True, timeout=30) # Added timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            with open(temp_file_to_delete, "wb") as f_out:
                for chunk in response.iter_content(chunk_size=8192):
                    f_out.write(chunk)
            video_path_to_process = temp_file_to_delete
            logger.debug(f"Video downloaded successfully to {video_path_to_process}")

        elif "file" in flask.request.files:
            video_file = flask.request.files["file"]
            if video_file.filename == "":
                return flask.jsonify({"error": "No selected file"}), 400
            
            filename = secure_filename(video_file.filename if video_file.filename else "uploaded_video")
            logger.info(f"Processing /get-video-feature for uploaded file: {filename}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_f:
                temp_file_to_delete = tmp_f.name
            video_file.save(temp_file_to_delete)
            video_path_to_process = temp_file_to_delete
            logger.debug(f"Video saved temporarily to {video_path_to_process}")
        else:
            return flask.jsonify({"error": "Missing 'video_url' or 'file' part in request"}), 400

        if not video_path_to_process: # Should not happen if logic above is correct
             return flask.jsonify({"error": "Internal error: video path not set"}), 500

        video_features = extractor.get_video_features(video_path_to_process)
        return flask.jsonify({
            "source": flask.request.form.get("video_url", flask.request.files.get("file", {}).get("filename", "uploaded_file")),
            "embedding": video_features.tolist()
        })

    except requests.exceptions.RequestException as re:
        logger.error(f"Error downloading video URL: {re}", exc_info=True)
        return flask.jsonify({"error": f"Failed to download video from URL: {str(re)}"}), 400
    except ValueError as ve: # e.g. frame loading issues
        logger.warning(f"ValueError in video feature extraction: {ve}")
        return flask.jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in /get-video-feature: {e}", exc_info=True)
        return flask.jsonify({"error": "Failed to extract video features"}), 500
    finally:
        if temp_file_to_delete and os.path.exists(temp_file_to_delete):
            try:
                os.unlink(temp_file_to_delete)
                logger.debug(f"Temporary file {temp_file_to_delete} deleted.")
            except Exception as e_del:
                logger.error(f"Error deleting temporary file {temp_file_to_delete}: {e_del}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP2Video Feature Extractor Service")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the service to."
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to bind the service to."
    )
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE, help="Device to run model on (e.g., 'cuda', 'cpu')."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run Flask in debug mode."
    )
    
    cli_args = parser.parse_args()

    # Override DEFAULT_DEVICE if specified via CLI for initialization
    if cli_args.device:
        DEFAULT_DEVICE = cli_args.device # Update global for get_extractor if called early

    if not cli_args.debug: # Only pre-load if not in debug mode (reloader might cause issues)
        try:
            logger.info(f"Pre-loading model on device: {DEFAULT_DEVICE}...")
            get_extractor() # Initialize model at startup
            logger.info("Model pre-loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to pre-load model during startup: {e}", exc_info=True)
            # Exit if model loading is critical for the service to be useful
            # exit(1) # Or handle gracefully depending on requirements

    app.run(host=cli_args.host, port=cli_args.port, debug=cli_args.debug)
