import os
import subprocess
import sys
import glob
from concurrent.futures import ThreadPoolExecutor
from visione.normalize_utils import normalize_path, normalize_filename

# Configuration
RAW_VIDEO_DIR = os.environ.get('VISIONE_RAW_VIDEO_DIR', 'raw_videos')
OUTPUT_VIDEO_DIR = os.environ.get('VISIONE_PREPROCESSED_VIDEO_DIR', 'videos')
MAX_WORKERS = int(os.environ.get('VISIONE_PREPROCESS_WORKERS', '2'))
FFMPEG_PATH = os.environ.get('VISIONE_FFMPEG_PATH', 'ffmpeg')

STANDARD_VIDEO_ARGS = [
    '-c:v', 'h264_nvenc',  # Use GPU if available, fallback handled below
    '-vf', 'scale=w=1920:h=1080:force_original_aspect_ratio=decrease,format=rgb24',
    '-pix_fmt', 'yuv420p',
    '-c:a', 'aac',
    '-b:a', '192k',
    '-ar', '44100',
    '-ac', '2',
    '-movflags', '+faststart',
    '-preset', 'fast',
    '-y'
]

CPU_FALLBACK_ARGS = [
    '-c:v', 'libx264',
    '-vf', 'scale=w=1920:h=1080:force_original_aspect_ratio=decrease,format=rgb24',
    '-pix_fmt', 'yuv420p',
    '-c:a', 'aac',
    '-b:a', '192k',
    '-ar', '44100',
    '-ac', '2',
    '-movflags', '+faststart',
    '-preset', 'fast',
    '-y'
]

SUPPORTED_INPUTS = ('.mp4', '.mov', '.avi', '.mkv', '.webm')

def has_gpu_ffmpeg():
    try:
        out = subprocess.check_output([FFMPEG_PATH, '-encoders'], stderr=subprocess.STDOUT, text=True)
        return 'h264_nvenc' in out
    except Exception:
        return False

def preprocess_video(input_path, output_path, use_gpu=True):
    args = [FFMPEG_PATH, '-hide_banner', '-loglevel', 'error', '-hwaccel', 'cuda'] if use_gpu else [FFMPEG_PATH, '-hide_banner', '-loglevel', 'error']
    args += ['-i', input_path]
    args += (STANDARD_VIDEO_ARGS if use_gpu else CPU_FALLBACK_ARGS)
    args += [output_path]
    try:
        print(f"Processing {input_path} -> {output_path} (GPU: {use_gpu})")
        subprocess.check_call(args)
        print(f"Success: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed ({'GPU' if use_gpu else 'CPU'}): {input_path} -> {output_path}\n{e}")
        return False

def index_to_elasticsearch(original_path, preprocessed_path, thumbnail_path=None):
    try:
        from es_video_metadata import index_video_metadata
        index_video_metadata(original_path, preprocessed_path, thumbnail_path)
    except ImportError:
        # Fallback to subprocess if not running in project root
        import subprocess, sys
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'es_video_metadata.py'), original_path, preprocessed_path]
        if thumbnail_path:
            cmd.append(thumbnail_path)
        try:
            subprocess.check_call(cmd)
        except Exception as e:
            print(f"[WARN] Could not index video in Elasticsearch: {e}")

def process_all():
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    files = []
    for ext in SUPPORTED_INPUTS:
        files.extend(glob.glob(os.path.join(RAW_VIDEO_DIR, f'*{ext}')))
    if not files:
        print(f"No supported video files found in {RAW_VIDEO_DIR}. Nothing to do.")
        return
    use_gpu = has_gpu_ffmpeg()
    def process_one(f):
        # Normalize input filename and output path
        norm_in = normalize_path(f)
        norm_out_name = normalize_filename(os.path.splitext(os.path.basename(norm_in))[0]) + '.mp4'
        norm_out_dir = normalize_path(OUTPUT_VIDEO_DIR)
        out = os.path.join(norm_out_dir, norm_out_name)
        if os.path.exists(out):
            print(f"Skipping {out}, already exists.")
            return
        if preprocess_video(norm_in, out, use_gpu=use_gpu) or preprocess_video(norm_in, out, use_gpu=False):
            index_to_elasticsearch(norm_in, out)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        pool.map(process_one, files)

if __name__ == '__main__':
    process_all()
