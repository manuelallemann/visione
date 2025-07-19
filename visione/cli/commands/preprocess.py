import os
import subprocess
import sys
import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .command import BaseCommand
from ...utils.normalize_utils import normalize_path, normalize_filename

class PreprocessCommand(BaseCommand):
    """ Implements the 'preprocess' CLI command. """

    def add_arguments(self, subparsers):
        parser = subparsers.add_parser('preprocess', help='Preprocesses all videos in the raw_video_dir specified in config.yaml.')
        parser.set_defaults(func=self)

    def __call__(self, **kwargs):
        super().__call__(**kwargs)

        # Load settings from config.yaml
        config = self.config.get('preprocessing', {})
        raw_video_dir = self.collection_dir / config.get('raw_video_dir', 'raw_videos')
        output_video_dir = self.collection_dir / config.get('output_video_dir', 'videos')
        max_workers = config.get('max_workers', 2)
        ffmpeg_path = config.get('ffmpeg_path', 'ffmpeg')
        skip_existing = config.get('skip_existing', True)

        # Build ffmpeg args from config
        self.standard_video_args, self.cpu_fallback_args = self._build_ffmpeg_args(config)

        print(f"Looking for input videos in: {raw_video_dir.absolute()}")
        os.makedirs(output_video_dir, exist_ok=True)

        files = []
        for ext in ('.mp4', '.mov', '.avi', '.mkv', '.webm'):
            files.extend(glob.glob(os.path.join(raw_video_dir, f'**/*{ext}'), recursive=True))

        if not files:
            print(f"No supported video files found in {raw_video_dir}. Nothing to do.")
            return 0

        use_gpu = self._has_gpu_ffmpeg(ffmpeg_path)

        def process_one(f):
            # Normalize input filename and output path
            relative_path = Path(f).relative_to(raw_video_dir)
            normalized_relative_path = normalize_path(str(relative_path.parent))
            
            norm_out_name = normalize_filename(relative_path.stem) + '.mp4'
            norm_out_dir = output_video_dir / normalized_relative_path
            out = norm_out_dir / norm_out_name

            os.makedirs(norm_out_dir, exist_ok=True)

            if skip_existing and out.exists():
                print(f"Skipping {out}, already exists.")
                return

            if self._preprocess_video(ffmpeg_path, f, str(out), use_gpu=use_gpu) or \
               self._preprocess_video(ffmpeg_path, f, str(out), use_gpu=False):
                self._index_to_elasticsearch(f, str(out))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pool.map(process_one, files)
        
        print("Preprocessing complete.")
        return 0

    def _build_ffmpeg_args(self, config):
        # GPU args
        standard_args = [
            '-c:v', config.get('video_codec', 'h264_nvenc'),
            '-vf', f"scale=w=1920:h=1080:force_original_aspect_ratio=decrease,format={config.get('color_format', 'rgb24')}",
            '-pix_fmt', config.get('pixel_format', 'yuv420p'),
            '-c:a', config.get('audio_codec', 'aac'),
            '-b:a', config.get('audio_bitrate', '192k'),
            '-ar', str(config.get('audio_sample_rate', '44100')),
            '-ac', str(config.get('audio_channels', '2')),
            '-preset', config.get('preset', 'fast'),
            '-y'
        ]
        if config.get('faststart', True):
            standard_args.extend(['-movflags', '+faststart'])

        # CPU fallback args
        cpu_args = list(standard_args)
        cpu_args[cpu_args.index('-c:v') + 1] = 'libx264'

        return standard_args, cpu_args

    def _has_gpu_ffmpeg(self, ffmpeg_path):
        try:
            out = subprocess.check_output([ffmpeg_path, '-encoders'], stderr=subprocess.STDOUT, text=True)
            return 'h264_nvenc' in out
        except Exception:
            return False

    def _preprocess_video(self, ffmpeg_path, input_path, output_path, use_gpu=True):
        args = [ffmpeg_path, '-hide_banner', '-loglevel', 'error']
        if use_gpu:
            args.extend(['-hwaccel', 'cuda'])
        
        args.extend(['-i', input_path])
        args.extend(self.standard_video_args if use_gpu else self.cpu_fallback_args)
        args.append(output_path)
        try:
            print(f"Processing {input_path} -> {output_path} (GPU: {use_gpu})")
            subprocess.check_call(args)
            print(f"Success: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed ({'GPU' if use_gpu else 'CPU'}): {input_path} -> {output_path}\n{e}")
            return False

    def _index_to_elasticsearch(self, original_path, preprocessed_path, thumbnail_path=None):
        try:
            from ...utils.es_video_metadata import index_video_metadata
            index_video_metadata(original_path, preprocessed_path, thumbnail_path)
        except Exception as e:
            print(f"[WARN] Could not index video in Elasticsearch: {e}")
