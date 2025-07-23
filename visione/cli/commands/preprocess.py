import os
import subprocess
import sys
import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn

from .command import BaseCommand
from ...utils.normalize_utils import normalize_path, normalize_filename

class PreprocessCommand(BaseCommand):
    """ Implements the 'preprocess' CLI command. """

    def add_arguments(self, subparsers):
        parser = subparsers.add_parser('preprocess', help='Preprocesses videos in raw_video_dir or a single file.')
        parser.add_argument('--single', type=Path, default=None,
                            help='Preprocess only this video file instead of scanning the raw video directory.')
        parser.set_defaults(func=self)

    def __call__(self, *, single: Path = None, **kwargs):
        super().__call__(**kwargs)

        # Load settings from config.yaml
        config = self.config.get('preprocessing', {})
        raw_video_dir = self.collection_dir / config.get('raw_video_dir', 'raw_videos')
        output_video_dir = self.collection_dir / config.get('output_video_dir', 'videos')
        import os
        max_workers = config.get('max_workers', min(max(os.cpu_count() - 8, 1), 56))  # Use ~80-90% of CPUs by default
        ffmpeg_path = config.get('ffmpeg_path', 'ffmpeg')
        skip_existing = config.get('skip_existing', True)

        # Build ffmpeg args from config
        self.standard_video_args, self.cpu_fallback_args = self._build_ffmpeg_args(config)

        print(f"Looking for input videos in: {raw_video_dir.absolute()}")
        os.makedirs(output_video_dir, exist_ok=True)

        # Determine input list
        if single is not None:
            files = [str(single)]
            if not Path(single).exists():
                print(f"[ERROR] --single file not found: {single}")
                return 1
        else:
            files = []
            for ext in ('.mp4', '.mov', '.avi', '.mkv', '.webm'):
                files.extend(glob.glob(os.path.join(raw_video_dir, f'**/*{ext}'), recursive=True))
            if not files:
                print(f"No supported video files found in {raw_video_dir}. Nothing to do.")
                return 0

        use_gpu = self._has_gpu_ffmpeg(ffmpeg_path)

        results = {'success': 0, 'skipped': 0, 'failed': []}

        def get_video_codec(input_path):
            import json
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name', '-of', 'json', str(input_path)
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                try:
                    info = json.loads(result.stdout)
                    return info['streams'][0]['codec_name'] if info.get('streams') else None
                except (json.JSONDecodeError, IndexError):
                    return None
            return None

        def process_one(f):
            # Normalize input filename and output path
            try:
                relative_path = Path(f).relative_to(raw_video_dir)
            except ValueError:
                relative_path = Path(f).name
            normalized_relative_path = normalize_path(str(relative_path.parent))

            norm_out_name = normalize_filename(relative_path.stem) + '.mp4'
            norm_out_dir = output_video_dir / normalized_relative_path
            out = norm_out_dir / norm_out_name

            os.makedirs(norm_out_dir, exist_ok=True)

            if skip_existing and out.exists():
                print(f"Skipping {out}, already exists.")
                return 'skipped'

            # Detect codec and choose pipeline
            codec = get_video_codec(f)
            gpu_codecs = {"h264", "hevc", "av1", "vp9", "mjpeg"}
            use_gpu_this = use_gpu and codec in gpu_codecs

            success, reason = False, ""
            if use_gpu_this:
                success, reason = self._preprocess_video(ffmpeg_path, f, str(out), use_gpu=True)

            if not success:
                success, reason = self._preprocess_video(ffmpeg_path, f, str(out), use_gpu=False)

            if success:
                # Validate output with ffprobe
                val_cmd = ['ffprobe', '-v', 'error', '-show_format', '-show_streams', str(out)]
                if subprocess.run(val_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                    self._index_to_elasticsearch(f, str(out))
                    return 'success'
                else:
                    reason = f"Output file could not be read by ffprobe: {out}"
                    success = False # Mark as failure

            if not success:
                return ('failed', f, reason)

        progress_cols = [
            SpinnerColumn(),
            *Progress.get_default_columns()[:1],
            MofNCompleteColumn(),
            *Progress.get_default_columns()[1:],
            TimeElapsedColumn()
        ]
        with Progress(*progress_cols, transient=True) as progress:
            task = progress.add_task('', total=len(files))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for i, result in enumerate(pool.map(process_one, files)):
                    if result == 'success':
                        results['success'] += 1
                    elif result == 'skipped':
                        results['skipped'] += 1
                    elif isinstance(result, tuple) and result[0] == 'failed':
                        results['failed'].append((result[1], result[2]))
                    progress.update(task, advance=1, description=f"Processing video {i+1}/{len(files)}")

        print("\n--- Preprocessing Summary ---")
        print(f"Successfully processed: {results['success']}")
        print(f"Skipped (already exist): {results['skipped']}")
        print(f"Failed: {len(results['failed'])}")

        if results['failed']:
            print("\n--- Failed Videos ---")
            for file, reason in results['failed']:
                print(f"File: {file}")
                print(f"Reason: {reason}\n")

        print("Preprocessing complete.")
        return 0

    def _build_ffmpeg_args(self, config):
        # GPU args
        standard_args = [
            '-c:v', config.get('video_codec', 'h264_nvenc'),
            '-vf', f"scale=w=1920:h=1080:force_original_aspect_ratio=decrease,format={config.get('color_format', 'rgb24')},scale=trunc(iw/2)*2:trunc(ih/2)*2",
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

        args.extend(['-i', str(input_path)])
        args.extend(self.standard_video_args if use_gpu else self.cpu_fallback_args)
        args.append(output_path)

        try:

            result = subprocess.run(args, check=True, capture_output=True, text=True)

            return True, None
        except subprocess.CalledProcessError as e:
            error_message = f"FFmpeg failed ({'GPU' if use_gpu else 'CPU'}) with exit code {e.returncode}.\n"
            error_message += f"Stderr: {e.stderr.strip()}"

            return False, error_message

    def _index_to_elasticsearch(self, original_path, preprocessed_path, thumbnail_path=None):
        try:
            from ...utils.es_video_metadata import index_video_metadata
            index_video_metadata(original_path, preprocessed_path, thumbnail_path)
        except Exception as e:
            print(f"[WARN] Could not index video in Elasticsearch: {e}")
