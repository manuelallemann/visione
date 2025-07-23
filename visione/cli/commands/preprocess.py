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
        # Detect number of GPUs
        num_gpus = self._detect_num_gpus()
        if num_gpus > 0:
            max_workers = config.get('max_workers', num_gpus)
        else:
            max_workers = config.get('max_workers', 2)
        if max_workers > num_gpus and num_gpus > 0:
            print(f"[WARN] max_workers ({max_workers}) > num_gpus ({num_gpus}). This may cause context-switch overhead.")

        # FFmpeg version/build check
        self._check_ffmpeg_build(ffmpeg_path)

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

        invalid_outputs = []

        def process_one(args):
            f, gpu_idx = args
            # Normalize input filename and output path
            try:
                relative_path = Path(f).relative_to(raw_video_dir)
            except ValueError:
                # If file is outside raw_video_dir (when using --single), keep only filename
                relative_path = Path(f).name
            normalized_relative_path = normalize_path(str(relative_path.parent))
            
            norm_out_name = normalize_filename(relative_path.stem) + '.mp4'
            norm_out_dir = output_video_dir / normalized_relative_path
            out = norm_out_dir / norm_out_name

            os.makedirs(norm_out_dir, exist_ok=True)

            if skip_existing and out.exists():
                print(f"Skipping {out}, already exists.")
                return

            success = False
            if self._preprocess_video(ffmpeg_path, f, str(out), use_gpu=use_gpu):
                success = True
            elif self._preprocess_video(ffmpeg_path, f, str(out), use_gpu=False):
                success = True

            if success:
                # Validate output with ffprobe
                val_cmd = ['ffprobe', '-v', 'error', '-show_format', '-show_streams', str(out)]
                if subprocess.call(val_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                    self._index_to_elasticsearch(f, str(out))
                else:
                    print(f"[WARN] ffprobe could not read {out}, marking as invalid.")
                    invalid_outputs.append(str(out))

        # Assign each job a GPU index in round-robin
        gpu_assignments = [(f, i % num_gpus if num_gpus > 0 else None) for i, f in enumerate(files)]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pool.map(process_one, gpu_assignments)

        if invalid_outputs:
            print("\n[WARN] The following outputs could not be probed by ffprobe (possibly corrupt):")
            for p in invalid_outputs:
                print(f" - {p}")
        print("Preprocessing complete.")
        return 0

    def _build_ffmpeg_args(self, config):
        # Determine codec and scale filter
        video_codec = config.get('video_codec', 'hevc_nvenc')  # default to HEVC
        if config.get('av1', False):
            video_codec = 'av1_nvenc'
        # Use scale_cuda for GPU scaling, fallback to scale for CPU
        scale_filter = config.get('scale_filter', 'scale_cuda')
        scale_str = f"{scale_filter}=w=1920:h=1080:force_original_aspect_ratio=decrease,format={config.get('color_format', 'rgb24')}"
        standard_args = [
            '-c:v', video_codec,
            '-vf', scale_str,
            '-pix_fmt', config.get('pixel_format', 'yuv420p'),
            '-c:a', config.get('audio_codec', 'aac'),
            '-b:a', config.get('audio_bitrate', '192k'),
            '-ar', str(config.get('audio_sample_rate', '44100')),
            '-ac', str(config.get('audio_channels', '2')),
            '-preset', config.get('preset', 'p1'),  # p1 for max speed
            '-y'
        ]
        if config.get('faststart', True):
            standard_args.extend(['-movflags', '+faststart'])
        # CPU fallback
        cpu_args = list(standard_args)
        cpu_args[cpu_args.index('-c:v') + 1] = 'libx264'
        cpu_args[cpu_args.index('-vf')] = '-vf'
        cpu_args[cpu_args.index('-vf') + 1] = f"scale=w=1920:h=1080:force_original_aspect_ratio=decrease,format={config.get('color_format', 'rgb24')}"
        return standard_args, cpu_args

    def _detect_num_gpus(self):
        try:
            import subprocess
            out = subprocess.check_output(['nvidia-smi', '-L'], text=True)
            return len([line for line in out.splitlines() if line.strip().startswith('GPU ')])
        except Exception:
            return 0

    def _check_ffmpeg_build(self, ffmpeg_path):
        try:
            import subprocess
            version = subprocess.check_output([ffmpeg_path, '-version'], text=True)
            if 'ffmpeg version' in version:
                vnum = version.split('ffmpeg version')[1].split()[0]
                major = int(vnum.split('.')[0])
                if major < 6:
                    print(f"[WARN] FFmpeg version {vnum} < 6.0. Full L40 performance requires FFmpeg >= 6.x.")
            conf = subprocess.check_output([ffmpeg_path, '-buildconf'], text=True)
            if '--enable-cuda-nvcc' not in conf or '--enable-libnpp' not in conf or '--enable-nonfree' not in conf:
                print("[WARN] FFmpeg is missing recommended build flags: --enable-cuda-nvcc --enable-libnpp --enable-nonfree")
        except Exception as e:
            print(f"[WARN] Could not check FFmpeg build: {e}")

    def _has_gpu_ffmpeg(self, ffmpeg_path):
        try:
            out = subprocess.check_output([ffmpeg_path, '-encoders'], stderr=subprocess.STDOUT, text=True)
            return 'h264_nvenc' in out
        except Exception:
            return False

    def _preprocess_video(self, ffmpeg_path, input_path, output_path, use_gpu=True, gpu_idx=None):
        args = [ffmpeg_path, '-hide_banner', '-loglevel', 'error']
        if use_gpu:
            args.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
            if gpu_idx is not None:
                args.extend(['-hwaccel_device', str(gpu_idx), '-gpu', str(gpu_idx)])
        args.extend(['-i', input_path])
        args.extend(self.standard_video_args if use_gpu else self.cpu_fallback_args)
        args.append(output_path)
        try:
            print(f"Processing {input_path} -> {output_path} (GPU: {use_gpu}, GPU_IDX: {gpu_idx})")
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
