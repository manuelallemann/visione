import argparse
import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)
# Ensure a handler is configured for the logger if not already done by the main CLI
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Default transcoding settings
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_AUDIO_CODEC = "aac"
DEFAULT_CRF = 23 # Constant Rate Factor (0-51 for x264, lower is better quality, 23 is a good default)
DEFAULT_PRESET = "medium" # Encoding speed preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
DEFAULT_OUTPUT_FORMAT = "mp4"

class TranscodeCommand:
    """
    CLI Command to transcode videos to optimal formats for processing.
    Detects problematic formats, transcodes them using ffmpeg,
    and generates a report of videos that couldn't be processed.
    """

    @staticmethod
    def register_command(subparsers: argparse._SubParsersAction) -> None:
        """Registers the transcode command and its arguments."""
        parser = subparsers.add_parser(
            "transcode",
            help="Transcode videos to optimal formats for processing.",
            description="Scans an input directory for videos, transcodes them to a specified format "
                        "and quality, and reports any failures.",
        )
        parser.add_argument(
            "input_dir",
            type=Path,
            help="Directory containing videos to transcode.",
        )
        parser.add_argument(
            "output_dir",
            type=Path,
            help="Directory where transcoded videos will be saved.",
        )
        parser.add_argument(
            "--output-format",
            type=str,
            default=DEFAULT_OUTPUT_FORMAT,
            help=f"Output container format (default: {DEFAULT_OUTPUT_FORMAT}). E.g., mp4, mkv.",
        )
        parser.add_argument(
            "--video-codec",
            type=str,
            default=DEFAULT_VIDEO_CODEC,
            help=f"Video codec for transcoding (default: {DEFAULT_VIDEO_CODEC}). E.g., libx264, libx265, copy.",
        )
        parser.add_argument(
            "--audio-codec",
            type=str,
            default=DEFAULT_AUDIO_CODEC,
            help=f"Audio codec for transcoding (default: {DEFAULT_AUDIO_CODEC}). E.g., aac, mp3, copy.",
        )
        parser.add_argument(
            "--crf",
            type=int,
            default=DEFAULT_CRF,
            help=f"Constant Rate Factor for video quality (default: {DEFAULT_CRF}). Lower is better quality.",
        )
        parser.add_argument(
            "--preset",
            type=str,
            default=DEFAULT_PRESET,
            help=f"Encoding speed/compression preset (default: {DEFAULT_PRESET}). "
                 "Affects encoding time and file size.",
        )
        parser.add_argument(
            "--target-resolution",
            type=str,
            default=None,
            help="Target resolution (e.g., '1920x1080' or '1280x-1' to keep aspect ratio). "
                 "If not set, original resolution is kept.",
        )
        parser.add_argument(
            "--target-framerate",
            type=int,
            default=None,
            help="Target framerate. If not set, original framerate is kept.",
        )
        parser.add_argument(
            "--num-audio-channels",
            type=int,
            default=None, # None means keep original, or specify e.g. 2 for stereo
            help="Target number of audio channels (e.g., 2 for stereo). "
                 "If not set, original audio channels are kept if possible.",
        )
        parser.add_argument(
            "--report-file",
            type=Path,
            default=None,
            help="Path to save the JSON report of failed transcodes. "
                 "Defaults to 'transcode_report.json' in the output directory.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force transcoding even if output file already exists.",
        )
        parser.add_argument(
            "--skip-problematic-check",
            action="store_true",
            help="Skip checking if video is problematic and transcode all videos. "
                 "Useful if you want to re-encode everything to a standard format.",
        )
        parser.add_argument(
            "--ffmpeg-path",
            type=str,
            default="ffmpeg",
            help="Path to the ffmpeg executable (default: ffmpeg, assumes it's in PATH)."
        )
        parser.add_argument(
            "--ffprobe-path",
            type=str,
            default="ffprobe",
            help="Path to the ffprobe executable (default: ffprobe, assumes it's in PATH)."
        )
        parser.set_defaults(func=TranscodeCommand.run_command)

    @staticmethod
    def run_command(args: argparse.Namespace) -> None:
        """Executes the transcode command."""
        command = TranscodeCommand(args)
        command.process_videos()

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.failed_transcodes: Dict[str, str] = {}

        if not self.args.input_dir.is_dir():
            logger.error(f"Input directory not found: {self.args.input_dir}")
            raise FileNotFoundError(f"Input directory not found: {self.args.input_dir}")

        self.args.output_dir.mkdir(parents=True, exist_ok=True)

        if self.args.report_file is None:
            self.report_file = self.args.output_dir / "transcode_report.json"
        else:
            self.report_file = self.args.report_file
            self.report_file.parent.mkdir(parents=True, exist_ok=True)


    def _get_video_info(self, video_path: Path) -> Optional[Dict[str, Any]]:
        """Uses ffprobe to get video stream information."""
        cmd = [
            self.args.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe error for {video_path}: {e.stderr}")
            self.failed_transcodes[str(video_path.name)] = f"ffprobe error: {e.stderr}"
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ffprobe JSON output for {video_path}: {e}")
            self.failed_transcodes[str(video_path.name)] = f"ffprobe JSON parse error: {e}"
            return None

    def _is_video_problematic(self, video_path: Path, info: Dict[str, Any]) -> bool:
        """
        Determines if a video needs transcoding based on its properties.
        This is a basic check; can be expanded based on specific needs.
        """
        if self.args.skip_problematic_check:
            return True # Transcode all videos if skipping check

        video_stream: Optional[Dict[str, Any]] = None
        audio_stream: Optional[Dict[str, Any]] = None

        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
            elif stream.get("codec_type") == "audio":
                audio_stream = stream
        
        if not video_stream:
            logger.warning(f"No video stream found in {video_path.name}")
            self.failed_transcodes[str(video_path.name)] = "No video stream found"
            return False # Cannot transcode if no video stream

        # Check 1: Video codec (if not 'copy' and different from target)
        if self.args.video_codec != "copy" and video_stream.get("codec_name") != self.args.video_codec.replace("lib",""): # e.g. libx264 vs x264
             # A more robust check would map libx264 to x264, etc.
            current_codec = video_stream.get("codec_name", "unknown")
            # Simple check against common problematic codecs for general purpose processing
            # ProRes, DNxHD, etc., are often large and not ideal for some pipelines.
            # This list can be expanded.
            problematic_input_codecs = ["prores", "dnxhd", "mjpeg", "mpeg2video"]
            if current_codec in problematic_input_codecs:
                 logger.info(f"Video {video_path.name} has problematic codec {current_codec}, marking for transcode.")
                 return True
            # If target is specific (e.g. h264) and current is not, transcode.
            if self.args.video_codec == "libx264" and current_codec != "h264":
                logger.info(f"Video {video_path.name} codec {current_codec} differs from target h264, marking for transcode.")
                return True


        # Check 2: Audio channels (if a specific number is targeted)
        if self.args.num_audio_channels is not None and audio_stream:
            current_channels = audio_stream.get("channels")
            if current_channels is not None and current_channels != self.args.num_audio_channels:
                logger.info(f"Video {video_path.name} has {current_channels} audio channels, target is {self.args.num_audio_channels}. Marking for transcode.")
                return True
        
        # Check 3: Container format
        current_format = info.get("format", {}).get("format_name", "")
        # ffprobe might return comma-separated list e.g. "mov,mp4,m4a,3gp,3g2,mj2"
        # We are interested in the primary container.
        # For simplicity, if it's not the target output format, consider transcoding.
        # This is a broad check and might lead to more transcoding than strictly necessary.
        if self.args.output_format not in current_format.split(','):
            logger.info(f"Video {video_path.name} format {current_format} differs from target {self.args.output_format}. Marking for transcode.")
            return True

        # Add more checks here: resolution, framerate, specific problematic audio codecs, etc.
        # For example, if target resolution is set and current resolution is different.
        if self.args.target_resolution:
            width = video_stream.get("width")
            height = video_stream.get("height")
            if width and height:
                target_w_str, target_h_str = self.args.target_resolution.split('x')
                if target_w_str != str(width) or (target_h_str != '-1' and target_h_str != str(height)):
                    logger.info(f"Video {video_path.name} resolution {width}x{height} differs from target {self.args.target_resolution}. Marking for transcode.")
                    return True
        
        logger.info(f"Video {video_path.name} seems OK, skipping transcode based on current checks.")
        return False


    def _transcode_video(self, input_path: Path, output_path: Path) -> bool:
        """Transcodes a single video using ffmpeg."""
        if output_path.exists() and not self.args.force:
            logger.info(f"Output file {output_path} already exists. Skipping (use --force to overwrite).")
            return True # Consider it success if already exists and not forcing

        ffmpeg_cmd_parts: List[str] = [
            self.args.ffmpeg_path,
            "-y",  # Overwrite output without asking
            "-i", str(input_path),
            "-c:v", self.args.video_codec,
        ]

        if self.args.video_codec != "copy":
            ffmpeg_cmd_parts.extend(["-crf", str(self.args.crf)])
            ffmpeg_cmd_parts.extend(["-preset", self.args.preset])
        
        if self.args.target_resolution:
            ffmpeg_cmd_parts.extend(["-vf", f"scale={self.args.target_resolution}"])
        
        if self.args.target_framerate:
            ffmpeg_cmd_parts.extend(["-r", str(self.args.target_framerate)])

        ffmpeg_cmd_parts.extend(["-c:a", self.args.audio_codec])
        if self.args.audio_codec != "copy" and self.args.num_audio_channels is not None:
             ffmpeg_cmd_parts.extend(["-ac", str(self.args.num_audio_channels)])
        
        # Add strict experimental for some AAC encoders if needed, though often not required with modern ffmpeg
        if self.args.audio_codec == "aac":
            ffmpeg_cmd_parts.extend(["-strict", "experimental"])


        ffmpeg_cmd_parts.append(str(output_path))

        logger.info(f"Transcoding {input_path.name} to {output_path.name}...")
        logger.debug(f"FFmpeg command: {shlex.join(ffmpeg_cmd_parts)}")

        try:
            process = subprocess.Popen(ffmpeg_cmd_parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                logger.info(f"Successfully transcoded {input_path.name}")
                return True
            else:
                logger.error(f"ffmpeg error for {input_path.name} (return code {process.returncode}):\n{stderr}")
                self.failed_transcodes[str(input_path.name)] = f"ffmpeg error (code {process.returncode}): {stderr}"
                if output_path.exists(): # Clean up partially created file on error
                    output_path.unlink()
                return False
        except FileNotFoundError:
            logger.error(f"ffmpeg executable not found at '{self.args.ffmpeg_path}'. Please ensure it's installed and in PATH or provide correct path via --ffmpeg-path.")
            self.failed_transcodes[str(input_path.name)] = f"ffmpeg not found at '{self.args.ffmpeg_path}'"
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during transcoding of {input_path.name}: {e}")
            self.failed_transcodes[str(input_path.name)] = f"Unexpected transcoding error: {str(e)}"
            if output_path.exists():
                 output_path.unlink()
            return False

    def _generate_report(self) -> None:
        """Generates a JSON report of failed transcodes."""
        logger.info(f"Generating transcode report at {self.report_file}")
        try:
            with open(self.report_file, "w") as f:
                json.dump(self.failed_transcodes, f, indent=4)
            logger.info(f"Report saved successfully.")
        except IOError as e:
            logger.error(f"Failed to write report file {self.report_file}: {e}")

    def process_videos(self) -> None:
        """
        Scans the input directory, identifies videos for transcoding,
        performs transcoding, and generates a report.
        """
        video_files: List[Path] = []
        # Common video extensions
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg", ".ts", ".mts", ".m2ts"]
        # ProRes and GoPro specific often in .mov
        # H264 is a codec, usually in .mp4 or .mov

        logger.info(f"Scanning {self.args.input_dir} for videos...")
        for ext in video_extensions:
            video_files.extend(list(self.args.input_dir.rglob(f"*{ext}"))) # rglob for recursive
            video_files.extend(list(self.args.input_dir.rglob(f"*{ext.upper()}")))


        if not video_files:
            logger.info("No video files found in the input directory.")
            self._generate_report() # Generate an empty report or report scan failure
            return

        logger.info(f"Found {len(video_files)} potential video files.")
        
        successful_transcodes = 0
        skipped_transcodes = 0

        for video_path in tqdm(video_files, desc="Processing videos", unit="video"):
            logger.debug(f"Checking video: {video_path}")
            video_info = self._get_video_info(video_path)

            if not video_info:
                # _get_video_info already logs and adds to failed_transcodes
                continue

            output_filename = f"{video_path.stem}_transcoded.{self.args.output_format}"
            # Preserve subdirectory structure if input was recursive
            relative_path = video_path.parent.relative_to(self.args.input_dir)
            output_subdir = self.args.output_dir / relative_path
            output_subdir.mkdir(parents=True, exist_ok=True)
            output_path = output_subdir / output_filename
            
            if output_path.exists() and not self.args.force:
                 logger.info(f"Output {output_path} exists, skipping (not forced).")
                 skipped_transcodes +=1
                 continue

            if self._is_video_problematic(video_path, video_info):
                if self._transcode_video(video_path, output_path):
                    successful_transcodes += 1
            else:
                # If not problematic, and user doesn't want to skip check (i.e. transcode all)
                # we can optionally copy the file if it's not already in the target format/location.
                # For now, we just log it.
                logger.info(f"Video {video_path.name} is not considered problematic and will not be transcoded.")
                skipped_transcodes +=1


        logger.info(f"Transcoding process finished.")
        logger.info(f"Successfully transcoded: {successful_transcodes}")
        logger.info(f"Skipped (already exists or not problematic): {skipped_transcodes}")
        logger.info(f"Failed to process/transcode: {len(self.failed_transcodes)}")

        self._generate_report()

def main_cli() -> None:
    """Example of how this command might be integrated or run standalone."""
    parser = argparse.ArgumentParser(description="Visione CLI - Transcode Tool")
    subparsers = parser.add_subparsers(title="commands", dest="command")
    TranscodeCommand.register_command(subparsers) # type: ignore

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    # This allows running the transcode command directly for testing
    # In a real CLI, `register_command` would be called by the main CLI entry point.
    main_cli()
