import argparse
from pathlib import Path

SUPPORTED_VIDEO_FORMATS = (
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.mpg', '.m4v'
)

def check_missing_previews(collection_path: Path):
    """Scans a Visione collection to find videos missing resized previews."""
    videos_dir = collection_path / 'videos'
    medium_dir = collection_path / 'resized-videos' / 'medium'
    tiny_dir = collection_path / 'resized-videos' / 'tiny'

    if not videos_dir.is_dir():
        print(f"Error: 'videos' directory not found in {collection_path}")
        return

    # Ensure resized directories exist
    medium_dir.mkdir(parents=True, exist_ok=True)
    tiny_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning collection: {collection_path}\n")

    # Get all original videos recursively
    original_videos = {p.stem: p for p in videos_dir.glob('**/*') if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_FORMATS}

    # Get all resized videos (they are always .mp4)
    medium_videos = {p.stem for p in medium_dir.glob('*.mp4')}
    tiny_videos = {p.stem for p in tiny_dir.glob('*.mp4')}

    missing_report = {}

    for video_id, video_path in original_videos.items():
        missing = []
        if video_id not in medium_videos:
            missing.append("medium")
        if video_id not in tiny_videos:
            missing.append("tiny")
        
        if missing:
            missing_report[video_id] = {
                'path': video_path,
                'missing': missing
            }

    if not missing_report:
        print("\n✅ All videos have both medium and tiny previews.")
        return

    print("\n--- Report: Missing Previews ---")
    for video_id, details in missing_report.items():
        missing_str = ' and '.join(details['missing'])
        print(f"- Video '{video_id}': Missing {missing_str} preview(s).")
        # print(f"  (Original path: {details['path']})")

    print(f"\nFound {len(missing_report)} videos with missing previews.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check for missing medium and tiny resized videos in a Visione collection."
    )
    parser.add_argument(
        "collection_path",
        type=str,
        help="The path to your Visione collection directory."
    )
    args = parser.parse_args()

    check_missing_previews(Path(args.collection_path))
