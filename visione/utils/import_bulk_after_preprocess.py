import os
import subprocess
import glob

PREPROCESSED_DIR = os.environ.get('VISIONE_PREPROCESSED_VIDEO_DIR', 'videos')
VISIONE_CLI = os.environ.get('VISIONE_CLI', 'visione')


def import_all_videos():
    mp4_files = glob.glob(os.path.join(PREPROCESSED_DIR, '*.mp4'))
    if not mp4_files:
        print(f"No .mp4 files found in {PREPROCESSED_DIR}")
        return
    cmd = [VISIONE_CLI, 'import', '--no-copy', '--bulk'] + mp4_files
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        print(f"Error running visione import: {e}")

if __name__ == '__main__':
    import_all_videos()
