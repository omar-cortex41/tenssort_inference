import subprocess
import time
import signal
import sys
import os

# ================= CONFIG =================

RTSP_HOST = "127.0.0.1"
START_PORT = 8554

VIDEO_PATHS = [
    "../videos/vid1.mp4",
    "../videos/vid2.mp4",
    "../videos/vid3.mp4",
    "../videos/vid4.mp4",
    "../videos/vid5.mp4",
    "../videos/vid6.mp4",
    "../videos/vid7.mp4",
    "../videos/vid8.mp4",
]

# =========================================

processes = []


def start_rtsp_stream(video_path, port):
    stream_name = os.path.splitext(os.path.basename(video_path))[0]
    rtsp_url = f"rtsp://{RTSP_HOST}:{port}/{stream_name}"

    cmd = [
    "ffmpeg",
    "-re",
    "-stream_loop", "-1",
    "-i", video_path,

    "-map", "0:v:0",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-pix_fmt", "yuv420p",

    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    "-rtsp_flags", "listen",

    rtsp_url,
]


    print(f"[INFO] Streaming {video_path}")
    print(f"[INFO] RTSP URL → {rtsp_url}")

    return subprocess.Popen(cmd)


def shutdown(sig, frame):
    print("\n[INFO] Stopping all RTSP streams...")
    for p in processes:
        p.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


def main():
    port = START_PORT

    for video in VIDEO_PATHS:
        if not os.path.exists(video):
            print(f"[WARN] File not found: {video}")
            continue

        p = start_rtsp_stream(video, port)
        processes.append(p)
        port += 1
        time.sleep(0.4)

    print("\n[INFO] All RTSP streams are LIVE.")
    print("[INFO] Open another terminal and run ffplay.")

    while True:
        time.sleep(5)


if __name__ == "__main__":
    main()
