import cv2
import subprocess
import requests
import time
import logging
import numpy as np
import json
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

RTSP_URL = "rtsp://admin:admin123456@95.158.17.167:554/Streaming/Channels/301"

API_URL = (
    "https://8002-01kzs015mz2nhds9kkftmwc39a"
    ".cloudspaces.litng.ai/api/v1/detect"
)

CAM_ID = "CAM1"

DETECTION_REQUEST = [
    {
        "detection_type": "person_detection",
        "roi": None
    }
]

# Camera processing resolution
WIDTH = 640
HEIGHT = 360

# FPS extracted from RTSP
FPS = 5

# Motion threshold
MOTION_THRESHOLD = 5000

# Minimum time between API calls
API_COOLDOWN = 10

# Reconnect delay
RECONNECT_DELAY = 3

# API timeout
API_TIMEOUT = 30


# ============================================================
# LOGGING
# ============================================================

LOG_FILE = "rtsp_motion_monitor.log"

logger = logging.getLogger("RTSP_MONITOR")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

file_handler = logging.FileHandler(
    LOG_FILE,
    encoding="utf-8"
)

file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ============================================================
# GLOBALS
# ============================================================

last_api_call = 0
motion_count = 0
restart_count = 0


# ============================================================
# START FFMPEG
# ============================================================

def start_ffmpeg():

    logger.info("=" * 80)
    logger.info("Starting FFmpeg")
    logger.info("RTSP URL: %s", RTSP_URL)

    command = [
        "ffmpeg",

        "-hide_banner",
        "-loglevel", "error",

        # RTSP over TCP
        "-rtsp_transport", "tcp",

        "-i",
        RTSP_URL,

        # No audio
        "-an",

        # Convert to raw frames
        "-f",
        "rawvideo",

        "-pix_fmt",
        "bgr24",

        # Resize + FPS
        "-vf",
        f"fps={FPS},scale={WIDTH}:{HEIGHT}",

        "pipe:1",
    ]

    start_time = time.perf_counter()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
    )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "FFmpeg process started in %.3f seconds | PID=%s",
        elapsed,
        process.pid,
    )

    return process


# ============================================================
# MOTION DETECTION
# ============================================================

def detect_motion(previous_frame, current_frame):

    if previous_frame is None:
        return False, 0

    start = time.perf_counter()

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY,
    )

    current_gray = cv2.cvtColor(
        current_frame,
        cv2.COLOR_BGR2GRAY,
    )

    previous_gray = cv2.GaussianBlur(
        previous_gray,
        (21, 21),
        0,
    )

    current_gray = cv2.GaussianBlur(
        current_gray,
        (21, 21),
        0,
    )

    diff = cv2.absdiff(
        previous_gray,
        current_gray,
    )

    _, threshold = cv2.threshold(
        diff,
        25,
        255,
        cv2.THRESH_BINARY,
    )

    threshold = cv2.dilate(
        threshold,
        None,
        iterations=2,
    )

    motion_pixels = cv2.countNonZero(
        threshold
    )

    elapsed = time.perf_counter() - start

    detected = motion_pixels > MOTION_THRESHOLD


    return detected, motion_pixels


# ============================================================
# CALL DETECTION API
# ============================================================

def call_detection_api(frame):

    global last_api_call
    global motion_count

    now = time.time()

    # Cooldown
    if now - last_api_call < API_COOLDOWN:

        remaining = API_COOLDOWN - (
            now - last_api_call
        )

        logger.info(
            "Motion detected but API cooldown active | "
            "remaining=%.2fs",
            remaining,
        )

        return

    motion_count += 1

    api_start = time.perf_counter()

    logger.info("-" * 80)
    logger.info(
        "MOTION EVENT #%s",
        motion_count,
    )

    # --------------------------------------------------------
    # Encode frame
    # --------------------------------------------------------

    encode_start = time.perf_counter()

    success, encoded_image = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            85,
        ],
    )

    encode_time = time.perf_counter() - encode_start

    if not success:

        logger.error(
            "Failed to encode motion frame | time=%.4fs",
            encode_time,
        )

        return

    image_bytes = encoded_image.tobytes()

    logger.info(
        "Image encoded | size=%.2f KB | time=%.4fs",
        len(image_bytes) / 1024,
        encode_time,
    )

    # --------------------------------------------------------
    # Prepare multipart request
    # --------------------------------------------------------

    files = {
        "image": (
            "motion.jpg",
            image_bytes,
            "image/jpeg",
        )
    }

    data = {
        "cam_id": CAM_ID,
        "detect_req": json.dumps(
            DETECTION_REQUEST
        ),
    }

    # --------------------------------------------------------
    # API request
    # --------------------------------------------------------

    try:

        logger.info(
            "API request started | URL=%s | cam_id=%s",
            API_URL,
            CAM_ID,
        )

        request_start = time.perf_counter()

        response = requests.post(
            API_URL,
            files=files,
            data=data,
            timeout=API_TIMEOUT,
        )

        request_time = (
            time.perf_counter()
            - request_start
        )

        total_time = (
            time.perf_counter()
            - api_start
        )

        last_api_call = time.time()

        logger.info(
            "API response received | "
            "status=%s | request_time=%.3fs | total_time=%.3fs",
            response.status_code,
            request_time,
            total_time,
        )

        # Response body
        try:

            response_json = response.json()

            logger.info(
                "API response: %s",
                json.dumps(
                    response_json,
                    ensure_ascii=False,
                ),
            )

        except ValueError:

            logger.info(
                "API response: %s",
                response.text[:2000],
            )

        if response.ok:

            logger.info(
                "Detection API SUCCESS"
            )

        else:

            logger.error(
                "Detection API FAILED | HTTP %s",
                response.status_code,
            )

    except requests.Timeout:

        elapsed = (
            time.perf_counter()
            - api_start
        )

        logger.error(
            "Detection API TIMEOUT | time=%.3fs",
            elapsed,
        )

    except requests.RequestException as exc:

        elapsed = (
            time.perf_counter()
            - api_start
        )

        logger.error(
            "Detection API ERROR | time=%.3fs | error=%s",
            elapsed,
            exc,
        )

    logger.info("-" * 80)


# ============================================================
# RTSP MONITOR
# ============================================================

def monitor():

    global restart_count

    previous_frame = None

    frame_count = 0

    frame_size = WIDTH * HEIGHT * 3

    while True:

        process = None

        try:

            restart_count += 1

            logger.info(
                "RTSP connection attempt #%s",
                restart_count,
            )

            connection_start = time.perf_counter()

            process = start_ffmpeg()

            logger.info(
                "RTSP/FFmpeg monitor active"
            )

            while True:

                # ------------------------------------------------
                # Read frame
                # ------------------------------------------------

                frame_start = time.perf_counter()

                raw_frame = process.stdout.read(
                    frame_size
                )

                read_time = (
                    time.perf_counter()
                    - frame_start
                )

                # ------------------------------------------------
                # Stream failed
                # ------------------------------------------------

                if len(raw_frame) != frame_size:

                    logger.warning(
                        "RTSP FRAME READ FAILED | "
                        "received=%s/%s bytes | read_time=%.3fs",
                        len(raw_frame),
                        frame_size,
                        read_time,
                    )

                    break

                # ------------------------------------------------
                # Convert raw bytes to frame
                # ------------------------------------------------

                frame = np.frombuffer(
                    raw_frame,
                    dtype=np.uint8,
                )

                frame = frame.reshape(
                    (HEIGHT, WIDTH, 3)
                )

                frame_count += 1

                # ------------------------------------------------
                # Motion detection
                # ------------------------------------------------

                motion, motion_pixels = detect_motion(
                    previous_frame,
                    frame,
                )

                if motion:

                    logger.info(
                        "MOTION DETECTED | "
                        "frame=%s | pixels=%s",
                        frame_count,
                        motion_pixels,
                    )

                    call_detection_api(
                        frame
                    )

                previous_frame = frame

        except Exception as exc:

            logger.exception(
                "MONITOR ERROR: %s",
                exc,
            )

        finally:

            if process:

                logger.info(
                    "Stopping FFmpeg process | PID=%s",
                    process.pid,
                )

                try:

                    process.kill()

                except Exception:
                    pass

                try:

                    process.wait(
                        timeout=3
                    )

                except Exception:
                    pass

        # --------------------------------------------------------
        # Reconnect
        # --------------------------------------------------------

        logger.warning(
            "RTSP disconnected. "
            "Reconnecting in %s seconds...",
            RECONNECT_DELAY,
        )

        time.sleep(
            RECONNECT_DELAY
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    logger.info("=" * 80)
    logger.info("RTSP MOTION MONITOR STARTED")
    logger.info("Camera ID: %s", CAM_ID)
    logger.info("Resolution: %sx%s", WIDTH, HEIGHT)
    logger.info("FPS: %s", FPS)
    logger.info(
        "Motion threshold: %s",
        MOTION_THRESHOLD,
    )
    logger.info(
        "API cooldown: %s seconds",
        API_COOLDOWN,
    )
    logger.info("Log file: %s", LOG_FILE)
    logger.info("=" * 80)

    monitor()