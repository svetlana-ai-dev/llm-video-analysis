"""Manual end-to-end smoke check using a generated two-second MP4."""

import json
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from fastapi.testclient import TestClient

from app.main import app


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        video = Path(directory) / "smoke.mp4"
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y",
            "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video),
        ]
        subprocess.run(command, check=True, capture_output=True)
        with video.open("rb") as handle:
            response = TestClient(app).post(
                "/api/analyze", files={"file": ("smoke.mp4", handle, "video/mp4")}
            )
        data = response.json()
        summary = {
            "http_status": response.status_code,
            "video_analyzer": data.get("video", {}).get("analyzer"),
            "frames_analyzed": data.get("video", {}).get("frames_analyzed"),
            "speech_status": data.get("speech", {}).get("status"),
            "audio_duration": data.get("speech", {}).get("duration_seconds"),
            "llm_status": data.get("llm", {}).get("status"),
        }
        print(json.dumps(summary, ensure_ascii=False))
        assert response.status_code == 200
        assert data["video"]["analyzer"] == "mediapipe-face-mesh"
        assert data["video"]["frames_analyzed"] > 0
        assert data["speech"]["duration_seconds"] > 1
        assert data["llm"]["status"] == "unavailable"


if __name__ == "__main__":
    main()
