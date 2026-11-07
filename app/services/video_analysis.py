from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.models.schemas import EyeTrackingQuality, TimelineEvent, VideoCalibration, VideoMetrics

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - environment-specific binary import
    mp = None


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
LEFT_IRIS = (468, 469, 470, 471, 472)
RIGHT_IRIS = (473, 474, 475, 476, 477)
MOUTH = (61, 291, 13, 14)
BROWS = ((105, 159), (334, 386))
BLINK_CLOSED_RATIO = 0.70
BLINK_REOPEN_RATIO = 0.80
CALIBRATION_SECONDS = 3.0
MIN_CALIBRATION_SAMPLES = 3
GAZE_ENTER_OFFSET = 0.07
GAZE_EXIT_OFFSET = 0.04
EVENT_CONFIRM_SECONDS = 0.35
HEAD_ENTER_DEGREES = 8.0
HEAD_EXIT_DEGREES = 5.0
EVENT_MERGE_INTERVAL_SECONDS = 2.5
EYE_RELIABLE_CONFIDENCE = 0.60
EYE_USABLE_CONFIDENCE = 0.40


def _merge_report_events(
    events: list[TimelineEvent], max_interval: float = EVENT_MERGE_INTERVAL_SECONDS
) -> list[TimelineEvent]:
    """Collapse same-type events in short time windows for report display."""
    merged: list[TimelineEvent] = []
    latest_by_type: dict[str, TimelineEvent] = {}
    for event in events:
        latest = latest_by_type.get(event.type)
        if latest and event.time - (latest.end_time or latest.time) <= max_interval:
            latest.end_time = event.end_time or event.time
        else:
            latest = event.model_copy(deep=True)
            merged.append(latest)
            latest_by_type[event.type] = latest
    return merged


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _eye_aspect(points: np.ndarray, ids: tuple[int, ...]) -> float:
    p1, p2, p3, p4, p5, p6 = (points[i] for i in ids)
    return (_distance(p2, p6) + _distance(p3, p5)) / max(2 * _distance(p1, p4), 1e-6)


def _mean_point(points: np.ndarray, ids: tuple[int, ...]) -> np.ndarray:
    return points[list(ids)].mean(axis=0)


def _face_width(points: np.ndarray) -> float:
    return max(_distance(points[234, :2], points[454, :2]), 1e-6)


@dataclass
class EyeMeasurement:
    center: np.ndarray
    iris_center: np.ndarray
    gaze_ratio: float
    ear: float
    confidence: float
    reasons: tuple[str, ...]


@dataclass
class GazeResult:
    label: str
    ratio: float | None
    confidence: float
    source: str


@dataclass
class _EyeTrackingState:
    left_center: np.ndarray | None = None
    right_center: np.ndarray | None = None
    left_ratio: float | None = None
    right_ratio: float | None = None


def _measure_eye(
    points: np.ndarray,
    contour: tuple[int, ...],
    iris: tuple[int, ...],
    previous_center: np.ndarray | None,
    previous_ratio: float | None,
    closed: bool,
    yaw: float,
) -> EyeMeasurement:
    """Port of benchmark_test.measure_eye with per-eye quality reasons."""
    a = points[contour[0], :2].astype(float)
    b = points[contour[3], :2].astype(float)
    if a[0] > b[0]:
        a, b = b, a
    axis = b - a
    width = float(np.linalg.norm(axis))
    center = _mean_point(points[:, :2], contour)
    iris_center = _mean_point(points[:, :2], iris)
    ear = _eye_aspect(points, contour)
    reasons: list[str] = []
    confidence = 1.0

    if width <= 1e-6:
        return EyeMeasurement(center, iris_center, 0.5, ear, 0.0, ("zero_eye_width",))
    gaze_ratio = float(np.dot(iris_center - a, axis) / np.dot(axis, axis))
    perpendicular = np.array((-axis[1], axis[0]), dtype=float)
    iris_vertical = abs(float(np.dot(iris_center - center, perpendicular) / (width * width)))
    iris_radius = float(np.mean([np.linalg.norm(points[index, :2] - iris_center) for index in iris[1:]])) / width
    relative_width = width / _face_width(points)

    if relative_width < 0.12:
        confidence *= 0.20
        reasons.append("eye_too_narrow")
    elif relative_width < 0.16:
        confidence *= 0.65
        reasons.append("eye_narrow")
    if not 0.08 <= ear <= 0.65:
        confidence *= 0.45
        reasons.append("implausible_ear")
    if not -0.15 <= gaze_ratio <= 1.15:
        confidence *= 0.10
        reasons.append("iris_outside_eye")
    elif not -0.05 <= gaze_ratio <= 1.05:
        confidence *= 0.65
        reasons.append("iris_near_eye_edge")
    if iris_vertical > 0.35:
        confidence *= 0.25
        reasons.append("iris_vertical_outlier")
    if not 0.015 <= iris_radius <= 0.30:
        confidence *= 0.55
        reasons.append("implausible_iris_size")
    if previous_center is not None and np.linalg.norm(center - previous_center) / _face_width(points) > 0.10:
        confidence *= 0.45
        reasons.append("eye_center_jump")
    if previous_ratio is not None and abs(gaze_ratio - previous_ratio) > 0.30:
        confidence *= 0.55
        reasons.append("iris_jump")
    if abs(yaw) > 25:
        confidence *= 0.75
        reasons.append("strong_yaw")
    if closed:
        confidence = 0.0
        reasons.append("eye_closed")
    return EyeMeasurement(
        center, iris_center, gaze_ratio, ear,
        max(0.0, min(1.0, confidence)), tuple(reasons),
    )


def _estimate_gaze(left: EyeMeasurement, right: EyeMeasurement) -> GazeResult:
    """Use only sufficiently reliable eyes; never substitute head pose for gaze."""
    eyes = [("left_eye", left), ("right_eye", right)]
    reliable = [(name, eye) for name, eye in eyes if eye.confidence >= EYE_RELIABLE_CONFIDENCE]
    usable = [(name, eye) for name, eye in eyes if eye.confidence >= EYE_USABLE_CONFIDENCE]
    if not reliable:
        confidence = max((eye.confidence for _, eye in usable), default=0.0)
        source = "low_eye_confidence" if usable else "unavailable"
        return GazeResult("unknown", None, confidence * 0.25, source)
    if len(reliable) == 1:
        name, eye = reliable[0]
        return GazeResult("available", eye.gaze_ratio, eye.confidence * 0.70, name)
    if abs(left.gaze_ratio - right.gaze_ratio) > 0.22:
        return GazeResult("unknown", None, min(left.confidence, right.confidence) * 0.25, "eye_disagreement")
    if not usable:
        return GazeResult("unknown", None, 0.0, "unavailable")
    total_weight = left.confidence + right.confidence
    ratio = (left.gaze_ratio * left.confidence + right.gaze_ratio * right.confidence) / total_weight
    return GazeResult("available", ratio, min(left.confidence, right.confidence), "both_eyes")


def _robust_median(values: list[float], default: float = 0.0) -> float:
    """Median with light outlier trimming for a short calibration window."""
    if not values:
        return default
    data = np.asarray(values, dtype=float)
    if len(data) >= 5:
        low, high = np.percentile(data, (10, 90))
        trimmed = data[(data >= low) & (data <= high)]
        if len(trimmed):
            data = trimmed
    return float(np.median(data))


def _gaze_ratio(points: np.ndarray) -> float | None:
    if len(points) < 478:
        return None
    ratios = []
    for iris_ids, eye_ids in ((LEFT_IRIS, LEFT_EYE), (RIGHT_IRIS, RIGHT_EYE)):
        iris = _mean_point(points, iris_ids)[0]
        left, right = points[eye_ids[0]][0], points[eye_ids[3]][0]
        ratios.append((iris - min(left, right)) / max(abs(right - left), 1e-6))
    return float(np.mean(ratios))


def _classify_gaze(ratio: float | None, baseline: float, previous: str = "center") -> str:
    """Classify gaze around a personal center with entry/exit hysteresis."""
    if ratio is None:
        return "unknown"
    offset = ratio - baseline
    if previous == "left" and offset < -GAZE_EXIT_OFFSET:
        return "left"
    if previous == "right" and offset > GAZE_EXIT_OFFSET:
        return "right"
    if offset < -GAZE_ENTER_OFFSET:
        return "left"
    if offset > GAZE_ENTER_OFFSET:
        return "right"
    return "center"


@dataclass
class _DebouncedState:
    active: bool = False
    candidate: bool | None = None
    candidate_frames: int = 0

    def update(self, desired: bool, required_frames: int) -> bool | None:
        """Return the confirmed new state, or None while stable/pending."""
        if desired == self.active:
            self.candidate = None
            self.candidate_frames = 0
            return None
        if desired != self.candidate:
            self.candidate = desired
            self.candidate_frames = 1
        else:
            self.candidate_frames += 1
        if self.candidate_frames < max(1, required_frames):
            return None
        self.active = desired
        self.candidate = None
        self.candidate_frames = 0
        return self.active


class VideoAnalyzer:
    """Extracts observable facial, gaze and head-motion signals from a video."""

    def __init__(self, sample_fps: float = 5.0) -> None:
        self.sample_fps = max(1.0, min(sample_fps, 15.0))

    def analyze(self, video_path: str | Path) -> VideoMetrics:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise ValueError("OpenCV не смог открыть видео")
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = frame_total / source_fps if source_fps else 0.0
            step = max(1, round(source_fps / self.sample_fps))
            if mp is None:
                return self._analyze_faces_only(capture, source_fps, frame_total, duration, step)
            return self._analyze_landmarks(capture, source_fps, frame_total, duration, step)
        finally:
            # Windows keeps the source file locked until VideoCapture is released.
            capture.release()

    def _analyze_faces_only(self, capture, fps, total, duration, step) -> VideoMetrics:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        analyzed = found = index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                analyzed += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if len(detector.detectMultiScale(gray, 1.1, 4)):
                    found += 1
            index += 1
        return VideoMetrics(
            analyzer="opencv-haar-fallback", duration_seconds=round(duration, 3), source_fps=round(fps, 3),
            analyzed_fps=round(fps / step, 3), frames_total=total, frames_analyzed=analyzed,
            frames_with_face=found, face_presence_ratio=round(found / max(analyzed, 1), 4),
            warnings=["MediaPipe недоступен: рассчитано только наличие лица через OpenCV Haar Cascade."],
        )

    def _analyze_landmarks(self, capture, fps, total, duration, step) -> VideoMetrics:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        try:
            return self._analyze_with_face_mesh(face_mesh, capture, fps, total, duration, step)
        finally:
            face_mesh.close()

    def _analyze_with_face_mesh(self, face_mesh, capture, fps, total, duration, step) -> VideoMetrics:
        analyzed = face_frames = blink_count = gaze_changes = 0
        closed_frames = smile_frames = mouth_open_frames = 0
        gaze_counter: Counter[str] = Counter()
        eye_modes: Counter[str] = Counter()
        gaze_sources: Counter[str] = Counter()
        left_quality_reasons: Counter[str] = Counter()
        right_quality_reasons: Counter[str] = Counter()
        left_confidences: list[float] = []
        right_confidences: list[float] = []
        brow_values: list[float] = []
        facial_changes: list[float] = []
        head_moves: list[float] = []
        poses: list[tuple[float, float, float]] = []
        events: list[TimelineEvent] = []
        previous_points = previous_pose = None
        previous_gaze = "center"
        eye_closed = False
        closed_run = 0
        pending_gaze = "center"
        pending_gaze_frames = 0
        index = 0
        analyzed_rate = fps / step
        confirm_frames = max(1, math.ceil(analyzed_rate * EVENT_CONFIRM_SECONDS))
        calibration_target = min(CALIBRATION_SECONDS, duration) if duration > 0 else CALIBRATION_SECONDS
        calibration_values: dict[str, list[float]] = {
            key: [] for key in ("ear", "gaze_ratio", "yaw", "pitch", "roll", "mouth_ratio", "smile_ratio", "brow")
        }
        calibration: VideoCalibration | None = None
        eye_tracking_state = _EyeTrackingState()
        mouth_state = _DebouncedState()
        smile_state = _DebouncedState()
        head_state = _DebouncedState()
        brow_state = _DebouncedState()
        facial_state = _DebouncedState()
        active_event: dict[str, TimelineEvent] = {}

        def finish_event(event_type: str, event_time: float) -> None:
            event = active_event.pop(event_type, None)
            if event is not None:
                event.end_time = round(event_time, 2)

        def build_calibration() -> VideoCalibration:
            samples = len(calibration_values["ear"])
            baseline = {
                key: round(_robust_median(values, 0.5 if key == "gaze_ratio" else 0.0), 5)
                for key, values in calibration_values.items()
            }
            completed = samples >= MIN_CALIBRATION_SAMPLES
            message = (
                "Калибровка выполнена по первым кадрам с обнаруженным лицом."
                if completed else "Недостаточно кадров с лицом для надёжной калибровки."
            )
            return VideoCalibration(
                completed=completed,
                duration_seconds=round(calibration_target, 2),
                samples=samples,
                baseline=baseline,
                message=message,
            )

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step:
                index += 1
                continue
            time = index / fps
            analyzed += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = face_mesh.process(rgb)
            if not output.multi_face_landmarks:
                eye_modes["unavailable"] += 1
                gaze_sources["unavailable"] += 1
                if eye_closed:
                    eye_closed = False
                index += 1
                continue
            face_frames += 1
            landmarks = output.multi_face_landmarks[0].landmark
            points = np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)
            left_ear = _eye_aspect(points, LEFT_EYE)
            right_ear = _eye_aspect(points, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2
            pose = self._head_pose(points, frame.shape[1], frame.shape[0])
            mouth_width = _distance(points[MOUTH[0]], points[MOUTH[1]])
            mouth_height = _distance(points[MOUTH[2]], points[MOUTH[3]])
            mouth_ratio = mouth_height / max(mouth_width, 1e-6)
            smile_ratio = mouth_width / max(_distance(points[234], points[454]), 1e-6)
            brow_value = float(np.mean([_distance(points[a], points[b]) for a, b in BROWS]))

            if calibration is None and time >= calibration_target:
                calibration = build_calibration()
                previous_points = None
                previous_pose = pose

            baseline = calibration.baseline if calibration and calibration.completed else {}
            open_ear_baseline = baseline.get("ear", _robust_median(calibration_values["ear"], ear))
            closed_threshold = open_ear_baseline * BLINK_CLOSED_RATIO
            reopen_threshold = open_ear_baseline * BLINK_REOPEN_RATIO
            currently_closed = ear < (reopen_threshold if eye_closed else closed_threshold)
            if currently_closed:
                closed_frames += 1
                closed_run += 1

            left_eye = _measure_eye(
                points, LEFT_EYE, LEFT_IRIS, eye_tracking_state.left_center,
                eye_tracking_state.left_ratio, currently_closed, pose[0],
            )
            right_eye = _measure_eye(
                points, RIGHT_EYE, RIGHT_IRIS, eye_tracking_state.right_center,
                eye_tracking_state.right_ratio, currently_closed, pose[0],
            )
            left_confidences.append(left_eye.confidence)
            right_confidences.append(right_eye.confidence)
            left_quality_reasons.update(left_eye.reasons)
            right_quality_reasons.update(right_eye.reasons)
            if left_eye.confidence >= EYE_USABLE_CONFIDENCE:
                eye_tracking_state.left_center = left_eye.center
                eye_tracking_state.left_ratio = left_eye.gaze_ratio
            if right_eye.confidence >= EYE_USABLE_CONFIDENCE:
                eye_tracking_state.right_center = right_eye.center
                eye_tracking_state.right_ratio = right_eye.gaze_ratio
            gaze_result = _estimate_gaze(left_eye, right_eye)
            gaze_sources[gaze_result.source] += 1
            if gaze_result.source == "both_eyes":
                eye_modes["both_eyes"] += 1
            elif gaze_result.source in ("left_eye", "right_eye"):
                eye_modes["single_eye"] += 1
            else:
                eye_modes["unavailable"] += 1

            if calibration is None and time < calibration_target:
                for key, value in (
                    ("ear", ear), ("yaw", pose[0]), ("pitch", pose[1]), ("roll", pose[2]),
                    ("mouth_ratio", mouth_ratio), ("smile_ratio", smile_ratio), ("brow", brow_value),
                ):
                    calibration_values[key].append(float(value))
                if gaze_result.ratio is not None:
                    calibration_values["gaze_ratio"].append(gaze_result.ratio)
            if calibration is not None and currently_closed and not eye_closed:
                events.append(TimelineEvent(time=round(time, 2), type="eyes_closed", description="Начало закрытия глаз", value=round(ear, 3)))
            if calibration is not None and eye_closed and not currently_closed:
                # At 15 Hz a normal blink occupies roughly 1-6 sampled frames.
                # Longer closures remain in eyes_closed_seconds but are not
                # automatically labelled as blinks.
                if 1 <= closed_run <= max(1, round(analyzed_rate * 0.40)):
                    blink_count += 1
                    events.append(TimelineEvent(time=round(time, 2), type="blink", description="Моргание завершено"))
                closed_run = 0
            eye_closed = currently_closed

            mouth_enter = max(0.16, baseline.get("mouth_ratio", 0.11) + 0.045)
            mouth_exit = max(0.12, mouth_enter - 0.025)
            mouth_is_open = mouth_ratio > (mouth_exit if mouth_state.active else mouth_enter)
            smile_enter = max(0.38, baseline.get("smile_ratio", 0.34) + 0.025)
            smile_exit = smile_enter - 0.015
            is_smiling = smile_ratio > (smile_exit if smile_state.active else smile_enter) and mouth_ratio < 0.32
            mouth_open_frames += mouth_is_open
            smile_frames += is_smiling
            if calibration is not None:
                mouth_transition = mouth_state.update(mouth_is_open, confirm_frames)
                if mouth_transition is True:
                    event = TimelineEvent(time=round(time, 2), type="mouth_open", description="Устойчивое открытие рта", value=round(mouth_ratio, 3))
                    events.append(event); active_event["mouth_open"] = event
                elif mouth_transition is False:
                    finish_event("mouth_open", time)
                smile_transition = smile_state.update(is_smiling, confirm_frames)
                if smile_transition is True:
                    event = TimelineEvent(time=round(time, 2), type="smile", description="Устойчивая улыбкоподобная геометрия", value=round(smile_ratio, 3))
                    events.append(event); active_event["smile"] = event
                elif smile_transition is False:
                    finish_event("smile", time)
            brow_values.append(brow_value)
            brow_delta = abs(brow_value - baseline.get("brow", brow_value))
            brow_desired = brow_delta > (0.008 if brow_state.active else 0.012)
            if calibration is not None:
                brow_transition = brow_state.update(brow_desired, confirm_frames)
                if brow_transition is True:
                    event = TimelineEvent(time=round(time, 2), type="eyebrow_movement", description="Устойчивое отклонение положения бровей от baseline", value=round(brow_delta, 4))
                    events.append(event); active_event["eyebrow_movement"] = event
                elif brow_transition is False:
                    finish_event("eyebrow_movement", time)

            gaze_baseline = baseline.get("gaze_ratio", _robust_median(calibration_values["gaze_ratio"], 0.5))
            gaze = _classify_gaze(gaze_result.ratio, gaze_baseline, previous_gaze)
            if gaze != "unknown":
                gaze_counter[gaze] += 1
            if gaze == pending_gaze:
                pending_gaze_frames += 1
            else:
                pending_gaze = gaze
                pending_gaze_frames = 1
            if calibration is not None and pending_gaze_frames == confirm_frames and gaze != previous_gaze:
                if previous_gaze != "unknown" and gaze != "unknown":
                    gaze_changes += 1
                    events.append(TimelineEvent(time=round(time, 2), type="gaze_change", description=f"Изменение направления взгляда: {previous_gaze} → {gaze}", value=gaze))
                previous_gaze = gaze

            poses.append(pose)
            if previous_pose is not None:
                delta = (np.array(pose) - np.array(previous_pose) + 180.0) % 360.0 - 180.0
                movement = float(np.linalg.norm(delta))
                head_moves.append(movement)
                pose_baseline = np.array([baseline.get("yaw", pose[0]), baseline.get("pitch", pose[1]), baseline.get("roll", pose[2])])
                pose_delta = (np.array(pose) - pose_baseline + 180.0) % 360.0 - 180.0
                deviation = float(np.linalg.norm(pose_delta))
                head_desired = deviation > (HEAD_EXIT_DEGREES if head_state.active else HEAD_ENTER_DEGREES)
                if calibration is not None:
                    head_transition = head_state.update(head_desired, confirm_frames)
                    if head_transition is True:
                        event = TimelineEvent(time=round(time, 2), type="head_movement", description="Устойчивое отклонение положения головы от baseline", value=round(deviation, 2))
                        events.append(event); active_event["head_movement"] = event
                    elif head_transition is False:
                        finish_event("head_movement", time)
            previous_pose = pose

            if previous_points is not None:
                stable_ids = [10, 33, 133, 362, 263, 61, 291, 152, 105, 334]
                change = float(np.mean(np.linalg.norm(points[stable_ids, :2] - previous_points[stable_ids, :2], axis=1)))
                facial_changes.append(change)
                facial_desired = change > (0.012 if facial_state.active else 0.018)
                if calibration is not None:
                    facial_transition = facial_state.update(facial_desired, confirm_frames)
                    if facial_transition is True:
                        event = TimelineEvent(time=round(time, 2), type="facial_change", description="Устойчивое изменение конфигурации лица", value=round(change, 4))
                        events.append(event); active_event["facial_change"] = event
                    elif facial_transition is False:
                        finish_event("facial_change", time)
            previous_points = points
            index += 1
        if calibration is None:
            calibration = build_calibration()
        for event_type in tuple(active_event):
            finish_event(event_type, duration)
        denominator = max(face_frames, 1)
        if poses:
            pose_radians = np.radians(np.array(poses))
            pose_mean = np.degrees(np.arctan2(np.mean(np.sin(pose_radians), axis=0), np.mean(np.cos(pose_radians), axis=0)))
        else:
            pose_mean = np.zeros(3)
        reliable_gaze_frames = sum(gaze_counter.values())
        gaze_distribution = {
            key: round(value / reliable_gaze_frames, 4)
            for key, value in gaze_counter.items()
        } if reliable_gaze_frames else {}
        both_eyes_frames = eye_modes["both_eyes"]
        single_eye_frames = eye_modes["single_eye"]
        reliable_eye_frames = both_eyes_frames + single_eye_frames
        unavailable_eye_frames = eye_modes["unavailable"]
        eye_denominator = max(analyzed, 1)
        reliable_eye_ratio = reliable_eye_frames / eye_denominator
        eye_status = (
            "unavailable" if reliable_eye_frames == 0
            else "reliable" if unavailable_eye_frames == 0
            else "partial"
        )
        limitation_reason = None
        if eye_status != "reliable":
            limitation_reason = (
                "Часть данных глаз и взгляда недоступна или имеет недостаточную надёжность. "
                "Возможные причины: очки, ракурс или поворот головы, частичная видимость глаза, "
                "освещение либо качество изображения. Положение головы измерено отдельно и не "
                "использовалось как направление взгляда."
            )
        eye_tracking = EyeTrackingQuality(
            status=eye_status,
            reliable_frames=reliable_eye_frames,
            reliable_ratio=round(reliable_eye_ratio, 4),
            both_eyes_frames=both_eyes_frames,
            single_eye_frames=single_eye_frames,
            unavailable_frames=unavailable_eye_frames,
            mode_distribution={key: round(value / eye_denominator, 4) for key, value in eye_modes.items()},
            gaze_source_distribution={key: round(value / eye_denominator, 4) for key, value in gaze_sources.items()},
            mean_left_eye_confidence=round(float(np.mean(left_confidences)), 4) if left_confidences else None,
            mean_right_eye_confidence=round(float(np.mean(right_confidences)), 4) if right_confidences else None,
            left_quality_reasons=dict(left_quality_reasons),
            right_quality_reasons=dict(right_quality_reasons),
            limitation_reason=limitation_reason,
        )
        warnings = ["Метрики являются приближенными геометрическими измерениями и зависят от ракурса, света и качества записи."]
        if not calibration.completed:
            warnings.append("Персональная калибровка не завершена: события рассчитаны с резервными порогами.")
        if limitation_reason:
            warnings.append(limitation_reason)
        return VideoMetrics(
            analyzer="mediapipe-face-mesh", duration_seconds=round(duration, 3), source_fps=round(fps, 3), analyzed_fps=round(analyzed_rate, 3),
            frames_total=total, frames_analyzed=analyzed, frames_with_face=face_frames,
            face_presence_ratio=round(face_frames / max(analyzed, 1), 4), blink_count=blink_count,
            eyes_closed_seconds=round(closed_frames / max(analyzed_rate, 1e-6), 3), gaze_distribution=gaze_distribution,
            gaze_changes=gaze_changes, smile_ratio=round(smile_frames / denominator, 4),
            eyebrow_movement_mean=round(float(np.std(brow_values)) if brow_values else 0, 5),
            mouth_open_ratio=round(mouth_open_frames / denominator, 4),
            facial_change_mean=round(float(np.mean(facial_changes)) if facial_changes else 0, 5),
            head_pose_mean={"yaw": round(float(pose_mean[0]), 2), "pitch": round(float(pose_mean[1]), 2), "roll": round(float(pose_mean[2]), 2)},
            head_movement_mean=round(float(np.mean(head_moves)) if head_moves else 0, 3), calibration=calibration,
            eye_tracking=eye_tracking, events=_merge_report_events(events)[:500], warnings=warnings,
        )

    @staticmethod
    def _gaze(points: np.ndarray) -> str:
        return _classify_gaze(_gaze_ratio(points), 0.5)

    @staticmethod
    def _head_pose(points: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
        image_points = np.array([points[i, :2] * [width, height] for i in (1, 152, 33, 263, 61, 291)], dtype=np.float64)
        # Use the same downward-positive Y convention as image coordinates.
        # The previous inverted Y model introduced a near-180° X rotation,
        # reported to users as a pitch around -160° for a frontal head.
        model_points = np.array([(0, 0, 0), (0, 63.6, -12.5), (-43.3, -32.7, -26), (43.3, -32.7, -26), (-28.9, 28.9, -24.1), (28.9, 28.9, -24.1)], dtype=np.float64)
        camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64)
        ok, rotation, _ = cv2.solvePnP(model_points, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return (0.0, 0.0, 0.0)
        matrix, _ = cv2.Rodrigues(rotation)
        angles, *_ = cv2.RQDecomp3x3(matrix)
        # Euler decompositions have equivalent branches separated by 180°.
        # FaceMesh tracks a front-facing face, so expose the principal branch;
        # otherwise a small tilt can appear as e.g. -164° or +177°.
        pitch, yaw, roll = (((float(angle) + 90.0) % 180.0) - 90.0 for angle in angles)
        return (yaw, pitch, roll)
