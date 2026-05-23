from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator
import os 
import cv2
from PIL import Image

def open_video(video_path: str) -> cv2.VideoCapture:
    """Open ``video_path`` with OpenCV, raising a diagnostic error on failure.

    ``cv2.VideoCapture`` reports failure with a single bool, which makes
    "Cannot open video" useless for telling apart a missing file, an empty or
    truncated download, an un-pulled Git LFS pointer, or a genuine
    codec/backend problem. This inspects the file and raises a message that
    names the actual cause.
    """
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        return cap
    cap.release()

    if not os.path.exists(video_path):
        raise RuntimeError(
            f"Cannot open video: {video_path} — file does not exist "
            f"(cwd={os.getcwd()})."
        )
    if os.path.isdir(video_path):
        raise RuntimeError(f"Cannot open video: {video_path} — path is a directory.")
    if not os.access(video_path, os.R_OK):
        raise RuntimeError(f"Cannot open video: {video_path} — file is not readable.")

    size = os.path.getsize(video_path)
    if size == 0:
        raise RuntimeError(f"Cannot open video: {video_path} — file is empty (0 bytes).")

    with open(video_path, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            f"Cannot open video: {video_path} — file is an unresolved Git LFS "
            f"pointer ({size} bytes). Run `git lfs pull` to download the content."
        )

    raise RuntimeError(
        f"Cannot open video: {video_path} — file exists ({size} bytes) but "
        f"OpenCV could not decode it. The codec may be unsupported by this "
        f"OpenCV build (missing FFMPEG backend) or the file may be corrupt."
    )
# ── output primitives ─────────────────────────────────────────────────────────

@dataclass
class Detection:
    """One detected/tracked object in a single frame.

    Attributes:
        bbox:       Bounding box in pixel coordinates (x1, y1, x2, y2).
        category:   Object class label, e.g. "car", "person", "emergency_vehicle".
        confidence: Detection confidence in [0, 1].
        track_id:   Persistent identity across frames; None for stateless detectors.
        caption:    Short status description, e.g. "injured, sitting", "crashed".
                    None when the pipeline does not produce captions.
    """

    bbox: tuple[float, float, float, float]
    category: str
    confidence: float = 1.0
    track_id: int | None = None
    caption: str | None = None


@dataclass
class FrameResult:
    """All detections produced by a pipeline for one video frame.

    Attributes:
        detections: List of Detection objects, one per tracked/detected instance.
        frame_idx:  Zero-based index of the frame within the current video.
        metadata:   Optional model-specific extras (e.g. raw VLM output, timing).
    """

    detections: list[Detection]
    frame_idx: int
    metadata: dict = field(default_factory=dict)


# ── abstract pipeline ─────────────────────────────────────────────────────────

class VistaPipeline(ABC):
    """Abstract base class for all VISTA video-understanding pipelines.

    Subclasses implement ``forward`` to process a single frame and return a
    ``FrameResult``. State (tracker database, caption history, etc.) lives
    entirely inside the subclass; callers never touch it directly.

    Typical usage::

        pipeline = MyPipeline(...)
        for video_path in videos:
            pipeline.reset()
            for frame, idx in iter_frames(video_path):
                result = pipeline(frame, idx)
                consume(result)

    For convenience, ``process_video`` wraps this loop and yields
    ``FrameResult`` objects one at a time.
    """

    # ── core interface ────────────────────────────────────────────────────────

    @abstractmethod
    def forward(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        """Process a single frame and return all active detections.

        Args:
            frame:     The current video frame as a PIL RGB image.
            frame_idx: Zero-based index of this frame in the current video.
                       Pipelines may use it to implement stride-based logic
                       (e.g. run the VLM only every N frames).

        Returns:
            A FrameResult containing zero or more Detection objects.
            Every Detection must have a valid ``bbox`` and ``category``.
            ``track_id`` and ``caption`` are optional but strongly encouraged
            for competitive leaderboard submissions.
        """

    def reset(self) -> None:
        """Clear all stateful components before processing a new video.

        Override this in subclasses that maintain a tracker database, VLM
        history buffer, or any other per-video state. The default is a no-op,
        which is correct for stateless (single-frame) detectors.
        """

    # ── convenience ───────────────────────────────────────────────────────────

    def __call__(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        return self.forward(frame, frame_idx)

    def process_video(
        self,
        video_path: str,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> Iterator[FrameResult]:
        """Iterate over a video file, yielding a FrameResult per frame.

        Calls ``reset()`` once before the loop so that this method can be
        called multiple times on different videos without leaking state.

        Args:
            video_path:  Path to the video file (any format supported by OpenCV).
            start_frame: Index of the first frame to process (inclusive).
            end_frame:   Index of the last frame to process (exclusive).
                         None means process until the video ends.

        Yields:
            FrameResult for each frame in [start_frame, end_frame).
        """
        self.reset()

        cap = open_video(video_path)

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        try:
            while True:
                if end_frame is not None and frame_idx >= end_frame:
                    break
                ret, bgr = cap.read()
                if not ret:
                    break
                frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                yield self.forward(frame, frame_idx)
                frame_idx += 1
        finally:
            cap.release()
