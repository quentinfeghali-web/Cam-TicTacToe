import numpy as np
import cv2
from time import time


class PostMoveCalibrator:
    """
    After each move, collects frames for CALIBRATION_DURATION seconds,
    builds a background HSV reference, and exposes a mask to suppress
    pixels that were already present before the hand appeared.
    """

    CALIBRATION_DURATION = 1.0  # seconds

    def __init__(self):
        self.active = False
        self.start_time = None
        self.hsv_samples = []       # raw HSV frames collected during calibration
        self._reference = None      # scalar stats dict, set after _finish()
        self._bg_mean = None        # (H, W, 3) float32 — per-pixel HSV mean
        self._bg_std = None         # (H, W, 3) float32 — per-pixel HSV std
        self._suppress_mask = None  # (H, W) bool — pixels to always ignore

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trigger(self):
        """Call immediately after a move is registered."""
        self.active = True
        self.start_time = time()
        self.hsv_samples = []

    @property
    def is_active(self):
        return self.active

    def feed(self, color_frame: np.ndarray):
        """
        Feed every camera frame while calibration is active.
        Returns (still_calibrating: bool, overlay_frame: np.ndarray).
        The overlay has a progress banner so the player sees the countdown.
        """
        if not self.active:
            return False, color_frame

        elapsed = time() - self.start_time
        remaining = self.CALIBRATION_DURATION - elapsed

        # Collect HSV snapshot
        hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        self.hsv_samples.append(hsv)

        # Draw overlay
        overlay = color_frame.copy()
        h, w = overlay.shape[:2]

        banner_h = 50
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, color_frame, 0.4, 0, overlay)

        progress = min(elapsed / self.CALIBRATION_DURATION, 1.0)
        bar_w = int(w * progress)
        cv2.rectangle(overlay, (0, h - 6), (bar_w, h), (0, 200, 255), -1)

        cv2.putText(overlay,
                    f"Calibrating... {remaining:.1f}s",
                    (10, h - banner_h + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        if elapsed >= self.CALIBRATION_DURATION:
            self._finish()
            return False, overlay

        return True, overlay

    def filter_mask(self, hand_mask: np.ndarray) -> np.ndarray:
        """
        Remove background pixels from a hand detection mask.
        hand_mask: (H, W) bool or uint8
        Returns a cleaned (H, W) bool mask with redundant pixels removed.

        If no calibration has run yet, returns the mask unchanged.
        """
        if self._suppress_mask is None:
            return hand_mask.astype(bool)

        # Suppress pixels that belong to the static background
        cleaned = hand_mask.astype(bool) & ~self._suppress_mask
        return cleaned

    def get_hsv_reference(self) -> dict | None:
        """
        Returns scalar HSV stats from the last calibration, or None.
        Keys: mean_h, mean_s, mean_v, std_h, std_s, std_v, frame_mean.
        """
        return self._reference

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _finish(self):
        self.active = False
        if not self.hsv_samples:
            return

        stack = np.stack(self.hsv_samples, axis=0)   # (N, H, W, 3)
        self._bg_mean = np.mean(stack, axis=0)        # (H, W, 3)
        self._bg_std  = np.std(stack, axis=0)         # (H, W, 3)

        # Build suppress mask: pixels whose HSV variance is low are static
        # background — they should never count as "hand".
        # Threshold: a pixel is background if std across all three channels
        # is below STD_THRESHOLD (tune if needed).
        STD_THRESHOLD = 5.0
        total_std = (self._bg_std[..., 0] +
                     self._bg_std[..., 1] +
                     self._bg_std[..., 2])
        self._suppress_mask = total_std < STD_THRESHOLD   # (H, W) bool

        # Scalar summaries
        self._reference = {
            "mean_h":     float(np.mean(self._bg_mean[..., 0])),
            "mean_s":     float(np.mean(self._bg_mean[..., 1])),
            "mean_v":     float(np.mean(self._bg_mean[..., 2])),
            "std_h":      float(np.mean(self._bg_std[..., 0])),
            "std_s":      float(np.mean(self._bg_std[..., 1])),
            "std_v":      float(np.mean(self._bg_std[..., 2])),
            "frame_mean": self._bg_mean,
        }

        suppressed_px = int(np.sum(self._suppress_mask))
        print(f"[CALIBRATION] Done. Suppressing {suppressed_px} background pixels. "
              f"H={self._reference['mean_h']:.1f}±{self._reference['std_h']:.1f}  "
              f"S={self._reference['mean_s']:.1f}±{self._reference['std_s']:.1f}  "
              f"V={self._reference['mean_v']:.1f}±{self._reference['std_v']:.1f}")

        self.hsv_samples = []