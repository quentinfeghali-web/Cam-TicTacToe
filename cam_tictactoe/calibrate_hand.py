import pyrealsense2 as rs
import numpy as np
import cv2
import os

# --- Load configuration from file ---
def load_config():
    """Load calibration values from detection_config.py"""
    config_file = "detection_config.py"

    # Default values
    defaults = {
        'MIN_MM': 1320,
        'MAX_MM': 1550,
        'H_MIN': 0,
        'H_MAX': 20,
        'S_MIN': 50,
        'S_MAX': 255,
        'V_MIN': 100,
        'V_MAX': 255,
        'MIN_PIXELS': 10,
        'ERODE_KERNEL': 0,
        'DILATE_KERNEL': 0,
    }

    # If file exists, read it
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                # Execute the file to get the values
                exec_globals = {}
                exec(content, exec_globals)

                # Extract values
                for key in defaults:
                    if key in exec_globals:
                        defaults[key] = exec_globals[key]

                print(f"✅ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
            print(f"Using default values")
    else:
        print(f"⚠️  {config_file} not found, using default values")

    return defaults

# Load config at startup
config = load_config()
MIN_MM = config['MIN_MM']
MAX_MM = config['MAX_MM']
H_MIN = config['H_MIN']
H_MAX = config['H_MAX']
S_MIN = config['S_MIN']
S_MAX = config['S_MAX']
V_MIN = config['V_MIN']
V_MAX = config['V_MAX']
MIN_PIXELS = config['MIN_PIXELS']
ERODE_KERNEL = config['ERODE_KERNEL']
DILATE_KERNEL = config['DILATE_KERNEL']

print(f"Current config: Depth={MIN_MM}-{MAX_MM}mm, H={H_MIN}-{H_MAX}, S={S_MIN}-{S_MAX}, V={V_MIN}-{V_MAX}")
print(f"                MIN_PIXELS={MIN_PIXELS}, ERODE={ERODE_KERNEL}, DILATE={DILATE_KERNEL}\n")

# --- Init camera ---
pipeline = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

pipeline.start(cfg)
align = rs.align(rs.stream.color)

# --- Windows ---
cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
cv2.namedWindow("Depth Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Color Mask (HSV)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Combined Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.namedWindow("Calibration Controls", cv2.WINDOW_NORMAL)

# --- Trackbars callback ---
def nothing(x):
    pass

# --- Distance trackbars (loaded from detection_config) ---
cv2.createTrackbar("Depth_min (mm)", "Calibration Controls", MIN_MM, 2000, nothing)
cv2.createTrackbar("Depth_max (mm)", "Calibration Controls", MAX_MM, 2000, nothing)

# --- HSV trackbars (loaded from detection_config) ---
cv2.createTrackbar("H_min", "Calibration Controls", H_MIN, 180, nothing)
cv2.createTrackbar("H_max", "Calibration Controls", H_MAX, 180, nothing)
cv2.createTrackbar("S_min", "Calibration Controls", S_MIN, 255, nothing)
cv2.createTrackbar("S_max", "Calibration Controls", S_MAX, 255, nothing)
cv2.createTrackbar("V_min", "Calibration Controls", V_MIN, 255, nothing)
cv2.createTrackbar("V_max", "Calibration Controls", V_MAX, 255, nothing)

# --- Pixels threshold trackbar (0-10000) ---
cv2.createTrackbar("MIN_PIXELS", "Calibration Controls", MIN_PIXELS, 10000, nothing)

# --- Morphology trackbars (loaded from detection_config) ---
cv2.createTrackbar("Erode", "Calibration Controls", ERODE_KERNEL, 10, nothing)
cv2.createTrackbar("Dilate", "Calibration Controls", DILATE_KERNEL, 10, nothing)

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        # --- Get trackbar values ---
        depth_min = cv2.getTrackbarPos("Depth_min (mm)", "Calibration Controls")
        depth_max = cv2.getTrackbarPos("Depth_max (mm)", "Calibration Controls")
        h_min = cv2.getTrackbarPos("H_min", "Calibration Controls")
        h_max = cv2.getTrackbarPos("H_max", "Calibration Controls")
        s_min = cv2.getTrackbarPos("S_min", "Calibration Controls")
        s_max = cv2.getTrackbarPos("S_max", "Calibration Controls")
        v_min = cv2.getTrackbarPos("V_min", "Calibration Controls")
        v_max = cv2.getTrackbarPos("V_max", "Calibration Controls")
        min_pixels = cv2.getTrackbarPos("MIN_PIXELS", "Calibration Controls")
        erode_val = cv2.getTrackbarPos("Erode", "Calibration Controls")
        dilate_val = cv2.getTrackbarPos("Dilate", "Calibration Controls")

        # --- Depth mask ---
        depth_mask = (depth >= depth_min) & (depth <= depth_max)

        # --- Color mask (HSV) ---
        h, s, v = cv2.split(hsv)
        color_mask = (h >= h_min) & (h <= h_max) & (s >= s_min) & (s <= s_max) & (v >= v_min) & (v <= v_max)

        # --- Combined mask ---
        combined_mask = depth_mask & color_mask

        # --- Morphological operations ---
        if erode_val > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_val * 2 + 1, erode_val * 2 + 1))
            combined_mask = cv2.erode(combined_mask.astype(np.uint8), kernel, iterations=1)

        if dilate_val > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_val * 2 + 1, dilate_val * 2 + 1))
            combined_mask = cv2.dilate(combined_mask.astype(np.uint8), kernel, iterations=1)

        combined_mask = combined_mask.astype(bool)

        # --- Count pixels ---
        depth_count = int(np.sum(depth_mask))
        color_count = int(np.sum(color_mask))
        combined_count = int(np.sum(combined_mask))

        # --- Visualizations ---
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        depth_vis[depth_mask] = [0, 255, 0]  # Highlight detection zone in green

        color_mask_vis = np.zeros_like(color)
        color_mask_vis[color_mask] = color[color_mask]

        combined_vis = np.zeros_like(color)
        combined_vis[combined_mask] = color[combined_mask]

        result = color.copy()
        result[combined_mask] = [0, 255, 0]  # Green where hand detected

        # --- Info overlay ---
        info_canvas = np.zeros((310, 600, 3), dtype=np.uint8)
        y_offset = 25
        cv2.putText(info_canvas, "=== DISTANCE (mm) ===", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 100), 1)
        cv2.putText(info_canvas, f"Depth: {depth_min}-{depth_max}mm  |  Pixels: {depth_count}", (10, y_offset + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(info_canvas, "=== HSV COLOR ===", (10, y_offset + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 200), 1)
        cv2.putText(info_canvas, f"H: {h_min}-{h_max}  S: {s_min}-{s_max}  V: {v_min}-{v_max}", (10, y_offset + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(info_canvas, f"Color pixels: {color_count}", (10, y_offset + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(info_canvas, "=== MORPHOLOGY ===", (10, y_offset + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 150, 100), 1)
        cv2.putText(info_canvas, f"Erode: {erode_val}  |  Dilate: {dilate_val}", (10, y_offset + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(info_canvas, "=== RESULT ===", (10, y_offset + 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 1)
        cv2.putText(info_canvas, f"MIN_PIXELS: {min_pixels}", (10, y_offset + 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        color_status = (0, 255, 0) if combined_count > min_pixels else (0, 0, 255)
        cv2.putText(info_canvas, f"Combined pixels: {combined_count}", (10, y_offset + 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)

        cv2.putText(info_canvas, "Press S to save values, Q to quit", (10, y_offset + 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # --- Display ---
        cv2.imshow("RGB", color)
        cv2.imshow("Depth Mask", depth_vis)
        cv2.imshow("Color Mask (HSV)", color_mask_vis)
        cv2.imshow("Combined Mask", combined_vis)
        cv2.imshow("Result", result)
        cv2.imshow("Calibration Controls", info_canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current values to detection_config.py
            config_content = f"""# ============================================================================
# HAND DETECTION CONFIGURATION
# ============================================================================
# These values are calibrated using calibrate_hand.py
# Update them by running: python calibrate_hand.py and pressing S

# Distance (mm)
MIN_MM = {depth_min}
MAX_MM = {depth_max}

# HSV Color calibration for human skin
H_MIN, H_MAX = {h_min}, {h_max}
S_MIN, S_MAX = {s_min}, {s_max}
V_MIN, V_MAX = {v_min}, {v_max}

# Minimum pixels to detect hand (0-10000)
MIN_PIXELS = {min_pixels}

# Morphological operations (0 = disabled)
ERODE_KERNEL = {erode_val}
DILATE_KERNEL = {dilate_val}
"""

            with open("detection_config.py", "w") as f:
                f.write(config_content)

            print(f"\n{'='*50}")
            print("✅ CALIBRATION SAVED TO detection_config.py")
            print(f"{'='*50}")
            print(f"Depth: {depth_min}-{depth_max}mm")
            print(f"HSV: H={h_min}-{h_max}, S={s_min}-{s_max}, V={v_min}-{v_max}")
            print(f"MIN_PIXELS: {min_pixels}")
            print(f"ERODE: {erode_val}, DILATE: {dilate_val}")
            print(f"Detected pixels: {combined_count}")
            print(f"{'='*50}\n")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
