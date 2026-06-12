# ============================================================================
# HAND DETECTION CONFIGURATION
# ============================================================================
# These values are calibrated using calibrate_hand.py
# Update them by running: python calibrate_hand.py and pressing S

# Distance (mm)
MIN_MM = 1320
MAX_MM = 1550

# HSV Color calibration for human skin
H_MIN, H_MAX = 0, 20
S_MIN, S_MAX = 50, 255
V_MIN, V_MAX = 100, 255

# Minimum pixels to detect hand (0-10000)
MIN_PIXELS = 10

# Morphological operations (0 = disabled)
ERODE_KERNEL = 0
DILATE_KERNEL = 0
