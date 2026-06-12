## Cam TicTacToe - Hand Detection Game

## Project Context

This project was developed as part of an internship at the **Primatology Station of Romainmotier** (SUPFP), focused on cognitive research with chimpanzees. The game was designed to be used as an **enrichment and cognitive stimulation tool** for chimpanzees, allowing them to interact with a touchscreen setup using hand detection rather than direct touch, in order to study their problem-solving and learning abilities.

## Prerequisites

```bash
pip install pyrealsense2 opencv-python numpy
```

## How to Use

### 1. Hand Detection Calibration (IMPORTANT)

First, you need to calibrate the detection for your hands:

```bash
python calibrate_hand.py
```

**Instructions:**
- Place your hand in front of the camera at the correct depth (10cm-15cm from the glass)
- Adjust the trackbars until your hand is detected in green
- When satisfied, press **S** to save the values
- Copy the values displayed in the console
- Paste them into `main.py` (lines 24-27)
- Press **Q** to quit

### 2. Launch the Game

```bash
python main.py
```

## How to Play

1. **Two players**: Each player places their hand in front of the camera
2. **3x3 Grid**: The tic-tac-toe grid is displayed on the video feed
3. **Validating a move**:
   - Place your hand in an empty cell
   - Hold for **1 second**
   - A progress bar will appear
   - The move validates automatically
4. **Turn alternation**: Players alternate automatically (X = Player 1, O = Player 2)
5. **Winner**: The game displays who won or if it is a draw

## Controls

| Key | Action |
|-----|--------|
| **R** | Restart a game |
| **Q** | Quit |

## Visual States

- 🟢 **Green** = Player 1 (X)
- 🔵 **Blue** = Player 2 (O)
- 💛 **Yellow** = First detection in a cell
- 🔴 **Red/Pink** = Progress bar for validation

## Parameters to Calibrate

The parameters are located at the top of `main.py`:

```python
# Depth (in mm)
MIN_MM = 1320      # Minimum distance
MAX_MM = 1550      # Maximum distance

# Hue
H_MIN, H_MAX = 0, 20

# Saturation
S_MIN, S_MAX = 50, 255

# Value/Brightness
V_MIN, V_MAX = 100, 255

# Minimum number of pixels to detect
MIN_PIXELS = 10

# Validation time (in seconds)
VALIDATION_TIME = 1.0
```

## Tips

- **Lighting**: Make sure you have good lighting
- **Distance**: Keep your hand 10-15cm from the glass (1320-1550mm)
- **Sensitivity**: If too many false positives, increase `MIN_PIXELS`
- **Validation**: You can change `VALIDATION_TIME` to make the game faster or slower

## Troubleshooting

**Hand not detected:**
- Run `calibrate_hand.py` and adjust the trackbars
- Make sure you are at the correct depth (10-15cm)
- Adjust the HSV values for your skin tone

**Accidental detections:**
- Increase `MIN_PIXELS` to reduce noise
- Refine the HSV parameters to be more selective
- Clean the glass (reflections can confuse the camera)

**Move not validating:**
- Keep your hand still inside the cell
- 1 second may feel long, you can reduce `VALIDATION_TIME`
- Make sure the cell is not already occupied

Have fun! 🎉