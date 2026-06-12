import pyrealsense2 as rs
import numpy as np
import cv2
from time import time
from detection_config import (MIN_MM, MAX_MM, MIN_PIXELS,
                              H_MIN, H_MAX, S_MIN, S_MAX, V_MIN, V_MAX,
                              ERODE_KERNEL, DILATE_KERNEL)

# CAMERA & DETECTION CONFIG
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

# GAME CONFIG
GRID_SIZE = 3
CELL_WIDTH = 848 // GRID_SIZE
CELL_HEIGHT = 480 // GRID_SIZE
VALIDATION_TIME = 1.0  # seconds to hold in cell to validate move

# GAME STATE
board = [[' ' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
current_player = 'X'  # Player 1
game_over = False
winner = None
last_cell = None
cell_entry_time = None

cv2.namedWindow("Cam TicTacToe", cv2.WINDOW_NORMAL)
cv2.namedWindow("Debug - Filters", cv2.WINDOW_NORMAL)


def get_hand_position(depth, color):
    """Detect hand and return (cx, cy) or None"""
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Depth mask
    depth_mask = (depth >= MIN_MM) & (depth <= MAX_MM)

    # Color mask
    color_mask = (h >= H_MIN) & (h <= H_MAX) & (s >= S_MIN) & (s <= S_MAX) & (v >= V_MIN) & (v <= V_MAX)

    # Combined
    mask = depth_mask & color_mask
 
    # Morphological operations
    if ERODE_KERNEL > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_KERNEL * 2 + 1, ERODE_KERNEL * 2 + 1))
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

    if DILATE_KERNEL > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL * 2 + 1, DILATE_KERNEL * 2 + 1))
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    mask = mask.astype(bool)

    if np.sum(mask) < MIN_PIXELS:
        return None, depth_mask, color_mask, mask

    ys, xs = np.where(mask)
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))

    return (cx, cy), depth_mask, color_mask, mask


def get_cell_from_position(cx, cy):
    """Convert pixel position to grid cell (row, col)"""
    col = int(cx // CELL_WIDTH)
    row = int(cy // CELL_HEIGHT)

    if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
        return row, col
    return None


def make_move(row, col, player):
    """Place player mark on board. Returns True if valid move"""
    if board[row][col] == ' ':
        board[row][col] = player
        return True
    return False


def check_winner():
    """Check if there's a winner. Returns 'X', 'O', 'DRAW', or None"""
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] != ' ':
            return row[0]

    # Check columns
    for col in range(GRID_SIZE):
        if board[0][col] == board[1][col] == board[2][col] != ' ':
            return board[0][col]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return board[0][2]

    # Check if board is full
    if all(board[i][j] != ' ' for i in range(GRID_SIZE) for j in range(GRID_SIZE)):
        return 'DRAW'

    return None


def draw_grid(frame):
    """Draw the game grid on frame"""
    for i in range(1, GRID_SIZE):
        # Vertical lines
        cv2.line(frame, (i * CELL_WIDTH, 0), (i * CELL_WIDTH, 480), (200, 200, 200), 2)
        # Horizontal lines
        cv2.line(frame, (0, i * CELL_HEIGHT), (848, i * CELL_HEIGHT), (200, 200, 200), 2)


def draw_board_state(frame):
    """Draw X and O on the grid"""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell = board[row][col]
            if cell != ' ':
                x = col * CELL_WIDTH + CELL_WIDTH // 2
                y = row * CELL_HEIGHT + CELL_HEIGHT // 2

                color = (0, 255, 0) if cell == 'X' else (255, 0, 0)  # Green for X, Blue for O
                cv2.putText(frame, cell, (x - 20, y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 3)


def draw_hand_cursor(frame, cx, cy):
    """Draw cursor where hand is detected"""
    cv2.circle(frame, (cx, cy), 15, (0, 255, 255), 2)
    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)


def draw_info(frame):
    """Draw game info"""
    y_pos = 30

    if not game_over:
        player_text = "PLAYER 1 (X)" if current_player == 'X' else "PLAYER 2 (O)"
        color = (0, 255, 0) if current_player == 'X' else (255, 0, 0)
        cv2.putText(frame, player_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, "Hold hand in cell for 1 sec to play", (10, y_pos + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    else:
        if winner == 'DRAW':
            cv2.putText(frame, "GAME OVER - DRAW!", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
        else:
            winner_text = "PLAYER 1 (X) WINS!" if winner == 'X' else "PLAYER 2 (O) WINS!"
            color = (0, 255, 0) if winner == 'X' else (255, 0, 0)
            cv2.putText(frame, winner_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

        cv2.putText(frame, "Press R to restart, Q to quit", (10, y_pos + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)


def reset_game():
    """Reset game state"""
    global board, current_player, game_over, winner, last_cell, cell_entry_time
    board = [[' ' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    current_player = 'X'
    game_over = False
    winner = None
    last_cell = None
    cell_entry_time = None


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

        # Detect hand
        hand_result = get_hand_position(depth, color)
        hand_pos, depth_mask, color_mask, combined_mask = hand_result

        # Draw grid and board state
        draw_grid(color)
        draw_board_state(color)

        # Create debug visualization
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        color_mask_vis = np.zeros_like(color)
        color_mask_vis[color_mask] = color[color_mask]

        combined_vis = np.zeros_like(color)
        combined_vis[combined_mask] = color[combined_mask]

        # Stack visualizations (2x2 grid)
        top_row = np.hstack([color, depth_vis])
        bottom_row = np.hstack([color_mask_vis, combined_vis])
        debug_vis = np.vstack([top_row, bottom_row])

        # Handle hand detection and game logic
        if hand_pos is not None and not game_over:
            cx, cy = hand_pos
            draw_hand_cursor(color, cx, cy)

            cell = get_cell_from_position(cx, cy)

            if cell:
                row, col = cell

                # First time entering a new cell
                if cell != last_cell:
                    last_cell = cell
                    cell_entry_time = time()
                    # Draw cell highlight when first detected
                    x1 = col * CELL_WIDTH
                    y1 = row * CELL_HEIGHT
                    cv2.rectangle(color, (x1, y1), (x1 + CELL_WIDTH, y1 + CELL_HEIGHT),
                                (255, 255, 0), 3)
                else:
                    # Hand still in same cell, check if validation time reached
                    time_held = time() - cell_entry_time
                    remaining = VALIDATION_TIME - time_held

                    # Draw cell with progress
                    x1 = col * CELL_WIDTH
                    y1 = row * CELL_HEIGHT
                    thickness = 3
                    color_intensity = int(255 * (time_held / VALIDATION_TIME))
                    cell_color = (100, color_intensity, 255)
                    cv2.rectangle(color, (x1, y1), (x1 + CELL_WIDTH, y1 + CELL_HEIGHT),
                                cell_color, thickness)

                    # Progress text
                    cv2.putText(color, f"{remaining:.1f}s",
                               (x1 + CELL_WIDTH // 2 - 30, y1 + CELL_HEIGHT // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # Validation!
                    if time_held >= VALIDATION_TIME:
                        if make_move(row, col, current_player):
                            print(f"[MOVE] Player {current_player} played at ({row}, {col})")

                            # Check for winner
                            result = check_winner()
                            if result:
                                game_over = True
                                winner = result
                                print(f"[GAME OVER] Winner: {winner}")
                            else:
                                # Switch player
                                current_player = 'O' if current_player == 'X' else 'X'

                        # Reset tracking
                        last_cell = None
                        cell_entry_time = None
            else:
                # Hand outside grid
                last_cell = None
                cell_entry_time = None
        else:
            # No hand detected or game over
            last_cell = None
            cell_entry_time = None

        # Draw info
        draw_info(color)

        # Display
        cv2.imshow("Cam TicTacToe", color)
        cv2.imshow("Debug - Filters", debug_vis)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_game()
            print("[GAME] Reset!")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
