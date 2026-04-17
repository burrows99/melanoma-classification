import pyautogui
import time
import logging
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

RADIUS = 30
STEPS = 48
CIRCLE_DURATION_SECONDS = 2.0
LOOP_INTERVAL_SECONDS = 10.0

# Distribute the full 2-second movement budget across all circle steps.
STEP_DURATION = CIRCLE_DURATION_SECONDS / STEPS

while True:
    try:
        loop_start = time.monotonic()
        center_x, center_y = pyautogui.position()
        logging.info("Starting circle around (%d, %d)", center_x, center_y)

        previous_x = 0
        previous_y = 0
        circle_aborted = False
        for step in range(STEPS):
            theta = 2 * math.pi * (step / STEPS)
            current_x = int(RADIUS * math.cos(theta))
            current_y = int(RADIUS * math.sin(theta))
            delta_x = current_x - previous_x
            delta_y = current_y - previous_y

            start_x, start_y = pyautogui.position()
            pyautogui.moveRel(delta_x, delta_y, duration=STEP_DURATION)
            end_x, end_y = pyautogui.position()
            expected_x = start_x + delta_x
            expected_y = start_y + delta_y
            if abs(end_x - expected_x) > 1 or abs(end_y - expected_y) > 1:
                logging.info("Mouse moved by user; stopping current circle")
                circle_aborted = True
                break

            previous_x = current_x
            previous_y = current_y

        end_x, end_y = pyautogui.position()
        if circle_aborted:
            logging.info("Circle aborted at (%d, %d)", end_x, end_y)
        else:
            logging.info("Completed circle at (%d, %d)", end_x, end_y)
        elapsed = time.monotonic() - loop_start
        remaining = max(0.0, LOOP_INTERVAL_SECONDS - elapsed)
        time.sleep(remaining)
    except KeyboardInterrupt:
        logging.info("Stopped by user")
        break
    except Exception as exc:
        logging.exception("Mouse movement failed: %s", exc)
        time.sleep(1)
