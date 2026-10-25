"""
Machine-specific settings for the P.H.A.S.E. sorter.

Edit these three values to match your setup. Nothing else in the project
should need to change to run on a different PC.

These were the values on the original build machine (Windows 11).
"""

# Serial port of the Arduino that drives the motors (see firmware/FinalSketch.ino).
# On Windows, check Device Manager > Ports (COM & LPT). On Linux it would look
# like "/dev/ttyACM0" (note: the main script also uses msvcrt, so it is Windows-only).
SERIAL_PORT = "COM8"

# Must match Serial.begin(...) in the Arduino sketch.
BAUD_RATE = 9600

# OpenCV camera index for the USB camera looking at the card.
# 0 is usually the first camera Windows finds; try 1 if you also have a webcam.
CAMERA_INDEX = 0
