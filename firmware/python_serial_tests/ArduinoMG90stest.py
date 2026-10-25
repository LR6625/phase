import serial
import time

# Configuration
COM_PORT = "COM8"  # Change to your Arduino's COM port
BAUD_RATE = 9600
ser = None  # Global serial connection


def connect():
    """Connect to Arduino"""
    global ser
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"✓ Connected to Arduino on {COM_PORT}")
        return True
    except serial.SerialException as e:
        print(f"✗ Connection error: {e}")
        return False


def rotate(angle):
    """
    Rotate servo to angle (0-180 degrees)
    """
    global ser
    if ser is None or not ser.is_open:
        print("✗ Not connected. Call connect() first.")
        return
    
    angle = max(0, min(180, angle))
    ser.write(f"{angle}\n".encode())
    print(f"Servo at {angle}°")


def sleep(seconds):
    """Pause for specified seconds"""
    time.sleep(seconds)


if __name__ == "__main__":
    connect()
    rotate(90)
    sleep(0.5)
    rotate(-90)
    sleep(0.5)
    rotate(90)
    sleep(0.5)
    rotate(-90)
    sleep(0.5)
    rotate(90)
    sleep(0.5)
    rotate(-90)
    sleep(0.5)
    rotate(90)
    sleep(0.5)
    rotate(-90)
    sleep(0.5)
    rotate(90)
    sleep(0.5)
    rotate(-90)
    sleep(0.5)

