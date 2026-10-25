import time
from typing import Optional, Literal
import msvcrt
import sys
import threading

import serial
import cv2


class StepperController:
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 120.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None

    def open(self) -> None:
        if self._ser and self._ser.is_open:
            return
        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        time.sleep(2.0)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()

    def _send(self, line: str) -> str:
        if not self._ser or not self._ser.is_open:
            self.open()
        command = line.strip() + "\n"
        print(f"→ Sending: {command.strip()}")
        self._ser.write(command.encode("utf-8"))
        response = self._ser.readline().decode("utf-8").strip()
        print(f"← Received: {response}")
        return response

    def _spin(self, direction: Literal["CW", "CCW"], motor: Literal["M1", "M2", "BOTH"], steps: int) -> str:
        if steps < 0:
            steps = -steps
        return self._send(f"SPIN_{direction}:{motor}:{steps}")

    # Motor 1 control
    def spin_down_m1(self, steps: int) -> str:
        """Rotate motor 1 clockwise"""
        return self._spin("CW", "M1", steps)

    def spin_up_m1(self, steps: int) -> str:
        """Rotate motor 1 counter-clockwise"""
        return self._spin("CCW", "M1", steps)

    def spin_up_m1_until_switch(self, max_steps: int = 200000) -> str:
        """Rotate motor 1 counter-clockwise until the limit switch is pressed"""
        if max_steps <= 0:
            max_steps = 200000
        return self._send(f"SPIN_UP_TO_SWITCH:M1:{max_steps}")

    def spin_down_m1_until_switch(self, max_steps: int = 200000) -> str:
        """Rotate motor 1 clockwise until the limit switch is pressed"""
        if max_steps <= 0:
            max_steps = 200000
        return self._send(f"SPIN_DOWN_TO_SWITCH:M1:{max_steps}")

    # Motor 2 control
    def spin_down_m2(self, steps: int) -> str:
        """Rotate motor 2 clockwise"""
        return self._spin("CW", "M2", steps)

    def spin_up_m2(self, steps: int) -> str:
        """Rotate motor 2 counter-clockwise"""
        return self._spin("CCW", "M2", steps)

    # Both motors control
    def spin_down_both(self, steps: int) -> str:
        """Rotate both motors clockwise simultaneously"""
        return self._spin("CW", "BOTH", steps)

    def spin_up_both(self, steps: int) -> str:
        """Rotate both motors counter-clockwise simultaneously"""
        return self._spin("CCW", "BOTH", steps)

    # Legacy methods for backward compatibility
    def spin_down(self, steps: int) -> str:
        """Rotate motor 1 clockwise (legacy)"""
        return self.spin_down_m1(steps)

    def spin_up(self, steps: int) -> str:
        """Rotate motor 1 counter-clockwise (legacy)"""
        return self.spin_up_m1(steps)

    def sleep(self, milliseconds: int) -> str:
        if milliseconds < 0:
            milliseconds = 0
        return self._send(f"SLEEP:{milliseconds}")

    # TT Motor control (L298N driver)
    def tt_motor_forward(self, speed: Literal["SLOW", "MEDIUM", "FAST"] = "MEDIUM") -> str:
        """Run TT motor forward at specified speed"""
        return self._send(f"TT_FORWARD:{speed}")

    def tt_motor_reverse(self, speed: Literal["SLOW", "MEDIUM", "FAST"] = "MEDIUM") -> str:
        """Run TT motor in reverse at specified speed"""
        return self._send(f"TT_REVERSE:{speed}")

    def tt_motor_stop(self) -> str:
        """Stop TT motor"""
        return self._send("TT_STOP")

    def is_switch_pressed(self) -> bool:
        """Check if limit switch is pressed"""
        response = self._send("GET_SWITCH")
        return response == "PRESSED"

    # MG90S servo control
    def servo1_move(self, angle: int) -> str:
        """Move servo 1 by signed offset from center (-90 to 90). Negative = down, positive = up."""
        return self._send(f"SERVO1:{angle}")

    def servo2_move(self, angle: int) -> str:
        """Move servo 2 by signed offset from center (-90 to 90). Negative = down, positive = up."""
        return self._send(f"SERVO2:{angle}")

    def servo1_move_abs(self, angle: int) -> str:
        """Move servo 1 to absolute angle (0-180). Use servo1_move() for signed offsets."""
        return self._send(f"SERVO1_ABS:{angle}")

    def servo2_move_abs(self, angle: int) -> str:
        """Move servo 2 to absolute angle (0-180). Use servo2_move() for signed offsets."""
        return self._send(f"SERVO2_ABS:{angle}")

    def servo_move(self, angle: int) -> str:
        """Move servo 1 by signed offset from center (-90 to 90). Negative = down, positive = up."""
        return self.servo1_move(angle)

    def servo_rotate_90_and_back(self, delay_seconds: float = 0.5) -> None:
        """Rotate servo 1 up 90 degrees, pause, then down 90 degrees (returns to center)."""
        self.servo1_move(90)
        time.sleep(max(0.0, delay_seconds))
        self.servo1_move(-90)
        time.sleep(max(0.0, delay_seconds))
        self.servo1_move(0)


def check_abort():
    """Check if 'p' key was pressed to abort. Returns True if abort requested."""
    if msvcrt.kbhit():
        key = msvcrt.getch().decode('utf-8').lower()
        if key == 'p':
            return True
    return False


def sleep_with_abort(duration):
    """Sleep for duration seconds, checking for abort every 50ms. Returns True if aborted."""
    start_time = time.time()
    while time.time() - start_time < duration:
        if check_abort():
            print("\n⚠️  ABORT REQUESTED - Stopping sequence...\n")
            return True
        time.sleep(0.05)
    return False


def run_motor_with_abort(motor_func, duration, controller):
    """Run a motor for duration seconds with abort checking. Returns True if aborted."""
    start_time = time.time()
    while time.time() - start_time < duration:
        if check_abort():
            controller.tt_motor_stop()
            print("\n⚠️  ABORT REQUESTED - Stopping sequence...\n")
            return True
        time.sleep(0.05)
    return False


def run_live_camera_feed(stop_event, camera_index: int = 0, window_name: str = "Live Camera Feed"):
    """Show a live camera feed until stop_event is set or user presses q/ESC."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {camera_index}.")
        return

    print("Live camera feed started. Press q in the camera window to stop.")
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            cv2.putText(
                frame,
                "Live Feed - Press q to stop",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                stop_event.set()
                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    controller = StepperController(port="COM8")
    controller.open()
    stop_event = threading.Event()
    camera_thread = threading.Thread(
        target=run_live_camera_feed,
        args=(stop_event, 0, "Live Camera Feed"),
        daemon=True,
    )
    camera_thread.start()
    
    try:
        print("Running small motor test sequence...", flush=True)

        print("Motor 1 down 200 steps", flush=True)
        controller.spin_down_m1(200)
        time.sleep(0.3)

        print("Motor 1 up 200 steps", flush=True)
        controller.spin_up_m1(200)
        time.sleep(0.3)

        print("TT motor forward SLOW", flush=True)
        controller.tt_motor_forward("SLOW")
        time.sleep(0.6)
        controller.tt_motor_stop()
        time.sleep(0.3)

        print("TT motor reverse SLOW", flush=True)
        controller.tt_motor_reverse("SLOW")
        time.sleep(0.6)
        controller.tt_motor_stop()
        time.sleep(0.3)

        print("Servo 2 up and down", flush=True)
        controller.servo2_move(90)
        time.sleep(0.5)
        controller.servo2_move(-90)
        time.sleep(0.5)
        controller.servo2_move(0)
        time.sleep(0.3)

        print("Servo 1 up and down", flush=True)
        controller.servo1_move(90)
        time.sleep(0.5)
        controller.servo1_move(-90)
        time.sleep(0.5)
        controller.servo1_move(0)
        time.sleep(0.3)

        print("Motor 2 up 200 steps", flush=True)
        controller.spin_up_m2(200)
        time.sleep(0.3)

        print("Motor 2 down 200 steps", flush=True)
        controller.spin_down_m2(200)
        time.sleep(0.3)

        print("Motor test sequence complete.", flush=True)
        print("Live camera feed is still running. Press q in the camera window to exit.", flush=True)

        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        try:
            controller.tt_motor_stop()
        except Exception:
            pass

        stop_event.set()
        camera_thread.join(timeout=2.0)
        controller.close()
