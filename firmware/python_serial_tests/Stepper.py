import time
from typing import Optional, Literal

import serial


class StepperController:
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 2.0) -> None:
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
            raise RuntimeError("Serial port is not open")
        self._ser.write((line.strip() + "\n").encode("utf-8"))
        return self._ser.readline().decode("utf-8").strip()

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


if __name__ == "__main__":
    controller = StepperController(port="COM8")
    controller.open()
    try:
        # Example usage with Motor 1 (pins 2,3) and Motor 2 (pins 6,7)
        controller.spin_down_m1(10000)  # Rotate motor 1
        controller.sleep(500)
        controller.spin_up_m1_until_switch()  # Run motor 1 until the limit switch is pressed
        # controller.spin_up_m2(5000)  # Rotate motor 2
        # controller.sleep(500)
        # controller.spin_up_both(5000)  # Rotate both simultaneously
    finally:
        controller.close()
