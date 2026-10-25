import time
from typing import Optional

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

	def spin_down(self, steps: int) -> str:
		"""Rotate motor clockwise"""
		if steps < 0:
			steps = -steps
		return self._send(f"SPIN_CW:{steps}")

	def spin_up(self, steps: int) -> str:
		"""Rotate motor counter-clockwise"""
		if steps < 0:
			steps = -steps
		return self._send(f"SPIN_CCW:{steps}")

	def sleep(self, milliseconds: int) -> str:
		"""Sleep for specified milliseconds"""
		if milliseconds < 0:
			milliseconds = 0
		return self._send(f"SLEEP:{milliseconds}")


if __name__ == "__main__":
    controller = StepperController(port="COM8")
    controller.open()
    try:
        controller.spin_up(5000)
    finally:
        controller.close()
