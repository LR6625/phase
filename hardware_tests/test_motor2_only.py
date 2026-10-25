import time
import serial
import msvcrt


# ======================
# EDITABLE TEST SETTINGS
# ======================
SERIAL_PORT = "COM8"
SERIAL_BAUDRATE = 9600
SERIAL_TIMEOUT_SECONDS = 20.0

# Movement command mode:
# - "sync": SPIN_CW/SPIN_CCW (blocks until movement completes)
# - "async": SPIN_CW_ASYNC/SPIN_CCW_ASYNC (returns immediately; firmware must support async commands)
MOVE_COMMAND_MODE = "sync"

# Steps sent per keypress.
STEP_CHUNK = 2050
STEP_CHUNK_MIN = 50
STEP_CHUNK_MAX = 10000
STEP_CHUNK_FINE_DELTA = 50
STEP_CHUNK_COARSE_DELTA = 250

# Optional keybinds
QUIT_KEYS = (b"q", b"Q", b"\x1b")

# Firmware speed model values for estimating shaft speed.
# These DO NOT change motor speed by themselves; they are for visibility while testing.
M2_START_DELAY_US = 1200
M2_END_DELAY_US = 500
MOTOR_FULL_STEPS_PER_REV = 200
MICROSTEPS = 32
MICROSTEP_OPTIONS = (1, 2, 4, 8, 16, 32)


def estimate_step_rate_hz(step_delay_us):
    if step_delay_us <= 0:
        return 0.0
    # One full pulse period is HIGH + LOW => 2 * step_delay_us.
    return 1_000_000.0 / (2.0 * float(step_delay_us))


def estimate_rpm(step_rate_hz, full_steps_per_rev=MOTOR_FULL_STEPS_PER_REV, microsteps=MICROSTEPS):
    if full_steps_per_rev <= 0 or microsteps <= 0:
        return 0.0
    steps_per_rev = float(full_steps_per_rev) * float(microsteps)
    return (float(step_rate_hz) / steps_per_rev) * 60.0


def print_key_controls():
    print("Runtime controls:", flush=True)
    print("- Left/Right arrows: move M2 down/up", flush=True)
    print("- + / - : fine adjust STEP_CHUNK (+/- 50)", flush=True)
    print("- ] / [ : coarse adjust STEP_CHUNK (+/- 250)", flush=True)
    print("- m : toggle command mode (sync/async)", flush=True)
    print("- 1 / 2 : decrease/increase END delay model (us)", flush=True)
    print("- 3 / 4 : decrease/increase START delay model (us)", flush=True)
    print("- 5 : cycle MICROSTEPS model", flush=True)
    print("- h : print controls and speed guidance", flush=True)
    print("- q or ESC: quit", flush=True)
    print("", flush=True)


def print_speed_guidance(config):
    start_delay = config["m2_start_delay_us"]
    end_delay = config["m2_end_delay_us"]
    full_steps = config["motor_full_steps_per_rev"]
    microsteps = config["microsteps"]

    start_rate = estimate_step_rate_hz(start_delay)
    end_rate = estimate_step_rate_hz(end_delay)
    start_rpm = estimate_rpm(start_rate, full_steps, microsteps)
    end_rpm = estimate_rpm(end_rate, full_steps, microsteps)

    print("NEMA17 light-load practical range (linear drive):", flush=True)
    print("- Start testing around 150 to 300 full-step RPM equivalent.", flush=True)
    print("- Typical upper practical range: ~600 to 1000 full-step RPM equivalent, depending on driver current and mechanics.", flush=True)
    print("- If you hear grinding/stalling, back off speed or increase acceleration ramp time.", flush=True)
    print("", flush=True)
    print("Using current firmware timing model:", flush=True)
    print(f"- Ramp start delay: {start_delay} us -> {start_rate:.1f} steps/s -> {start_rpm:.1f} RPM (microstep shaft RPM)", flush=True)
    print(f"- Ramp end delay:   {end_delay} us -> {end_rate:.1f} steps/s -> {end_rpm:.1f} RPM (microstep shaft RPM)", flush=True)
    print(f"- Steps/rev model: {full_steps} full-steps * {microsteps} microsteps", flush=True)
    print("", flush=True)


def print_runtime_config(config):
    print("Current runtime config:", flush=True)
    print(f"- Command mode: {config['command_mode']}", flush=True)
    print(f"- Step chunk: {config['step_chunk']}", flush=True)
    print(f"- Start delay model: {config['m2_start_delay_us']} us", flush=True)
    print(f"- End delay model: {config['m2_end_delay_us']} us", flush=True)
    print(f"- Full steps/rev model: {config['motor_full_steps_per_rev']}", flush=True)
    print(f"- Microsteps model: {config['microsteps']}", flush=True)
    print("", flush=True)


class Motor2Tester:
    def __init__(self, port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT_SECONDS):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def open(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        time.sleep(2.0)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def send(self, cmd):
        line = cmd.strip() + "\n"
        print(f"> {cmd}", flush=True)
        self.ser.write(line.encode("utf-8"))
        resp = self.ser.readline().decode("utf-8", errors="ignore").strip()
        print(f"< {resp}", flush=True)
        return resp


def run_interactive(port=SERIAL_PORT, step_chunk=STEP_CHUNK, command_mode=MOVE_COMMAND_MODE):
    tester = Motor2Tester(port=port)
    tester.open()
    try:
        config = {
            "command_mode": command_mode.lower(),
            "step_chunk": int(step_chunk),
            "m2_start_delay_us": int(M2_START_DELAY_US),
            "m2_end_delay_us": int(M2_END_DELAY_US),
            "motor_full_steps_per_rev": int(MOTOR_FULL_STEPS_PER_REV),
            "microsteps": int(MICROSTEPS),
        }

        print("Motor2 interactive control started.", flush=True)
        print_key_controls()
        print_runtime_config(config)
        print_speed_guidance(config)

        while True:
            key = msvcrt.getch()

            if key in QUIT_KEYS:
                print("Stopping interactive motor2 control.", flush=True)
                break

            if key == b"+":
                config["step_chunk"] = min(STEP_CHUNK_MAX, config["step_chunk"] + STEP_CHUNK_FINE_DELTA)
                print_runtime_config(config)
                continue
            if key == b"-":
                config["step_chunk"] = max(STEP_CHUNK_MIN, config["step_chunk"] - STEP_CHUNK_FINE_DELTA)
                print_runtime_config(config)
                continue
            if key == b"]":
                config["step_chunk"] = min(STEP_CHUNK_MAX, config["step_chunk"] + STEP_CHUNK_COARSE_DELTA)
                print_runtime_config(config)
                continue
            if key == b"[":
                config["step_chunk"] = max(STEP_CHUNK_MIN, config["step_chunk"] - STEP_CHUNK_COARSE_DELTA)
                print_runtime_config(config)
                continue
            if key in (b"m", b"M"):
                config["command_mode"] = "async" if config["command_mode"] == "sync" else "sync"
                print_runtime_config(config)
                continue
            if key == b"1":
                config["m2_end_delay_us"] = max(100, config["m2_end_delay_us"] - 50)
                print_speed_guidance(config)
                continue
            if key == b"2":
                config["m2_end_delay_us"] = min(5000, config["m2_end_delay_us"] + 50)
                print_speed_guidance(config)
                continue
            if key == b"3":
                config["m2_start_delay_us"] = max(config["m2_end_delay_us"], config["m2_start_delay_us"] - 100)
                print_speed_guidance(config)
                continue
            if key == b"4":
                config["m2_start_delay_us"] = min(10000, config["m2_start_delay_us"] + 100)
                print_speed_guidance(config)
                continue
            if key == b"5":
                idx = MICROSTEP_OPTIONS.index(config["microsteps"]) if config["microsteps"] in MICROSTEP_OPTIONS else 0
                config["microsteps"] = MICROSTEP_OPTIONS[(idx + 1) % len(MICROSTEP_OPTIONS)]
                print_speed_guidance(config)
                continue
            if key in (b"h", b"H"):
                print_key_controls()
                print_runtime_config(config)
                print_speed_guidance(config)
                continue

            if key in (b"\x00", b"\xe0"):
                arrow = msvcrt.getch()
                if arrow == b"K":
                    if config["command_mode"] == "async":
                        tester.send(f"SPIN_CW_ASYNC:M2:{config['step_chunk']}")
                    else:
                        tester.send(f"SPIN_CW:M2:{config['step_chunk']}")
                elif arrow == b"M":
                    if config["command_mode"] == "async":
                        tester.send(f"SPIN_CCW_ASYNC:M2:{config['step_chunk']}")
                    else:
                        tester.send(f"SPIN_CCW:M2:{config['step_chunk']}")
    finally:
        tester.close()


if __name__ == "__main__":
    run_interactive(port=SERIAL_PORT, step_chunk=STEP_CHUNK, command_mode=MOVE_COMMAND_MODE)
