
import serial
import time

arduino_port = 'COM7'
baud_rate = 9600

# Connecting to Arduino
arduino = serial.Serial(port=arduino_port, baudrate=baud_rate, timeout=1)
time.sleep(10)  # Waiting for the connection to be completely established


def send_command(command):
    arduino.write((command + '\n').encode())  # Send command over serial
    print(f"Sent: {command}")

def spinallf(sec):
    time.sleep(sec)
    send_command("forwardA")
    send_command("forwardB")

def spinallr(sec):
    time.sleep(sec)
    send_command("reverseA")
    send_command("reverseB")

def stopall(sec):
    time.sleep(sec)
    send_command("stopA")
    send_command("stopB")
spinallf(0)
stopall(8)




time.sleep(1000)
# Test sequence
arduino.reset_input_buffer()

# Motor A test
send_command("forwardA")
time.sleep(5)
send_command("stopA")
time.sleep(2)

send_command("reverseA")
time.sleep(5)
send_command("stopA")
time.sleep(2)

# Motor B test
send_command("forwardB")
time.sleep(5)
send_command("stopB")
time.sleep(2)

send_command("reverseB")
time.sleep(5)
send_command("stopB")