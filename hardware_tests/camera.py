from ArducamEvkSDK import Camera, Param
import cv2

try:
    import ArducamEvkSDK
    print("ArducamEvkSDK imported successfully.")
except ImportError as e:
    print("ImportError:", e)

try:
    print("ArducamEvkSDK attributes:", dir(ArducamEvkSDK))
    # Look for something like DeviceManager, enumerate_devices, get_device_list, etc.
    # Example:
    # devices = ArducamEvkSDK.DeviceManager().enumerate()
    # print(f"Available devices: {devices}")
    # devices = ArducamEvkSDK.list_devices()
    # print(f"Available devices: {devices}")
    # if len(devices) < 2:
    #     print("❌ Less than two cameras detected. Please connect a second camera.")
    # else:
    #     param = Param()
    #     param.device = devices[1]  # Select the second camera (index 1)
    #     print(f"Selected device: {param.device}")

    #     cam = Camera()
    #     cam.open(param)
    #     print("Camera opened successfully.")

    #     # Set manual exposure
    #     cam.set_auto_exposure(False)
    #     if not cam.get_auto_exposure():
    #         print("Auto exposure disabled (manual mode).")
    #     else:
    #         print("Failed to disable auto exposure.")

    #     cam.set_exposure(-8)
    #     if cam.get_exposure() == -8:
    #         print("Exposure set to -8.")
    #     else:
    #         print(f"Exposure not set correctly, current: {cam.get_exposure()}")

    #     # Set ISO if supported
    #     try:
    #         cam.set_iso(100)
    #         if cam.get_iso() == 100:
    #             print("ISO set to 100.")
    #         else:
    #             print(f"ISO not set correctly, current: {cam.get_iso()}")
    #     except AttributeError:
    #         print("ISO setting not supported.")

    #     # Set white balance
    #     cam.set_auto_white_balance(False)
    #     if not cam.get_auto_white_balance():
    #         print("Auto white balance disabled.")
    #     else:
    #         print("Failed to disable auto white balance.")

    #     try:
    #         cam.set_white_balance_temperature(6500)
    #         if cam.get_white_balance_temperature() == 6500:
    #             print("White balance temperature set to 6500K.")
    #         else:
    #             print(f"White balance temperature not set correctly, current: {cam.get_white_balance_temperature()}")
    #     except AttributeError:
    #         print("White balance temperature setting not supported.")

    #     while True:
    #         frame = cam.capture()
    #         if frame is None:
    #             print("❌ Failed to grab frame")
    #             break
    #         cv2.imshow("Arducam Live Feed", frame)
    #         # Press 'q' to quit
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    #     cam.close()
    #     cv2.destroyAllWindows()

    # Enumerate devices using DeviceList
    device_list = ArducamEvkSDK.DeviceList()
    devices = device_list.get()  # or device_list.get_devices(), check which works
    print(f"Available devices: {devices}")
    if len(devices) < 2:
        print("❌ Less than two cameras detected. Please connect a second camera.")
    else:
        param = Param()
        param.device = devices[1]  # Select the second camera (index 1)
        print(f"Selected device: {param.device}")

        cam = Camera()
        cam.open(param)
        print("Camera opened successfully.")

        # Set manual exposure
        cam.set_auto_exposure(False)
        if not cam.get_auto_exposure():
            print("Auto exposure disabled (manual mode).")
        else:
            print("Failed to disable auto exposure.")

        cam.set_exposure(-8)
        if cam.get_exposure() == -8:
            print("Exposure set to -8.")
        else:
            print(f"Exposure not set correctly, current: {cam.get_exposure()}")

        # Set ISO if supported
        try:
            cam.set_iso(100)
            if cam.get_iso() == 100:
                print("ISO set to 100.")
            else:
                print(f"ISO not set correctly, current: {cam.get_iso()}")
        except AttributeError:
            print("ISO setting not supported.")

        # Set white balance
        cam.set_auto_white_balance(False)
        if not cam.get_auto_white_balance():
            print("Auto white balance disabled.")
        else:
            print("Failed to disable auto white balance.")

        try:
            cam.set_white_balance_temperature(6500)
            if cam.get_white_balance_temperature() == 6500:
                print("White balance temperature set to 6500K.")
            else:
                print(f"White balance temperature not set correctly, current: {cam.get_white_balance_temperature()}")
        except AttributeError:
            print("White balance temperature setting not supported.")

        while True:
            frame = cam.capture()
            if frame is None:
                print("❌ Failed to grab frame")
                break
            cv2.imshow("Arducam Live Feed", frame)
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.close()
        cv2.destroyAllWindows()

except Exception as e:
    print(f"❌ Could not open camera: {e}")