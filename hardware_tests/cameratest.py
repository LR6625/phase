import cv2
import time
import argparse

# Try importing Arducam SDK; fall back to OpenCV capture if unavailable
try:
    import ArducamEvkSDK
    from ArducamEvkSDK import Camera, Param
    HAVE_ARDUCAM = True
except Exception:
    ArducamEvkSDK = None
    Camera = None
    Param = None
    HAVE_ARDUCAM = False


class CamWrapper:
    def __init__(self, index=0, force_arducam=False, device_id=None):
        self.using_arducam = False
        self.cap = None
        self.cam = None

        if HAVE_ARDUCAM:
            print('Arducam SDK available, attempting enumeration...')
            try:
                # Try multiple ways to get device list from SDK
                devs = None
                try:
                    dl = ArducamEvkSDK.DeviceList()
                    print('DeviceList object methods:', [m for m in dir(dl) if not m.startswith('_')])
                    for attr in ('get', 'get_devices', 'get_device_list', 'getList', 'getDeviceList', 'devices'):
                        if hasattr(dl, attr):
                            member = getattr(dl, attr)
                            try:
                                devs = member()
                                print(f'Called DeviceList.{attr}() ->', devs)
                            except Exception:
                                devs = member
                                print(f'Got DeviceList.{attr} attribute ->', devs)
                            break
                    if devs is None:
                        try:
                            devs = list(dl)
                            print('Iterated DeviceList ->', devs)
                        except Exception:
                            devs = None
                except Exception as e:
                    print('DeviceList access failed:', e)
                    devs = None

                # Try module-level helpers
                if (not devs or len(devs) == 0) and hasattr(ArducamEvkSDK, 'enumerate_devices'):
                    try:
                        devs = ArducamEvkSDK.enumerate_devices()
                        print('ArducamEvkSDK.enumerate_devices() ->', devs)
                    except Exception as e:
                        print('enumerate_devices failed:', e)
                if (not devs or len(devs) == 0) and hasattr(ArducamEvkSDK, 'list_devices'):
                    try:
                        devs = ArducamEvkSDK.list_devices()
                        print('ArducamEvkSDK.list_devices() ->', devs)
                    except Exception as e:
                        print('list_devices failed:', e)

                # Normalize devs to list
                if devs is None:
                    devs = []
                else:
                    try:
                        devs = list(devs)
                    except Exception:
                        devs = [devs]

                print(f"Arducam: found {len(devs)} device(s): {devs}")

                # Auto-select device by provided device_id or known Arducam name
                target_names = []
                if device_id:
                    target_names.append(device_id)
                target_names.append('Arducam_MP12')
                selected_index = None
                for name in target_names:
                    for i, d in enumerate(devs):
                        try:
                            if name in str(d):
                                selected_index = i
                                print(f"Auto-selected device matching '{name}' at index {i}: {d}")
                                break
                        except Exception:
                            continue
                    if selected_index is not None:
                        break
                if selected_index is not None:
                    index = selected_index

                # If the user provided a device_id but it wasn't matched in list, keep it for forced open
                opened = False
                chosen = None
                if len(devs) > index:
                    p = Param()
                    chosen = devs[index]
                    # Try setting param.device in a few ways
                    assigned = False
                    for candidate in (chosen, getattr(chosen, 'device', None), getattr(chosen, 'id', None), getattr(chosen, 'index', None)):
                        if candidate is None:
                            continue
                        try:
                            p.device = candidate
                            assigned = True
                            print('Assigned Param.device =', candidate)
                            break
                        except Exception:
                            pass

                    # If none assigned, try to assign the raw chosen object
                    if not assigned:
                        try:
                            p.device = chosen
                            assigned = True
                            print('Assigned Param.device = chosen object')
                        except Exception:
                            pass

                    cam = Camera()
                    print('Camera object methods:', [m for m in dir(cam) if not m.startswith('_')])

                    # Try various open methods
                    for open_try in ('open', 'open_device', 'open_by_id', 'open_by_index', 'open_device_by_id'):
                        if hasattr(cam, open_try):
                            try:
                                member = getattr(cam, open_try)
                                try:
                                    member(p)
                                    print(f'Opened camera using cam.{open_try}(Param)')
                                except Exception:
                                    try:
                                        member(chosen)
                                        print(f'Opened camera using cam.{open_try}(chosen)')
                                    except Exception:
                                        try:
                                            member(index)
                                            print(f'Opened camera using cam.{open_try}(index)')
                                        except Exception:
                                            continue
                                opened = True
                                break
                            except Exception as e:
                                print(f'cam.{open_try} failed:', e)
                                continue

                    # Fall back to cam.open(param) if available
                    if not opened and hasattr(cam, 'open'):
                        try:
                            cam.open(p)
                            opened = True
                            print('Opened camera using cam.open(Param)')
                        except Exception as e:
                            print('cam.open(Param) failed:', e)
                            opened = False

                    if opened:
                        self.cam = cam
                        self.using_arducam = True
                        print(f"Opened Arducam device {index} using SDK")
                    else:
                        print("Arducam: failed to open device with SDK fallbacks")

                else:
                    print('Arducam: no matching device index')

                # If not opened yet but user requested forcing Arducam, try to open by index/device_id directly
                if not opened and (force_arducam or device_id is not None):
                    cam = Camera()
                    print('Attempting forced open methods on Camera object...')
                    print('Camera object methods:', [m for m in dir(cam) if not m.startswith('_')])
                    for open_try in ('open', 'open_device', 'open_by_id', 'open_by_index', 'open_device_by_id'):
                        if hasattr(cam, open_try):
                            try:
                                member = getattr(cam, open_try)
                                # Try device_id first, then index
                                tried = False
                                if device_id is not None:
                                    try:
                                        member(device_id)
                                        print(f'Forced open with device_id via cam.{open_try}({device_id})')
                                        opened = True
                                        tried = True
                                    except Exception as e:
                                        print(f'cam.{open_try}({device_id}) failed:', e)
                                if not tried:
                                    try:
                                        member(index)
                                        print(f'Forced open with index via cam.{open_try}({index})')
                                        opened = True
                                    except Exception as e:
                                        print(f'cam.{open_try}({index}) failed:', e)
                                if opened:
                                    break
                            except Exception as e:
                                print(f'cam.{open_try} invocation error:', e)
                                continue

                    # try cam.open(index) or cam.open(device_id)
                    if not opened:
                        if hasattr(cam, 'open'):
                            try:
                                cam.open(index)
                                opened = True
                                print('Forced cam.open(index) succeeded')
                            except Exception as e:
                                print('Forced cam.open(index) failed:', e)
                                try:
                                    if device_id is not None:
                                        cam.open(device_id)
                                        opened = True
                                        print('Forced cam.open(device_id) succeeded')
                                except Exception as e2:
                                    print('Forced cam.open(device_id) failed:', e2)

                    if opened:
                        self.cam = cam
                        self.using_arducam = True
                        print('Forced Arducam open succeeded')

            except Exception as e:
                print("Arducam open failed with exception:", e)

        if not self.using_arducam:
            # Try OpenCV VideoCapture
            print('Falling back to OpenCV VideoCapture')
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # try without backend
                self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera index {index}")
            print(f"Opened VideoCapture device {index}")

    def read(self):
        if self.using_arducam and self.cam is not None:
            # Try multiple capture-like method names used by different SDK versions
            for fn in ('capture', 'read', 'get_frame', 'read_frame', 'getImage', 'get_image', 'get_buffer', 'get_buffered_image'):
                if hasattr(self.cam, fn):
                    try:
                        frame = getattr(self.cam, fn)()
                        # Some SDKs return (ret, frame) or a tuple; try to extract numpy array
                        if isinstance(frame, tuple) and len(frame) > 0:
                            for item in frame:
                                try:
                                    import numpy as _np
                                    if isinstance(item, _np.ndarray):
                                        return item
                                except Exception:
                                    pass
                            # if first element looks like frame
                            return frame[0]
                        return frame
                    except Exception:
                        continue
            # Last-resort: try cam.capture() and hope for a raw buffer convertible by OpenCV
            try:
                res = self.cam.capture()
                return res
            except Exception:
                return None
        else:
            ret, frame = self.cap.read()
            return frame if ret else None

    def set_auto_focus(self, enable: bool) -> bool:
        if self.using_arducam and self.cam is not None:
            try:
                # try various possible SDK method names
                for name in ('set_auto_focus', 'set_auto_focus_mode', 'set_auto_focus_enable'):
                    if hasattr(self.cam, name):
                        try:
                            getattr(self.cam, name)(bool(enable))
                            return True
                        except Exception:
                            continue
                # Some SDKs expose controls via a generic set_control
                if hasattr(self.cam, 'set_control'):
                    try:
                        self.cam.set_control('auto_focus', 1 if enable else 0)
                        return True
                    except Exception:
                        pass
            except Exception:
                return False
            return False
        else:
            try:
                return self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if enable else 0)
            except Exception:
                return False

    def set_focus(self, value: int) -> bool:
        if self.using_arducam and self.cam is not None:
            # try a number of possible SDK names for manual focus
            for name in ('set_focus', 'set_manual_focus', 'set_manual_focus_value', 'set_focus_absolute', 'set_focus_value', 'set_focus_pos'):
                if hasattr(self.cam, name):
                    try:
                        getattr(self.cam, name)(int(value))
                        return True
                    except Exception:
                        continue
            # Some SDKs expose controls via a generic set_control(name, value)
            if hasattr(self.cam, 'set_control'):
                try:
                    self.cam.set_control('focus', int(value))
                    return True
                except Exception:
                    pass
            # Try a generic control via set_controls or controls dict
            if hasattr(self.cam, 'set_controls'):
                try:
                    getattr(self.cam, 'set_controls')({'focus': int(value)})
                    return True
                except Exception:
                    pass
            return False
        else:
            try:
                return self.cap.set(cv2.CAP_PROP_FOCUS, float(value))
            except Exception:
                return False

    def close(self):
        try:
            if self.using_arducam and self.cam is not None:
                try:
                    self.cam.close()
                except Exception:
                    pass
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


def scan_video_devices(max_index=10, timeout=1.0):
    """Try opening VideoCapture indices 0..max_index using DSHOW and default backend.
    Print which indices return a frame and basic frame size. Useful to map OS device name to index.
    """
    print(f"Scanning video indices 0..{max_index} (timeout {timeout}s)...")
    for i in range(0, max_index + 1):
        found = False
        # Try DSHOW (Windows) first
        for backend in (cv2.CAP_DSHOW, None):
            try:
                if backend is None:
                    cap = cv2.VideoCapture(i)
                else:
                    cap = cv2.VideoCapture(i, backend)
            except Exception as e:
                print(f"  index {i} backend {backend}: open failed: {e}")
                continue
            if not cap or not cap.isOpened():
                # cleanup
                try:
                    cap.release()
                except Exception:
                    pass
                continue
            # try read a frame quickly
            t0 = time.time()
            frame = None
            while time.time() - t0 < timeout:
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  index {i} backend {backend}: OK frame {w}x{h}")
                found = True
            else:
                print(f"  index {i} backend {backend}: opened but no frame")
            try:
                cap.release()
            except Exception:
                pass
            if found:
                break
        if not found:
            print(f"  index {i}: no device or no frames")


def main():
    parser = argparse.ArgumentParser(description='Camera test with focus control')
    parser.add_argument('--index', type=int, default=0, help='Camera index (0-based) or Arducam device index')
    parser.add_argument('--max-focus', type=int, default=255, help='Maximum focus slider value (try 255 or 1000)')
    parser.add_argument('--force-arducam', action='store_true', help='Try extra SDK open methods to force using Arducam')
    parser.add_argument('--device-id', type=str, default=None, help='Device id/serial to pass to SDK open methods')
    parser.add_argument('--scan', action='store_true', help='Scan indices 0..10 to find which index corresponds to your camera')
    args = parser.parse_args()

    if args.scan:
        scan_video_devices(max_index=10, timeout=1.0)
        print('\nIf you see an index that returns a frame for DSHOW, try running:\n')
        print('  python cameratest.py --index <that-index>')
        print("Or to force the Arducam SDK try:\n  python cameratest.py --force-arducam --device-id Arducam_MP12")
        return

    cw = CamWrapper(index=args.index, force_arducam=args.force_arducam, device_id=args.device_id)

    window_name = 'Camera Test - Press q to quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Attempt to disable auto-focus so manual control works
    af_ok = cw.set_auto_focus(False)
    if af_ok:
        print('Auto-focus disabled (manual focus mode)')
    else:
        print('Auto-focus control not supported or failed; manual focus may not work')

    # Trackbar callback will call cw.set_focus
    focus_supported = {'ok': True}

    def on_focus(val):
        if not focus_supported['ok']:
            return
        ok = cw.set_focus(val)
        if not ok:
            # If not supported, disable further attempts to avoid spamming
            print('Focus control not supported by this camera/API')
            focus_supported['ok'] = False

    # create trackbar
    cv2.createTrackbar('Focus', window_name, 0, args.max_focus, on_focus)

    # set initial focus to middle
    init_f = args.max_focus // 2
    cv2.setTrackbarPos('Focus', window_name, init_f)
    # try to apply initial focus
    try:
        cw.set_focus(init_f)
    except Exception:
        pass

    try:
        while True:
            frame = cw.read()
            if frame is None:
                print('Failed to read frame, exiting')
                break

            # If Arducam returns a grayscale or BGR numpy array, just show it.
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # small sleep to reduce CPU usage
            time.sleep(0.005)
    finally:
        cw.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
