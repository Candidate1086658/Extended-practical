"""2D Material Transfer – Camera & Annotation Tool

Features:
    - Camera source selection (Toupcam / system Webcam)
    - Live image with manual exposure control (Toupcam only)
    - Freehand and straight-line drawing (red / blue, adjustable size)
    - Pixel-level eraser
    - Click-based optical contrast measurementSaneChips
    - Screenshot with annotations saved as PNG

Platform:  auto-detects Windows / macOS / Linux for correct BGR handling.
"""


import sys
import ctypes
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import queue
import threading
import traceback

# ===============================
# Optional imports with fallback
# ===============================
TOUPCAM_AVAILABLE = False
try:
    import toupcam
    TOUPCAM_AVAILABLE = True
except (ImportError, OSError):
    pass

WEBCAM_AVAILABLE = False
try:
    import cv2
    _test_cap = cv2.VideoCapture(0)
    WEBCAM_AVAILABLE = _test_cap.isOpened()
    _test_cap.release()
    del _test_cap
except (ImportError, Exception):
    pass

TUCAM_AVAILABLE = False
try:
    import os as _os
    _tucam_script_dir = _os.path.dirname(_os.path.abspath(__file__))
    _tucam_prev_cwd   = _os.getcwd()
    if _tucam_prev_cwd != _tucam_script_dir:
        _os.chdir(_tucam_script_dir)
    try:
        import TUCam as _tucam_mod
        TUCAM_AVAILABLE = True
    finally:
        if _tucam_prev_cwd != _tucam_script_dir:
            _os.chdir(_tucam_prev_cwd)
except Exception:
    pass


# ===============================
# Utility
# ===============================
def rgb_to_gray(rgb):
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


# ===============================
# Camera Selection Dialog
# ===============================
class CameraSelector:
    """
    Startup dialog.
    - If only one source is available  →  auto-select (no dialog shown).
    - If both are available            →  show a small radio-button dialog.
    Returns 'toupcam', 'webcam', or None.
    """

    def __init__(self):
        self._toupcam_available = False
        self._webcam_available = WEBCAM_AVAILABLE
        self._tucam_available = False

        if TOUPCAM_AVAILABLE:
            try:
                cams = toupcam.Toupcam.EnumV2()
                self._toupcam_available = bool(cams)
            except Exception:
                pass

        if TUCAM_AVAILABLE:
            try:
                from TUCam import TUCAM_INIT, TUCAM_Api_Init, TUCAM_Api_Uninit
                stInit = TUCAM_INIT()
                stInit.uiCamCount = 0
                stInit.pstrConfigPath = None
                TUCAM_Api_Init(ctypes.byref(stInit), 0)
                self._tucam_available = stInit.uiCamCount > 0
                TUCAM_Api_Uninit()
            except Exception:
                pass

    def run(self):
        # Always show dialog for user to choose camera
        root = tk.Tk()
        ttk.Style().theme_use('clam')
        root.withdraw()

        dialog = tk.Toplevel(root)
        dialog.title("Select Camera")
        dialog.geometry("300x250")
        dialog.resizable(False, False)
        dialog.grab_set()
        dialog.focus_set()

        tk.Label(dialog, text="Select camera source:",
                 font=("Arial", 11)).pack(pady=(18, 10))

        # Set default to first available camera (priority: toupcam > tucam > webcam)
        if self._toupcam_available:
            default = 'toupcam'
        elif self._tucam_available:
            default = 'tucam'
        else:
            default = 'webcam'
        var = tk.StringVar(value=default)

        # Show availability status
        toupcam_text = "Toupcam"      + ("" if self._toupcam_available else " (not detected)")
        tucam_text   = "TUCam"        + ("" if self._tucam_available   else " (not detected)")
        webcam_text  = "System Webcam" + ("" if self._webcam_available  else " (not detected)")

        tk.Radiobutton(dialog, text=toupcam_text,
                        variable=var, value='toupcam').pack(pady=3, anchor="w", padx=50)
        tk.Radiobutton(dialog, text=tucam_text,
                        variable=var, value='tucam').pack(pady=3, anchor="w", padx=50)
        tk.Radiobutton(dialog, text=webcam_text,
                        variable=var, value='webcam').pack(pady=3, anchor="w", padx=50)
        tk.Radiobutton(dialog, text="No Camera (Demo Mode)",
                        variable=var, value='demo').pack(pady=3, anchor="w", padx=50)

        result = [None]

        def ok():
            selected = var.get()
            # Check if selected camera is available
            if selected == 'toupcam' and not self._toupcam_available:
                messagebox.showwarning("Camera Not Available",
                    "Toupcam not detected.\nPlease install libtoupcam.dylib or select another camera.")
                return
            if selected == 'tucam' and not self._tucam_available:
                messagebox.showwarning("Camera Not Available",
                    "TUCam not detected.\nPlease check the TUCam SDK DLL or select another camera.")
                return
            if selected == 'webcam' and not self._webcam_available:
                messagebox.showwarning("Camera Not Available",
                    "System Webcam not detected.\nPlease connect a webcam or select another camera.")
                return
            result[0] = selected
            dialog.destroy()
            root.destroy()

        tk.Button(dialog, text="OK", command=ok, width=10).pack(pady=14)
        dialog.wait_window()
        return result[0]


# ===============================
# Camera: Toupcam
# ===============================
class ToupcamCamera:

    def __init__(self):
        self.hcam = None
        self.buf = None
        self.width = 0
        self.height = 0
        self._frame_queue = None
        self._frame_count = 0  # Debug counter

    def start(self, frame_queue):
        self._frame_queue = frame_queue

        cams = toupcam.Toupcam.EnumV2()
        if not cams:
            raise RuntimeError("No Toupcam device found")

        cam_info = cams[0]
        print(f"[Toupcam] Opening: {cam_info.displayname}")

        self.hcam = toupcam.Toupcam.Open(cam_info.id)
        if self.hcam is None:
            raise RuntimeError("Failed to open Toupcam device")

        self.width, self.height = self.hcam.get_Size()
        print(f"[Toupcam] Resolution: {self.width}x{self.height}")

        # Use bytes buffer (same as Test.py)
        bufsize = toupcam.TDIBWIDTHBYTES(self.width * 24) * self.height
        self.buf = bytes(bufsize)
        print(f"[Toupcam] Buffer: {bufsize} bytes")

        self.hcam.put_AutoExpoEnable(0)          # manual exposure

        # Keep a reference to the callback to prevent garbage collection
        self._callback_ref = self._static_cb
        self.hcam.StartPullModeWithCallback(self._callback_ref, self)
        print("[Toupcam] Pull mode started with callback")

    # -- callback bridge (toupcam requires a plain function, not a method) --
    @staticmethod
    def _static_cb(nEvent, ctx):
        if nEvent == toupcam.TOUPCAM_EVENT_IMAGE:
            ctx._on_image()
        elif nEvent == toupcam.TOUPCAM_EVENT_ERROR:
            print(f"[Toupcam] ERROR event received!")
        elif nEvent == toupcam.TOUPCAM_EVENT_DISCONNECTED:
            print(f"[Toupcam] Camera disconnected!")

    def _on_image(self):
        try:
            self._frame_count += 1
            if self._frame_count <= 3 or self._frame_count % 100 == 0:
                print(f"[Toupcam] Callback #{self._frame_count}")

            self.hcam.PullImageV4(self.buf, 0, 24, 0, None)

            arr = np.frombuffer(self.buf, dtype=np.uint8)
            expected_size = self.width * self.height * 3
            if arr.size < expected_size:
                print(f"[Toupcam] Buffer size mismatch: {arr.size} < {expected_size}")
                return

            # Reshape to image
            frame = arr[:expected_size].reshape((self.height, self.width, 3)).copy()

            # PullImageV4 (24-bit) on Windows returns BGR → flip to RGB.
            # On macOS/Linux returns RGB, no flip needed.
            if sys.platform == 'win32':
                frame = frame[:, :, ::-1].copy()

            if not self._frame_queue.full():
                self._frame_queue.put_nowait(frame)

        except Exception as e:
            print(f"[Toupcam] callback error: {e}")
            traceback.print_exc()

    def stop(self):
        if self.hcam:
            self.hcam.Stop()
            self.hcam.Close()
            self.hcam = None

    def get_size(self):
        return (self.width, self.height)

    def set_exposure(self, ms):
        if self.hcam:
            self.hcam.put_ExpoTime(int(ms * 1000))
            print(f"[Toupcam] Exposure → {ms:.2f} ms")


# ===============================
# Camera: TUCam
# ===============================
class TUCamCamera:

    def __init__(self):
        self.handle = None
        self.width = 0
        self.height = 0
        self._frame_queue = None
        self._thread = None
        self._stop_flag = False
        self._stFrame = None
        self._frame_count = 0

    def start(self, frame_queue):
        from TUCam import (TUCAM_INIT, TUCAM_OPEN, TUCAM_FRAME, TUFRM_FORMATS,
                           TUCAM_CAPTURE_MODES, TUCAM_Api_Init, TUCAM_Dev_Open,
                           TUCAM_Cap_Start, TUCAM_Buf_Alloc,
                           TUCAM_VALUE_INFO, TUCAM_Dev_GetInfo, TUCAM_IDINFO)
        self._frame_queue = frame_queue

        # --- 初始化 API，获取摄像头数量 ---
        stInit = TUCAM_INIT()
        stInit.uiCamCount = 0
        stInit.pstrConfigPath = None
        TUCAM_Api_Init(ctypes.byref(stInit), 0)
        if stInit.uiCamCount == 0:
            raise RuntimeError("No TUCam device found")
        print(f"[TUCam] Found {stInit.uiCamCount} camera(s)")

        # --- 打开第一个摄像头 ---
        stOpen = TUCAM_OPEN()
        stOpen.uiIdxOpen = 0
        stOpen.hIdxTUCam = None
        TUCAM_Dev_Open(ctypes.byref(stOpen))
        self.handle = stOpen.hIdxTUCam
        if not self.handle:
            raise RuntimeError("Failed to open TUCam device")

        # --- 查询当前分辨率 ---
        stInfo = TUCAM_VALUE_INFO()
        buf = ctypes.create_string_buffer(32)
        stInfo.pText = ctypes.cast(buf, ctypes.c_char_p)
        stInfo.nTextSize = 32
        stInfo.nID = TUCAM_IDINFO.TUIDI_CURRENT_WIDTH.value
        TUCAM_Dev_GetInfo(self.handle, ctypes.byref(stInfo))
        self.width = stInfo.nValue
        stInfo.nID = TUCAM_IDINFO.TUIDI_CURRENT_HEIGHT.value
        TUCAM_Dev_GetInfo(self.handle, ctypes.byref(stInfo))
        self.height = stInfo.nValue
        print(f"[TUCam] Resolution: {self.width}x{self.height}")

        # --- 分配帧缓冲区（RGB888 格式）---
        self._stFrame = TUCAM_FRAME()
        self._stFrame.ucFormatGet = TUFRM_FORMATS.TUFRM_FMT_RGB888.value
        self._stFrame.uiRsdSize = 1
        TUCAM_Buf_Alloc(self.handle, ctypes.byref(self._stFrame))

        # 若 Buf_Alloc 填充了尺寸则以其为准
        if self._stFrame.usWidth > 0 and self._stFrame.usHeight > 0:
            self.width  = self._stFrame.usWidth
            self.height = self._stFrame.usHeight

        # --- 启动连续采集 ---
        TUCAM_Cap_Start(self.handle, TUCAM_CAPTURE_MODES.TUCCM_SEQUENCE.value)
        print("[TUCam] Capture started")

        self._stop_flag = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        from TUCam import TUCAM_Buf_WaitForFrame
        while not self._stop_flag:
            try:
                TUCAM_Buf_WaitForFrame(self.handle, ctypes.byref(self._stFrame), 1000)
            except OSError:
                # 超时或 AbortWait 触发 — 检查停止标志
                continue

            w    = self._stFrame.usWidth
            h    = self._stFrame.usHeight
            step = self._stFrame.uiWidthStep
            ptr  = self._stFrame.pBuffer

            if not (ptr and w > 0 and h > 0):
                continue

            # 从指针读取原始字节，按行步长截取有效像素
            raw = (ctypes.c_uint8 * (step * h)).from_address(ptr)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, step))
            frame = np.ascontiguousarray(arr[:, :w * 3].reshape(h, w, 3))

            # TUCam RGB888 实际通道顺序为 R,B,G → 对调第1和第2通道转为 R,G,B
            frame = frame[:, :, [0, 2, 1]].copy()

            if not self._frame_queue.full():
                self._frame_queue.put_nowait(frame)

            self._frame_count += 1
            if self._frame_count <= 3 or self._frame_count % 100 == 0:
                print(f"[TUCam] 已生产帧#{self._frame_count}")

    def stop(self):
        from TUCam import (TUCAM_Buf_AbortWait, TUCAM_Cap_Stop,
                           TUCAM_Buf_Release, TUCAM_Dev_Close,
                           TUCAM_Api_Uninit)
        self._stop_flag = True
        if self.handle:
            try:
                TUCAM_Buf_AbortWait(self.handle)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self.handle:
            try:
                TUCAM_Cap_Stop(self.handle)
                TUCAM_Buf_Release(self.handle)
                TUCAM_Dev_Close(self.handle)
                TUCAM_Api_Uninit()
            except Exception:
                pass
            self.handle = None

    def get_size(self):
        return (self.width, self.height)

    def set_exposure(self, ms):
        from TUCam import TUCAM_Prop_SetValue, TUCAM_IDPROP
        if self.handle:
            try:
                # TUIDP_EXPOSURETM 单位为毫秒
                TUCAM_Prop_SetValue(self.handle,
                                    TUCAM_IDPROP.TUIDP_EXPOSURETM.value,
                                    float(ms), 0)
                print(f"[TUCam] Exposure → {ms:.2f} ms")
            except Exception as e:
                print(f"[TUCam] set_exposure error: {e}")


# ===============================
# Camera: Webcam (OpenCV)
# ===============================
class WebcamCamera:

    def __init__(self):
        self.cap = None
        self.width = 0
        self.height = 0
        self._frame_queue = None
        self._thread = None
        self._stop_flag = False

    def start(self, frame_queue):
        self._frame_queue = frame_queue

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Webcam] Opened: {self.width}x{self.height}")

        self._stop_flag = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        # 摄像头读帧线程主循环
        _produced = 0
        _consecutive_failures = 0
        _max_failures = 30  # Allow up to 30 consecutive failures (about 1 second at 30fps)

        # Give camera time to warm up
        time.sleep(0.3)

        while not self._stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                _consecutive_failures += 1
                if _consecutive_failures <= 5:
                    print(f"[Webcam] cap.read() failed (attempt {_consecutive_failures}), retrying...")
                if _consecutive_failures >= _max_failures:
                    print(f"[Webcam] Too many consecutive failures ({_max_failures}), thread exiting")
                    break
                time.sleep(0.05)  # Brief wait before retry
                continue

            # Reset failure counter on success
            _consecutive_failures = 0

            # OpenCV 所有平台都是 BGR → 翻转为 RGB
            frame = frame[:, :, ::-1].copy()
            if not self._frame_queue.full():
                self._frame_queue.put_nowait(frame)
            _produced += 1
            if _produced <= 3 or _produced % 300 == 0:
                print(f"[Webcam] 已生产帧#{_produced}, shape={frame.shape}, "
                      f"dtype={frame.dtype}, queue.full={self._frame_queue.full()}")

    def stop(self):
        self._stop_flag = True
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_size(self):
        return (self.width, self.height)

    def set_exposure(self, ms):
        # Best-effort: not all webcams support this
        if self.cap:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, ms / 1000.0)

# ===============================
# Camera: DemoCamera, allow you to open local photos
# ===============================

class DemoCamera:
    def __init__(self):
        self.width = 1280
        self.height = 720

    def start(self, frame_queue):
        import threading
        def _loop():
            import time
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            while True:
                if not frame_queue.full():
                    frame_queue.put_nowait(blank.copy())
                time.sleep(0.05)
        threading.Thread(target=_loop, daemon=True).start()

    def stop(self): pass
    def get_size(self): return (self.width, self.height)
    def set_exposure(self, ms): pass

# ===============================
# HBN Theoretical Reference (TMM + 5700K + CIE 1964, 285 nm SiO2/Si)
# Computed at module load: linear R and B contrast vs hBN thickness (0–300 nm)
# ===============================
HBN_DATA_AVAILABLE = False
HBN_THEO_T = None    # thickness array (nm)
HBN_THEO_CR = None   # linear R-channel contrast vs thickness
HBN_THEO_CB = None   # linear B-channel contrast vs thickness

try:
    # Wavelength grid + Si refractive index (380–780 nm, 5 nm step)
    _wl_nm = np.arange(380, 781, 5)
    _si_n = np.array([
        4.0812, 4.1404, 4.1903, 4.231, 4.2717, 4.3124, 4.3531, 4.3852, 4.4004, 4.4157,
        4.4309, 4.4462, 4.4615, 4.4713, 4.4742, 4.4772, 4.4801, 4.483, 4.486, 4.4889,
        4.4868, 4.4815, 4.4763, 4.471, 4.4681, 4.4656, 4.4632, 4.4608, 4.4528, 4.4421,
        4.4315, 4.4209, 4.4102, 4.3996, 4.3889, 4.3783, 4.3677, 4.3568, 4.3452, 4.3337,
        4.3221, 4.3106, 4.299, 4.2875, 4.276, 4.2644, 4.2529, 4.2413, 4.2298, 4.2206,
        4.2115, 4.2023, 4.1931, 4.1839, 4.1748, 4.1647, 4.1536, 4.1426, 4.1315, 4.1205,
        4.1094, 4.0984, 4.0876, 4.0778, 4.0679, 4.058, 4.0481, 4.0382, 4.0284, 4.0185,
        4.0088, 4.0, 3.9912, 3.9825, 3.9737, 3.9649, 3.9561, 3.9474, 3.9386, 3.9299,
        3.9231
    ])

    # 5700 K blackbody illuminant (peak-normalized)
    _wl_m = _wl_nm * 1e-9
    _h = 6.626e-34; _c = 3e8; _kB = 1.381e-23; _T = 5700
    _radiance = (2 * _h * _c**2 / (_wl_m**5)) * (1 / (np.exp(_h * _c / (_wl_m * _kB * _T)) - 1))
    _L5700 = _radiance / np.max(_radiance)

    # CIE 1964 10-deg color matching functions (Gaussian fits, same as RGB 5700 fit V2.py)
    _x_bar = (0.398 * np.exp(-1250 * (np.log((_wl_nm + 570.1) / 1014))**2)
              + 1.132 * np.exp(-234 * (np.log((1338 - _wl_nm) / 743.5))**2))
    _y_bar = 1.011 * np.exp(-0.5 * ((_wl_nm - 556.1) / 46.14)**2)
    _z_bar = 2.060 * np.exp(-32 * (np.log((_wl_nm - 265.8) / 180.4))**2)

    _X_white = float(np.sum(_L5700 * _x_bar))
    _Y_white = float(np.sum(_L5700 * _y_bar))
    _Z_white = float(np.sum(_L5700 * _z_bar))

    def _hbn_theo_linear_rgb(d_hbn, d_sio2=285.0):
        """TMM at fixed SiO2/Si stack, returns linear R, G, B before clip/gamma."""
        n0 = 1.0; n1 = 2.2; n2 = 1.46
        reflectance = []
        for wl in _wl_nm:
            n3 = float(np.interp(wl, _wl_nm, _si_n))
            delta1 = 2 * np.pi * n1 * d_hbn / wl
            delta2 = 2 * np.pi * n2 * d_sio2 / wl
            M1 = np.array([[np.cos(delta1), -1j * (1/n1) * np.sin(delta1)],
                           [-1j * n1 * np.sin(delta1), np.cos(delta1)]])
            M2 = np.array([[np.cos(delta2), -1j * (1/n2) * np.sin(delta2)],
                           [-1j * n2 * np.sin(delta2), np.cos(delta2)]])
            M = np.matmul(M1, M2)
            m11, m12, m21, m22 = M[0,0], M[0,1], M[1,0], M[1,1]
            r = ((m11 + m12 * n3) * n0 - (m21 + m22 * n3)) / ((m11 + m12 * n3) * n0 + (m21 + m22 * n3))
            reflectance.append(np.abs(r)**2)
        spectrum = np.array(reflectance) * _L5700
        X = float(np.sum(spectrum * _x_bar)) / _X_white
        Y = float(np.sum(spectrum * _y_bar)) / _Y_white
        Z = float(np.sum(spectrum * _z_bar)) / _Z_white
        R_lin =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
        G_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
        B_lin =  0.0557 * X - 0.2040 * Y + 1.0570 * Z
        return R_lin, G_lin, B_lin

    # Sweep hBN thickness 0–300 nm in 1 nm steps; compute R and B contrast vs bare substrate
    HBN_THEO_T = np.arange(0, 301, 1, dtype=float)
    _R_base, _, _B_base = _hbn_theo_linear_rgb(0.0)
    _cr_list = []
    _cb_list = []
    for _d in HBN_THEO_T:
        _R_lin, _, _B_lin = _hbn_theo_linear_rgb(float(_d))
        _cr_list.append((_R_lin - _R_base) / _R_base)
        _cb_list.append((_B_lin - _B_base) / _B_base)
    HBN_THEO_CR = np.array(_cr_list)
    HBN_THEO_CB = np.array(_cb_list)

    HBN_DATA_AVAILABLE = True
    print(f"[HBN] Theoretical R/B contrast computed: {len(HBN_THEO_T)} points "
          f"(0–300 nm), 285 nm SiO2/Si, 5700 K + CIE 1964")
except Exception as _hbn_err:
    print(f"[HBN] Could not compute theoretical contrast: {_hbn_err}")


def _find_local_minima_thicknesses(residual, t_array, abs_threshold=0.01):
    """
    Return all local minima in ``residual`` as (thickness_nm, residual) pairs,
    sorted by residual ascending (best first).

    A local minimum is any point strictly smaller than both neighbours; the two
    array endpoints are also checked. Only minima with ``residual < abs_threshold``
    are kept; if none qualify, the global minimum is returned as a single fallback.
    """
    n = len(residual)
    if n == 0:
        return []

    is_min = np.zeros(n, dtype=bool)
    if n >= 3:
        is_min[1:-1] = (residual[1:-1] < residual[:-2]) & (residual[1:-1] < residual[2:])
    if n >= 2:
        if residual[0] < residual[1]:
            is_min[0] = True
        if residual[-1] < residual[-2]:
            is_min[-1] = True

    minima_idx = np.where(is_min)[0]
    if minima_idx.size == 0:
        minima_idx = np.array([int(np.argmin(residual))])

    minima_idx = minima_idx[np.argsort(residual[minima_idx])]
    good = [int(i) for i in minima_idx if residual[i] < abs_threshold]
    if not good:
        good = [int(np.argmin(residual))]

    return [(float(t_array[i]), float(residual[i])) for i in good]


def fit_hbn_thickness(flake_rgb, sub_rgb):
    """
    Fit hBN thickness from raw 8-bit (R,G,B) tuples by comparing gamma-corrected
    linear contrasts against the TMM theoretical curves at 285 nm SiO2/Si.

    Steps:
        1. Linearize raw sRGB via (X/255)**2.2.
        2. Compute linear contrast per channel: c_X = (I_flake - I_sub)/I_sub.
        3. Search HBN_THEO_T for thickness that minimizes joint R+B residual.
        4. Also compute per-channel best fits (for diagnostics; not displayed).
        5. Collect all per-channel local-minimum candidates.

    G channel is excluded — theory and measurement disagree there.

    Returns
    -------
    best_t        : float  – joint best-fit thickness (nm)
    best_t_R      : float  – R-channel-only global best-fit thickness (nm)
    best_t_B      : float  – B-channel-only global best-fit thickness (nm)
    quality       : float  – 1 - RMSE / max(|c_R|, |c_B|), clipped to [0, 1]
    candidates_R  : list   – R-channel local minima as (thickness, residual), best first
    candidates_B  : list   – B-channel local minima as (thickness, residual), best first
    """
    if not HBN_DATA_AVAILABLE:
        return None, None, None, None, [], []

    f_r, _f_g, f_b = flake_rgb
    s_r, _s_g, s_b = sub_rgb

    # 1. Linearize via gamma 2.2
    Iflake_R = (f_r / 255.0) ** 2.2
    Iflake_B = (f_b / 255.0) ** 2.2
    Isub_R = (s_r / 255.0) ** 2.2
    Isub_B = (s_b / 255.0) ** 2.2

    if Isub_R <= 0 or Isub_B <= 0:
        return None, None, None, None, [], []

    # 2. Linear contrast (sign convention: (flake - sub)/sub, matches theory)
    c_R_meas = (Iflake_R - Isub_R) / Isub_R
    c_B_meas = (Iflake_B - Isub_B) / Isub_B

    # 3. Per-channel residuals across the theoretical thickness grid
    rR = (HBN_THEO_CR - c_R_meas) ** 2
    rB = (HBN_THEO_CB - c_B_meas) ** 2

    # 4. Joint best fit + per-channel best fits (R and B are quasi-periodic, so
    #    each channel alone admits multiple matches — the joint argmin breaks
    #    the degeneracy because R and B have different periods).
    sse = rR + rB
    best_idx = int(np.argmin(sse))
    best_idx_R = int(np.argmin(rR))
    best_idx_B = int(np.argmin(rB))

    best_t = float(HBN_THEO_T[best_idx])
    best_t_R = float(HBN_THEO_T[best_idx_R])
    best_t_B = float(HBN_THEO_T[best_idx_B])

    # 5. Quality metric: 1 - RMSE / max contrast magnitude, clipped to [0, 1]
    rmse = float(np.sqrt(sse[best_idx] / 2.0))
    norm = max(abs(c_R_meas), abs(c_B_meas), 1e-6)
    quality = max(0.0, min(1.0, 1.0 - rmse / norm))

    # 6. Per-channel local-minimum candidates (for the "Individual Channel
    #    Fittings" panel — exposes the periodic ambiguity to the user).
    candidates_R = _find_local_minima_thicknesses(rR, HBN_THEO_T)
    candidates_B = _find_local_minima_thicknesses(rB, HBN_THEO_T)

    return best_t, best_t_R, best_t_B, quality, candidates_R, candidates_B


# ===============================
# Main Application
# ===============================
class ClickContrastApp:

    def __init__(self, root, camera_type):
        self.root = root
        self.root.title("2D Material Transfer Tool")
        self.camera_type = camera_type

        # ---- canvas dimensions ----
        self.canvas_w = 960
        self.canvas_h = 720

        # ---- layout ----
        main = tk.Frame(root)
        main.pack(fill="both", expand=True)

        # Use Canvas for image display
        self.canvas = tk.Canvas(main, width=self.canvas_w, height=self.canvas_h,
                                bg='black', highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self._canvas_image_id = None
        self.tk_img = None
        self._img_offset = (0, 0)  # Letterbox offset
        self._img_scale = 1.0      # Scale factor

        right = tk.Frame(main)
        right.pack(side="right", fill="y", padx=2)

        self._build_panel_camera(right)   # Panel 1: camera / contrast / screenshot
        self._build_panel_drawing(right)  # Panel 2: drawing tools

        # ---- camera state ----
        self.width = 0
        self.height = 0
        self.camera = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.current_frame = None
        self.tk_img = None

        # ---- drawing state ----
        self.overlay = Image.new('RGBA', (self.canvas_w, self.canvas_h), (0, 0, 0, 0))
        self.mode = 'draw'                # 'draw' | 'measure'
        self.current_tool = 'brush'       # 'brush' | 'eraser'
        self.current_color = (255, 0, 0)  # red
        self.brush_size = 3
        self.drawing = False
        self.last_point = None
        self.line_start = None
        self.overlay_backup = None        # used for straight-line preview

        # ---- material mode ----
        self.material = 'graphene'   # 'graphene' | 'hbn'

        # ---- contrast state ----
        self.stage = 0
        self.flake_I = None
        self.flake_rgb = None        # HBN mode: stores (R, G, B) of flake region

        # ---- hold mode (locked substrate reference) ----
        self.hold_active  = False
        self.hold_ref_I   = None     # graphene: locked substrate luminance
        self.hold_ref_rgb = None     # hbn: locked substrate (R, G, B)

        # Initialize open picture state (for demo camera)
        self._static_image = None   # set when user opens a local picture

        # ---- measure rectangle drag state ----
        self._measure_rect_start = None   # canvas (x, y) where drag began
        self._canvas_rect_id    = None    # temporary canvas rectangle item

        # ---- debug 帧计数 ----
        self._frame_count = 0

        # ---- resize debounce ----
        self._resize_after_id = None

        # ---- mouse bindings ----
        self.canvas.bind("<Button-1>",        self.on_mouse_down)
        self.canvas.bind("<B1-Motion>",       self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>",       self._on_canvas_resize)

        # ---- window close ----
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # ---- start ----
        self._start_camera()
        self._update_gui()

    # =============================================================
    # Panel 1  –  Camera & Measure
    # =============================================================
    def _build_panel_camera(self, parent):
        panel = tk.LabelFrame(parent, text="Camera & Measure")
        panel.pack(side="left", fill="y", padx=(4, 2), pady=5)

        # -- Exposure (hidden when using webcam) --
        self.exposure_frame = tk.Frame(panel)
        self.exposure_frame.pack(fill="x", padx=10, pady=(8, 0))

        tk.Label(self.exposure_frame, text="Exposure",
                  font=("Arial", 10, "bold")).pack(anchor="w")
        tk.Label(self.exposure_frame, text="Time (ms):").pack(anchor="w", pady=(4, 0))

        self.expo_ms_var = tk.StringVar(value="10.0")
        expo_entry = tk.Entry(self.exposure_frame,
                               textvariable=self.expo_ms_var, width=10)
        expo_entry.pack(anchor="w", pady=2)
        expo_entry.bind("<Return>", self._on_exposure_enter)

        tk.Label(self.exposure_frame, text="Press Enter to apply",
                  foreground="gray").pack(anchor="w")

        tk.Frame(panel, height=1, bg="gray").pack(fill="x", pady=8, padx=10)

        # -- Contrast info label --
        self.info_label = tk.Label(
            panel,
            text="Switch to Measure mode\nto measure contrast",
            font=("Arial", 10),
            wraplength=170
        )
        self.info_label.pack(padx=10, pady=8, anchor="w")

        # -- HBN results frame (hidden until HBN mode is active) --
        self.hbn_results_frame = tk.Frame(panel)
        # Not packed here; shown/hidden by _on_material_change()

        tk.Label(self.hbn_results_frame, text="HBN Measurement",
                 font=("Arial", 10, "bold")).pack(anchor="w", pady=(4, 2))

        self._hbn_flake_lbl = tk.Label(self.hbn_results_frame,
                                        text="Flake  R=\u2014  G=\u2014  B=\u2014",
                                        font=("Courier", 9), wraplength=170)
        self._hbn_flake_lbl.pack(anchor="w")

        self._hbn_sub_lbl = tk.Label(self.hbn_results_frame,
                                      text="Subs   R=\u2014  G=\u2014  B=\u2014",
                                      font=("Courier", 9), wraplength=170)
        self._hbn_sub_lbl.pack(anchor="w")

        tk.Frame(self.hbn_results_frame, height=1, bg="gray").pack(fill="x", pady=4)

        self._hbn_thickness_lbl = tk.Label(self.hbn_results_frame,
                                            text="t = \u2014",
                                            font=("Arial", 10, "bold"), wraplength=170)
        self._hbn_thickness_lbl.pack(anchor="w")

        tk.Frame(panel, height=1, bg="gray").pack(fill="x", pady=8, padx=10)

        # -- Screenshot --
        tk.Button(panel, text="Screenshot",
                   command=self._take_screenshot).pack(pady=8, padx=10)

        # -- Hold (lock substrate reference) --
        self.hold_btn = tk.Button(panel, text="Hold",
                                  command=self._on_hold_toggle)
        self.hold_btn.pack(pady=(0, 4), padx=10)
        # Save default colours so we can restore them on deactivation
        self._hold_btn_default_bg = self.hold_btn.cget('bg')
        self._hold_btn_default_fg = self.hold_btn.cget('fg')

        # -- Individual Channel Fittings (shows R/B candidate lists) --
        self.individual_btn = tk.Button(panel, text="Individual Channel Fittings",
                                         command=self._on_individual_fittings)
        self.individual_btn.pack(pady=(0, 2), padx=10)
        self._individual_btn_default_bg = self.individual_btn.cget('bg')
        self._individual_btn_default_fg = self.individual_btn.cget('fg')

        self._inaccurate_lbl = tk.Label(panel, text="",
                                         font=("Arial", 9, "bold"),
                                         fg="red")
        self._inaccurate_lbl.pack(padx=10, pady=(0, 6))

        # Window handle + cached candidate lists for the popup
        self._individual_window = None
        self._last_R_candidates = []
        self._last_B_candidates = []

        tk.Frame(panel, height=1, bg="gray").pack(fill="x", pady=8, padx=10)

        tk.Button(panel, text="Open Picture",
            command=self._open_picture).pack(pady=(0, 8), padx=10)

    # =============================================================
    # Panel 2  –  Drawing Tools
    # =============================================================
    def _build_panel_drawing(self, parent):
        panel = tk.LabelFrame(parent, text="Drawing Tools")
        panel.pack(side="left", fill="y", padx=(2, 4), pady=5)

        # -- Material --
        tk.Label(panel, text="Material",
                  font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        mat_row = tk.Frame(panel)
        mat_row.pack(anchor="w", padx=10)

        self.material_var = tk.StringVar(value='graphene')
        tk.Radiobutton(mat_row, text="Graphene",
                        variable=self.material_var, value='graphene',
                        command=self._on_material_change).pack(side="left")
        tk.Radiobutton(mat_row, text="HBN",
                        variable=self.material_var, value='hbn',
                        command=self._on_material_change).pack(side="left")

        tk.Frame(panel, height=1, bg="gray").pack(fill="x", pady=6, padx=10)

        # -- Mode --
        tk.Label(panel, text="Mode",
                  font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=(4, 2))
        mode_row = tk.Frame(panel)
        mode_row.pack(anchor="w", padx=10)

        self.mode_var = tk.StringVar(value='draw')
        tk.Radiobutton(mode_row, text="Draw",
                        variable=self.mode_var, value='draw',
                        command=self._on_mode_change).pack(side="left")
        tk.Radiobutton(mode_row, text="Measure",
                        variable=self.mode_var, value='measure',
                        command=self._on_mode_change).pack(side="left")

        tk.Frame(panel, height=1, bg="gray").pack(fill="x", pady=6, padx=10)

        # -- Tool --
        tk.Label(panel, text="Tool",
                  font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=(4, 2))
        tool_row = tk.Frame(panel)
        tool_row.pack(anchor="w", padx=10)

        self.btn_brush = tk.Button(tool_row, text="Brush", width=7,
                                   relief="sunken",
                                   command=lambda: self._set_tool('brush'))
        self.btn_brush.pack(side="left", padx=(0, 3))
        self.btn_eraser = tk.Button(tool_row, text="Eraser", width=7,
                                    relief="raised",
                                    command=lambda: self._set_tool('eraser'))
        self.btn_eraser.pack(side="left")

        tk.Frame(panel, height=1, bg="gray").pack(fill="x", pady=6, padx=10)

        # -- Color --
        tk.Label(panel, text="Color",
                  font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=(4, 2))
        color_row = tk.Frame(panel)
        color_row.pack(anchor="w", padx=10)

        self.btn_red = tk.Button(color_row, text="Red", width=6,
                                 relief="sunken",
                                 command=lambda: self._set_color((255, 0, 0)))
        self.btn_red.pack(side="left", padx=(0, 3))
        self.btn_blue = tk.Button(color_row, text="Blue", width=6,
                                  relief="raised",
                                  command=lambda: self._set_color((0, 0, 255)))
        self.btn_blue.pack(side="left")

        tk.Frame(panel, height=1, bg="gray").pack(fill="x", pady=6, padx=10)

        # -- Size slider --
        tk.Label(panel, text="Size",
                  font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=(4, 0))
        self.size_var = tk.IntVar(value=3)
        tk.Scale(panel, from_=1, to=40, variable=self.size_var,
                  orient="horizontal", length=150,
                  showvalue=0).pack(padx=10, pady=2)
        self.size_lbl = tk.Label(panel, text="3 px")
        self.size_lbl.pack(anchor="w", padx=10)
        self.size_var.trace_add("write", lambda *_: self._on_size_change())

        tk.Frame(panel, height=1, bg="gray").pack(fill="x", pady=6, padx=10)

        # -- Straight-line toggle --
        self.straight_var = tk.BooleanVar(value=False)
        tk.Checkbutton(panel, text="Straight Line",
                        variable=self.straight_var).pack(anchor="w", padx=10, pady=4)

        tk.Frame(panel, height=1, bg="gray").pack(fill="x", pady=6, padx=10)

        # -- Clear All --
        tk.Button(panel, text="Clear All",
                   command=self._clear_overlay).pack(anchor="w", padx=10, pady=4)

    # =============================================================
    # UI callbacks
    # =============================================================
    def _on_material_change(self):
        self.material = self.material_var.get()
        # Reset measurement state (including hold)
        self.stage = 0
        self.flake_I = None
        self.flake_rgb = None
        # Reset HBN-specific candidate cache + warning
        self._last_R_candidates = []
        self._last_B_candidates = []
        self._reset_individual_warning()
        self._populate_individual_window()
        if self.hold_active:
            self.hold_active  = False
            self.hold_ref_I   = None
            self.hold_ref_rgb = None
            self.hold_btn.config(bg=self._hold_btn_default_bg,
                                 fg=self._hold_btn_default_fg)
        if self._canvas_rect_id is not None:
            self.canvas.delete(self._canvas_rect_id)
            self._canvas_rect_id = None
        self._measure_rect_start = None
        # Show/hide HBN results panel
        if self.material == 'hbn':
            self.hbn_results_frame.pack(padx=10, pady=(0, 4), fill="x",
                                        after=self.info_label)
            self.info_label.config(
                text="HBN mode: drag to\nselect flake, then substrate.")
        else:
            self.hbn_results_frame.pack_forget()
            if self.mode == 'measure':
                self.info_label.config(text="Drag to select flake,\nthen substrate")
            else:
                self.info_label.config(text="Drawing mode active")

    def _update_individual_warning(self, candidates_R, candidates_B):
        """Three-state warning based on cross-channel candidate matching:
          - No (R, B) pair within 10 nm anywhere
                → red, "Individual channel fittings mismatch"
          - Multiple disjoint matching clusters (pairs whose averages are
            > 30 nm apart, e.g. one match near 50 nm AND another near 150 nm)
                → red, "Multiple possible fittings"
          - One consistent matching cluster
                → button stays default, no warning text
        """
        if not candidates_R or not candidates_B:
            self._reset_individual_warning()
            return

        # Collect every (R, B) pair within 10 nm
        matches = []
        for t_R, _ in candidates_R:
            for t_B, _ in candidates_B:
                if abs(t_R - t_B) <= 10.0:
                    matches.append((t_R, t_B))

        if not matches:
            self.individual_btn.config(bg='red', fg='white')
            self._inaccurate_lbl.config(text="Individual channel fittings mismatch")
            return

        # Cluster matches by their average thickness; a gap > 30 nm between
        # consecutive averages = two genuinely different candidate fits.
        cluster_gap = 30.0  # nm
        avg_thicknesses = sorted((tR + tB) / 2.0 for tR, tB in matches)
        multiple_fits = any(
            avg_thicknesses[i] - avg_thicknesses[i - 1] > cluster_gap
            for i in range(1, len(avg_thicknesses))
        )

        if multiple_fits:
            self.individual_btn.config(bg='red', fg='white')
            self._inaccurate_lbl.config(text="Multiple possible fittings")
        else:
            self.individual_btn.config(bg=self._individual_btn_default_bg,
                                        fg=self._individual_btn_default_fg)
            self._inaccurate_lbl.config(text="")

    def _reset_individual_warning(self):
        self.individual_btn.config(bg=self._individual_btn_default_bg,
                                    fg=self._individual_btn_default_fg)
        self._inaccurate_lbl.config(text="")

    def _on_individual_fittings(self):
        """Open (or refresh) a small window listing R and B candidate fits."""
        # Reuse window if already open
        if (self._individual_window is not None
                and self._individual_window.winfo_exists()):
            self._individual_window.lift()
            self._individual_window.focus_set()
            self._populate_individual_window()
            return

        win = tk.Toplevel(self.root)
        win.title("Individual Channel Fittings")
        win.geometry("360x320")
        win.transient(self.root)
        self._individual_window = win

        self._individual_text = tk.Text(win, font=("Courier", 10),
                                         wrap="word", state="disabled")
        self._individual_text.pack(fill="both", expand=True, padx=10, pady=10)

        self._populate_individual_window()

    def _populate_individual_window(self):
        if self._individual_window is None or not self._individual_window.winfo_exists():
            return

        lines = []
        if not self._last_R_candidates and not self._last_B_candidates:
            lines.append("No fit available yet.\n")
            lines.append("Make an HBN measurement first.")
        else:
            lines.append("R channel candidates (best first):")
            if self._last_R_candidates:
                for t, res in self._last_R_candidates:
                    lines.append(f"  t = {t:6.1f} nm   residual = {res:.5f}")
            else:
                lines.append("  (none)")
            lines.append("")
            lines.append("B channel candidates (best first):")
            if self._last_B_candidates:
                for t, res in self._last_B_candidates:
                    lines.append(f"  t = {t:6.1f} nm   residual = {res:.5f}")
            else:
                lines.append("  (none)")

        text_widget = self._individual_text
        text_widget.config(state="normal")
        text_widget.delete("1.0", "end")
        text_widget.insert("1.0", "\n".join(lines))
        text_widget.config(state="disabled")

    def _on_hold_toggle(self):
        """Toggle hold mode: lock/unlock the substrate reference."""
        self.hold_active = not self.hold_active
        if self.hold_active:
            # If a stage-0 measurement is already stored, promote it to the
            # locked substrate reference immediately.
            if self.flake_I is not None:
                self.hold_ref_I   = self.flake_I
                self.hold_ref_rgb = self.flake_rgb
                self.stage     = 0
                self.flake_I   = None
                self.flake_rgb = None
                self.info_label.config(text="Hold: substrate locked.\nSelect flake.")
            else:
                self.info_label.config(
                    text="Hold active.\nSelect substrate to lock.")
            self.hold_btn.config(bg='red', fg='white')
        else:
            # Deactivate: clear reference, full reset
            self.hold_ref_I   = None
            self.hold_ref_rgb = None
            self.stage     = 0
            self.flake_I   = None
            self.flake_rgb = None
            self.hold_btn.config(bg=self._hold_btn_default_bg,
                                 fg=self._hold_btn_default_fg)
            if self.mode == 'measure':
                if self.material == 'hbn':
                    self.info_label.config(
                        text="HBN mode: drag to\nselect flake, then substrate.")
                else:
                    self.info_label.config(
                        text="Drag to select flake,\nthen substrate")
            else:
                self.info_label.config(text="Drawing mode active")

    def _on_mode_change(self):
        self.mode = self.mode_var.get()
        if self.mode == 'measure':
            if self.material == 'hbn':
                self.info_label.config(text="HBN mode: drag to\nselect flake, then substrate.")
            else:
                self.info_label.config(text="Drag to select flake,\nthen substrate")
            self.stage = 0
            self.flake_I = None
            self.flake_rgb = None
            if self.hold_active:
                self.hold_active  = False
                self.hold_ref_I   = None
                self.hold_ref_rgb = None
                self.hold_btn.config(bg=self._hold_btn_default_bg,
                                     fg=self._hold_btn_default_fg)
            # 清除可能残留的矩形预览
            if self._canvas_rect_id is not None:
                self.canvas.delete(self._canvas_rect_id)
                self._canvas_rect_id = None
            self._measure_rect_start = None
        else:
            if self.hold_active:
                self.hold_active  = False
                self.hold_ref_I   = None
                self.hold_ref_rgb = None
                self.hold_btn.config(bg=self._hold_btn_default_bg,
                                     fg=self._hold_btn_default_fg)
            self.info_label.config(text="Drawing mode active")

    def _clear_overlay(self):
        """清除所有已绘制的标注。"""
        self.overlay = Image.new('RGBA', (self.canvas_w, self.canvas_h), (0, 0, 0, 0))

    def _set_tool(self, tool):
        self.current_tool = tool
        self.btn_brush.config(relief="sunken" if tool == 'brush' else "raised")
        self.btn_eraser.config(relief="sunken" if tool == 'eraser' else "raised")

    def _set_color(self, color):
        self.current_color = color
        self.btn_red.config(relief="sunken"  if color == (255, 0, 0) else "raised")
        self.btn_blue.config(relief="sunken" if color == (0, 0, 255) else "raised")

    def _on_size_change(self):
        self.brush_size = self.size_var.get()
        self.size_lbl.config(text=f"{self.brush_size} px")

    def _on_exposure_enter(self, event):
        try:
            ms = float(self.expo_ms_var.get())
            if not (0.1 <= ms <= 1000):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid", "Enter a number between 0.1 and 1000 (ms).")
            return
        self.camera.set_exposure(ms)

    # =============================================================
    # Camera lifecycle
    # =============================================================
    def _start_camera(self):
        if self.camera_type == 'toupcam':
            self.camera = ToupcamCamera()
        elif self.camera_type == 'tucam':
            self.camera = TUCamCamera()
        elif self.camera_type == 'webcam':
            self.camera = WebcamCamera()
        else :
            self.camera = DemoCamera()



        self.camera.start(self.frame_queue)
        self.width, self.height = self.camera.get_size()
        print(f"[Camera] 启动成功 | 类型={self.camera_type} | 原始分辨率={self.width}x{self.height}")

        if self.camera_type == 'webcam':
            self.exposure_frame.pack_forget()   # hide exposure controls
        else:
            # apply default exposure
            try:
                self.camera.set_exposure(float(self.expo_ms_var.get()))
            except Exception:
                pass

    def on_close(self):
        if self.camera:
            self.camera.stop()
        self.root.destroy()

    # =============================================================
    # Canvas resize  —  保持比例，同步缩放 overlay
    # =============================================================
    def _on_canvas_resize(self, event):
        new_w, new_h = event.width, event.height
        if new_w < 10 or new_h < 10:
            return
        if new_w == self.canvas_w and new_h == self.canvas_h:
            return
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(
            150, lambda w=new_w, h=new_h: self._apply_canvas_resize(w, h))

    def _apply_canvas_resize(self, new_w, new_h):
        self._resize_after_id = None
        if self.overlay.size != (new_w, new_h):
            self.overlay = self.overlay.resize((new_w, new_h), Image.LANCZOS)
        self.canvas_w = new_w
        self.canvas_h = new_h

    # =============================================================
    # Mouse event routing
    # =============================================================
    def on_mouse_down(self, event):
        if self.mode == 'measure':
            self._measure_drag_start(event)
        else:
            self._draw_start(event)

    def on_mouse_move(self, event):
        if self.mode == 'measure' and self._measure_rect_start is not None:
            self._measure_drag_move(event)
        elif self.mode == 'draw' and self.drawing:
            self._draw_move(event)

    def on_mouse_up(self, event):
        if self.mode == 'measure' and self._measure_rect_start is not None:
            self._measure_drag_end(event)
        elif self.mode == 'draw' and self.drawing:
            self._draw_end(event)

    # =============================================================
    # Drawing
    # =============================================================
    @staticmethod
    def _clip(event, w, h):
        """Clamp canvas-event coordinates into [0, size)."""
        return (max(0, min(event.x, w - 1)),
                max(0, min(event.y, h - 1)))

    def _draw_start(self, event):
        self.drawing = True
        pt = self._clip(event, self.canvas_w, self.canvas_h)
        self.last_point = pt

        if self.current_tool == 'brush':
            if self.straight_var.get():
                # Straight-line mode: save start point + backup overlay for preview
                self.line_start = pt
                self.overlay_backup = self.overlay.copy()
            else:
                # Freehand: paint a dot at the start so single-click marks are visible
                self._paint_dot(pt)

    def _draw_move(self, event):
        pt = self._clip(event, self.canvas_w, self.canvas_h)

        if self.current_tool == 'brush':
            if self.straight_var.get():
                # Straight-line preview: restore backup, draw preview segment
                self.overlay = self.overlay_backup.copy()
                self._paint_line(self.line_start, pt)
            else:
                # Freehand segment
                self._paint_line(self.last_point, pt)
                self.last_point = pt

        elif self.current_tool == 'eraser':
            self._erase_at(pt)
            self.last_point = pt

    def _draw_end(self, event):
        pt = self._clip(event, self.canvas_w, self.canvas_h)

        if self.current_tool == 'brush' and self.straight_var.get():
            # Confirm the straight line onto the real overlay
            self.overlay = self.overlay_backup.copy()
            self._paint_line(self.line_start, pt)
            self.overlay_backup = None
            self.line_start = None

        self.drawing = False
        self.last_point = None

    # -- low-level paint helpers --
    def _paint_line(self, p1, p2):
        draw = ImageDraw.Draw(self.overlay)
        draw.line([p1, p2],
                  fill=self.current_color + (255,),
                  width=self.brush_size)

    def _paint_dot(self, pt):
        r = max(1, self.brush_size // 2)
        draw = ImageDraw.Draw(self.overlay)
        draw.ellipse([pt[0] - r, pt[1] - r,
                      pt[0] + r, pt[1] + r],
                     fill=self.current_color + (255,))

    def _erase_at(self, center):
        """Pixel-level circular erase at *center* with radius = brush_size."""
        cx, cy = center
        r = self.brush_size

        # Tight bounding box to avoid scanning the whole image
        x0 = max(0,              cx - r)
        x1 = min(self.canvas_w,  cx + r + 1)
        y0 = max(0,              cy - r)
        y1 = min(self.canvas_h,  cy + r + 1)

        arr = np.array(self.overlay)          # PIL → numpy (copy)
        sub = arr[y0:y1, x0:x1]              # view into the bounding box

        # Circular mask in local coordinates
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

        sub[mask] = [0, 0, 0, 0]             # set to fully transparent

        self.overlay = Image.fromarray(arr)   # numpy → PIL

    # =============================================================
    # Contrast measurement  —  矩形拖选区域
    # =============================================================
    def _measure_drag_start(self, event):
        """按下鼠标：记录起点，清除旧预览框。"""
        self._measure_rect_start = (event.x, event.y)
        if self._canvas_rect_id is not None:
            self.canvas.delete(self._canvas_rect_id)
            self._canvas_rect_id = None

    def _measure_drag_move(self, event):
        """拖动中：实时更新黄色虚线预览框。"""
        x0, y0 = self._measure_rect_start
        if self._canvas_rect_id is not None:
            self.canvas.delete(self._canvas_rect_id)
        self._canvas_rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline='yellow', width=2, dash=(6, 3)
        )

    def _measure_drag_end(self, event):
        """松开鼠标：计算选区内平均值，更新对比度 / HBN 厚度。"""
        if self.current_frame is None or self._measure_rect_start is None:
            self._measure_rect_start = None
            return

        x0c, y0c = self._measure_rect_start
        x1c, y1c = event.x, event.y
        self._measure_rect_start = None

        # 删除预览框
        if self._canvas_rect_id is not None:
            self.canvas.delete(self._canvas_rect_id)
            self._canvas_rect_id = None

        # Canvas 坐标 → 原始图像坐标 (shared for both materials)
        offset_x, offset_y = self._img_offset
        scale = self._img_scale

        def to_img(cx, cy):
            return (int((cx - offset_x) / scale),
                    int((cy - offset_y) / scale))

        ix0, iy0 = to_img(min(x0c, x1c), min(y0c, y1c))
        ix1, iy1 = to_img(max(x0c, x1c), max(y0c, y1c))

        # 裁剪到图像范围
        ix0 = max(0, min(ix0, self.width  - 1))
        iy0 = max(0, min(iy0, self.height - 1))
        ix1 = max(0, min(ix1, self.width  - 1))
        iy1 = max(0, min(iy1, self.height - 1))

        # ── Graphene mode ────────────────────────────────────────────────────
        if self.material == 'graphene':
            if ix1 - ix0 < 2 or iy1 - iy0 < 2:
                rgb  = self.current_frame[iy0, ix0]
                gray = rgb_to_gray(rgb)
                area_str = ""
            else:
                region   = self.current_frame[iy0:iy1 + 1, ix0:ix1 + 1]
                gray_map = (0.299 * region[:, :, 0].astype(float)
                          + 0.587 * region[:, :, 1].astype(float)
                          + 0.114 * region[:, :, 2].astype(float))
                gray     = float(gray_map.mean())
                n_px     = (iy1 - iy0 + 1) * (ix1 - ix0 + 1)
                area_str = f" ({n_px} px)"

            if self.hold_active:
                if self.hold_ref_I is None:
                    # First selection in hold mode → lock as substrate reference
                    self.hold_ref_I = gray
                    self.info_label.config(
                        text=f"Hold: substrate I={gray:.1f}{area_str}\nSelect flake.")
                else:
                    # Every subsequent selection → flake vs locked reference
                    if self.hold_ref_I == 0:
                        self.info_label.config(text="Reference I = 0!\nRe-select.")
                    else:
                        contrast = (gray - self.hold_ref_I) / self.hold_ref_I
                        self.info_label.config(
                            text=f"Contrast = {contrast:.4f}\n"
                                 f"Flake: I = {gray:.1f}{area_str}")
            else:
                if self.stage == 0:
                    self.flake_I = gray
                    self.info_label.config(
                        text=f"Flake: I = {gray:.1f}{area_str}\nNow select substrate.")
                    self.stage = 1
                else:
                    sub_I = gray
                    if sub_I == 0:
                        self.info_label.config(
                            text="Substrate I = 0!\n(re-select flake)")
                    else:
                        contrast = (self.flake_I - sub_I) / sub_I
                        self.info_label.config(
                            text=f"Contrast = {contrast:.4f}\n"
                                 f"Sub: I = {sub_I:.1f}{area_str}")
                    self.stage = 0
                    self.flake_I = None

        # ── HBN mode: measure R,G,B separately; calculate contrast after both ─
        else:
            # Extract per-channel mean R, G, B from the selected region
            if ix1 - ix0 < 2 or iy1 - iy0 < 2:
                px    = self.current_frame[iy0, ix0].astype(float)
                r_val = float(px[0])
                g_val = float(px[1])
                b_val = float(px[2])
                area_str = ""
            else:
                region   = self.current_frame[iy0:iy1 + 1, ix0:ix1 + 1].astype(float)
                r_val    = float(region[:, :, 0].mean())
                g_val    = float(region[:, :, 1].mean())
                b_val    = float(region[:, :, 2].mean())
                n_px     = (iy1 - iy0 + 1) * (ix1 - ix0 + 1)
                area_str = f" ({n_px} px)"

            lum = rgb_to_gray((r_val, g_val, b_val))

            if self.hold_active:
                if self.hold_ref_rgb is None:
                    # First selection in hold mode → lock as substrate reference
                    self.hold_ref_rgb = (r_val, g_val, b_val)
                    self.hold_ref_I   = lum
                    self._hbn_sub_lbl.config(
                        text=f"Subs   R={r_val:.0f}  G={g_val:.0f}  B={b_val:.0f}")
                    self.info_label.config(
                        text=f"Hold: substrate locked{area_str}.\nSelect flake.")
                else:
                    # Every subsequent selection → flake vs locked reference
                    if r_val == 0 and g_val == 0 and b_val == 0:
                        self.info_label.config(text="Flake is black!\nRe-select.")
                        return
                    self._hbn_flake_lbl.config(
                        text=f"Flake  R={r_val:.0f}  G={g_val:.0f}  B={b_val:.0f}")
                    if HBN_DATA_AVAILABLE:
                        (best_t, _best_t_R, _best_t_B, _quality,
                         cand_R, cand_B) = fit_hbn_thickness(
                            (r_val, g_val, b_val), self.hold_ref_rgb)
                        self._last_R_candidates = cand_R
                        self._last_B_candidates = cand_B
                        self._populate_individual_window()
                        if best_t is not None:
                            self._hbn_thickness_lbl.config(text=f"t = {best_t:.1f} nm")
                            self.info_label.config(text=f"t = {best_t:.1f} nm")
                            self._update_individual_warning(cand_R, cand_B)
                        else:
                            self._hbn_thickness_lbl.config(text="t = \u2014  (fit failed)")
                            self.info_label.config(text="Fit failed")
                            self._reset_individual_warning()
                    else:
                        self._hbn_thickness_lbl.config(text="t = \u2014  (no ref data)")
                        self.info_label.config(text="No reference data")
                        self._reset_individual_warning()

            else:
                if self.stage == 0:
                    # Stage 0: record raw flake pixel values — no contrast yet
                    self.flake_rgb = (r_val, g_val, b_val)
                    self.flake_I   = lum
                    self._hbn_flake_lbl.config(
                        text=f"Flake  R={r_val:.0f}  G={g_val:.0f}  B={b_val:.0f}")
                    self.info_label.config(
                        text=f"Flake: R={r_val:.0f} G={g_val:.0f} B={b_val:.0f}{area_str}"
                             f"\nNow select substrate.")
                    self.stage = 1

                else:
                    # Stage 1: record raw substrate values, then calculate contrast
                    if r_val == 0 and g_val == 0 and b_val == 0:
                        self.info_label.config(
                            text="Substrate is black!\nRe-select substrate.")
                        return

                    self._hbn_sub_lbl.config(
                        text=f"Subs   R={r_val:.0f}  G={g_val:.0f}  B={b_val:.0f}")

                    if HBN_DATA_AVAILABLE:
                        (best_t, _best_t_R, _best_t_B, _quality,
                         cand_R, cand_B) = fit_hbn_thickness(
                            self.flake_rgb, (r_val, g_val, b_val))
                        self._last_R_candidates = cand_R
                        self._last_B_candidates = cand_B
                        self._populate_individual_window()
                        if best_t is not None:
                            self._hbn_thickness_lbl.config(text=f"t = {best_t:.1f} nm")
                            self.info_label.config(text=f"t = {best_t:.1f} nm")
                            self._update_individual_warning(cand_R, cand_B)
                        else:
                            self._hbn_thickness_lbl.config(text="t = \u2014  (fit failed)")
                            self.info_label.config(text="Fit failed")
                            self._reset_individual_warning()
                    else:
                        self._hbn_thickness_lbl.config(text="t = \u2014  (no ref data)")
                        self.info_label.config(text="No reference data")
                        self._reset_individual_warning()

                    self.stage     = 0
                    self.flake_rgb = None
                    self.flake_I   = None

    # =============================================================
    # Screenshot
    # =============================================================
    def _take_screenshot(self):
        if self.current_frame is None:
            messagebox.showwarning("No Image", "No frame captured yet.")
            return

        # Composite at original camera resolution
        frame_pil = Image.fromarray(self.current_frame)
        overlay_scaled = self.overlay.resize(
            (self.width, self.height), Image.LANCZOS)
        frame_pil.paste(overlay_scaled, (0, 0), overlay_scaled)

        filepath = filedialog.asksaveasfilename(
            title="Save Screenshot",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")]
        )
        if filepath:
            frame_pil.save(filepath, "PNG")
            print(f"[Screenshot] Saved: {filepath}")
            messagebox.showinfo("Saved", f"Saved to:\n{filepath}")


    # =============================================================
    # Open picture
    # =============================================================
    def _open_picture(self):
        filepath = filedialog.askopenfilename(
            title="Open Picture",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All Files", "*.*")
            ]
        )
        if not filepath:
            return

        img = Image.open(filepath).convert("RGB")
        img = img.resize((self.width, self.height), Image.LANCZOS)
        frame = np.array(img)
        self._static_image = frame
        self.current_frame = frame
        print(f"[OpenPicture] Loaded: {filepath} → {frame.shape}")


    # =============================================================
    # GUI 刷新循环  — 由 root.after() 在主线程反复驱动
    # =============================================================
    def _update_gui(self):
        try:
            # ── 从队列里取最新一帧（中间帧丢弃，保证始终显示最新） ──如果有 _static_image 则忽略摄像头输入，保持显示静态图像（用于调试）
            frame = None
            if not hasattr(self, '_static_image') or self._static_image is None:
                while not self.frame_queue.empty():
                    try:
                        frame = self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                if frame is not None:
                    self.current_frame = frame
                    self._frame_count += 1

            # ── 合成画面并更新 Canvas ──
            if self.current_frame is not None:
                # 读取 Canvas 当前实际尺寸（随窗口动态变化）
                cw = max(self.canvas.winfo_width(),  1)
                ch = max(self.canvas.winfo_height(), 1)

                # 保持原始比例缩放（letterbox，无畸变）
                orig_h, orig_w = self.current_frame.shape[:2]
                scale = min(cw / orig_w, ch / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)

                # 缩放图像（保持比例）
                img = Image.fromarray(self.current_frame).resize(
                    (new_w, new_h), Image.LANCZOS)

                # 创建黑色背景画布，将图像居中
                canvas_img = Image.new('RGB', (cw, ch), (0, 0, 0))
                offset_x = (cw - new_w) // 2
                offset_y = (ch - new_h) // 2
                canvas_img.paste(img, (offset_x, offset_y))

                # 保存偏移量供绘图和测量使用
                self._img_offset = (offset_x, offset_y)
                self._img_scale = scale

                # 叠加注释层：overlay 若与当前 canvas 尺寸不符则临时缩放
                overlay = self.overlay if self.overlay.size == (cw, ch) else \
                          self.overlay.resize((cw, ch), Image.LANCZOS)
                canvas_img.paste(overlay, (0, 0), overlay)

                self.tk_img = ImageTk.PhotoImage(canvas_img)

                # Create or update the canvas image
                if self._canvas_image_id is None:
                    self.canvas.delete('all')
                    self._canvas_image_id = self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)
                    print(f"[GUI] Created canvas image | scale={scale:.3f} | offset=({offset_x},{offset_y})")
                else:
                    self.canvas.itemconfig(self._canvas_image_id, image=self.tk_img)

                # 前 3 帧 + 之后每 60 帧打印一次 debug 信息
                if self._frame_count <= 3 or self._frame_count % 60 == 0:
                    print(f"[GUI] 帧#{self._frame_count} "
                          f"| img={img.size} mode={img.mode} "
                          f"| canvas={self.canvas.winfo_width()}x{self.canvas.winfo_height()}")
            else:
                # 还没收到第一帧时偶尔提醒一下
                if self._frame_count == 0 and not hasattr(self, '_warned'):
                    self._warned = True
                    print("[GUI] 等待摄像头第一帧…")

        except Exception as e:
            # 异常全部打印，不让循环中断
            print(f"[GUI] _update_gui 异常: {e}")
            traceback.print_exc()

        # ── 无论是否出错，都继续调度下一帧（约 50 fps） ──
        self.root.after(20, self._update_gui)


# ===============================
# Entry point
# ===============================
if __name__ == "__main__":
    # --- camera selection ---
    selector = CameraSelector()
    camera_type = selector.run()

    if camera_type is None:
        camera_type = 'demo'

    print(f"[Main] Using camera: {camera_type}")

    root = tk.Tk()
    ttk.Style().theme_use('clam')
    app = ClickContrastApp(root, camera_type)
    root.mainloop()
