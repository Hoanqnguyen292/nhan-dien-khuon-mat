"""Phát hiện, nhận dạng và theo dõi khuôn mặt.

Kiến trúc 3 tầng, mỗi tầng tự lùi về bản cũ nếu thiếu model:
    FaceDetector   YuNet (có 5 điểm mốc)      -> Haar cascade
    FaceMatcher    SFace (vector 128 chiều)     -> LBPH
    FaceTracker    khớp track theo IoU + vector, làm mượt ô vuông, CSRT nối frame hụt

Ba nâng cấp chính so với bản cũ (Haar + LBPH):
  1. KHÔNG NHẦM NGƯỜI khi khung hình có nhiều người:
     - so khớp vector mặt thay vì khoảng cách pixel;
     - ngưỡng riêng cho từng người, tính từ độ tự phân bố mẫu của chính họ;
     - bắt buộc có khoảng cách bỏ ngỏ so với ứng viên thứ hai (loại người nhập nhằng);
     - mỗi người chỉ được chiếm tối đa 1 ô trong cùng một khung hình.
  2. CĂN MẶT theo điểm mốc trước khi trích đặc trưng: đầu xoay/nghiêng vẫn khớp.
  3. THEO DÕI ỔN ĐỊNH: ô vuông làm mượt theo thời gian, giữ nguyên ID khi mất
     detect, chỉ chốt tên sau khi đủ số khung xác nhận -> nhãn không nhấp nháy.

Nhãn vẽ bằng Pillow để hiển thị đúng tiếng Việt (có dấu).
"""
import shutil
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import database as db

#hằng số
SFACE_SIZE = (112, 112)          # ảnh chuẩn cho SFace
LBPH_SIZE = (200, 200)           # ảnh cũ cho LBPH

# Ngưỡng mặc định theo đơn vị của từng bộ so khớp (đều là "càng nhỏ càng giống"):
#   - cosin (SFace): đo trên ảnh thật -> cùng người <= 0.34, người khác 0.84.
#     Ngưỡng chính thức của OpenCV là 0.363; 0.42 chừa biên cho ảnh nhỏ/ánh sáng lệch.
#   - LBPH (cũ): cùng người ~60, người khác ~89 => 75 là điểm giữa an toàn.
DEFAULTS = {
    "match_threshold": 0.42,      # khoảng cách lớn nhất vẫn coi là khớp
    "match_margin": 0.10,         # chênh lệch tối thiểu so với người thứ hai
    "adaptive_threshold": True,   # tự siết ngưỡng cho người có mẫu chuẩn
    "detect_score": 0.62,         # ngưỡng chắc chắn của YuNet
    "min_face_size": 44,          # bỏ mặt nhỏ hơn (px) -> giảm nhầm, tăng tốc
    "detect_width": 640,          # bề rộng chuẩn cho ảnh chạy detector (0 = giữ nguyên).
                                 # Luôn đưa ảnh về bề rộng này: thu lớn để tiết kiệm,
                                 # phóng nhỏ để YuNet bắt được mặt nhỏ (camera độ phân giải thấp).
    "detect_every": 2,            # chạy detector mỗi N khung (1 = mỗi khung)
    "feature_every": 2,           # trích vector mới mỗi N khung cho mỗi track
    "blur_min": 12.0,             # biến Laplacian dưới ngưỡng này coi như mờ nét
    "sample_min": 48,             # mặt nhỏ hơn (px) thì không nhận làm mẫu
    # --- Cổng chất lượng khi CHỐT TÊN (chống nhận nhầm người lạ) ------------
    # Mặt càng nhỏ / càng mờ thì vector SFace càng nhiễu, dễ trùng ngẫu nhiên
    # với mẫu của học sinh khác. Hai ngưỡng dưới đây chặn việc CHỐT TÊN
    # (không chặn việc phát hiện hay đăng ký mẫu).
    "recognize_min_face": 64,     # cạnh mặt tối thiểu, px quy về khung rộng 640
    "recognize_blur_min": 15.0,   # độ nét Laplacian tối thiểu của vùng mặt
    "sample_dedup": 0.12,         # bỏ mẫu gần trùng mẫu đã có
    "augment_flip": True,         # lưu thêm bản lật ngang cho mỗi mẫu
    # Cân bằng sáng (CLAHE) trước khi trích đặc trưng: giúp nhận diện ổn định khi
    # ánh sáng đổi (tối/sáng, ngược sáng). 0 = tắt (quay lại hành vi cũ).
    "light_norm": True,
    "track_enabled": True,
    "track_backend": "medianflow",  # medianflow | kcf | csrt | none
    "track_confirm": 3,           # số khung liên tiếp khớp mới chốt tên
    "track_max_age": 12,          # số khung liên tiếp mất detect tối đa vẫn giữ track
    "track_iou": 0.20,            # IoU tối thiểu để coi là cùng một track
    "track_emb": 0.45,            # khoảng cách vector cho phép ghép khi hình học hụt
    "track_smooth": 0.45,         # hệ số làm mượt ô vuông (0 = đứng yên, 1 = không mượt)
    "square_pad": 1.18,           # cạnh ô vuông = max(w, h) * hệ số này
}

INT_KEYS = ("min_face_size", "detect_width", "detect_every", "feature_every",
            "sample_min", "track_confirm", "track_max_age", "recognize_min_face")
STR_KEYS = ("track_backend",)

CASCADE_NAME = "haarcascade_frontalface_default.xml"
CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/" + CASCADE_NAME
)

# Model mạnh hơn Haar/LBPH, tải một lần về data/models/
MODELS = {
    "yunet": (
        "face_detection_yunet_2023mar.onnx",
        "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
    "sface": (
        "face_recognition_sface_2021dec.onnx",
        # Chu y: file SFace (~38MB) luu dang Git LFS tren opencv_zoo nen host
        # raw.githubusercontent.com chi tra POINTER 133 byte (khong phai model).
        # Dung media.githubusercontent.com de lay binary that.
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
}
MODEL_DIR = db.DATA_DIR / "models"

if not hasattr(cv2, "face"):
    raise RuntimeError(
        "Thiếu module cv2.face. Hãy cài: pip uninstall opencv-python "
        "opencv-contrib-python && pip install opencv-contrib-python"
    )

_FONT_PATHS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
_fonts = {}


def _font_size(img) -> int:
    return max(14, img.shape[0] // 36)


def _font(size: int):
    size = int(size)
    if size not in _fonts:
        path = next((p for p in _FONT_PATHS if Path(p).exists()), None)
        _fonts[size] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
    return _fonts[size]


def _text_height(img, text) -> int:
    """Chiều cao thực tế của một dòng chữ trên ảnh này (để không đè lên ô vuông)."""
    font = _font(_font_size(img))
    try:
        return int(font.getbbox(text or "Ag")[3])
    except AttributeError:
        return int(getattr(font, "size", 14) * 1.4)


def _new_lbph():
    factory = getattr(cv2.face, "LBPHFaceRecognizer_create", None)
    if factory:
        return factory()
    return cv2.face.LBPHFaceRecognizer.create()


def _coerce_threshold(metric: str, value: float) -> float:
    """Ngưỡng hợp lệ theo metric đang chạy.

    Thang cosin (gia tri <= 1.5, vd 0.42) VO NGHIA voi LBPH (thang 0..150):
    giu nguyen thi khong bao gio khop ai -> doi sang mac dinh 75.
    Nguong thang LBPH do nguoi dung tu chinh thi ton trong.
    """
    value = float(value)
    if metric != "cosine" and value <= 1.5:
        return 75.0
    return value


def _ensure_model(key: str) -> Path:
    """Đường dẫn model; tải từ opencv_zoo một lần nếu chưa có."""
    name, url = MODELS[key]
    local = MODEL_DIR / name
    if not local.exists() or local.stat().st_size < 1024:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[face] Dang tai model {name} ...")
        tmp = local.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(local)
        print(f"[face] Xong {name} ({local.stat().st_size // 1024} KB)")
    return local


def _iou(a, b) -> float:
    """IoU của hai hộp (x, y, w, h)."""
    ax1, ay1 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def square_box(x, y, w, h, pad=1.18):
    """Ô vuông bao quanh mặt, cùng tâm -> (x, y, cạnh). Dùng cho cả vẽ và theo dõi."""
    side = max(float(w), float(h)) * pad
    return x + w / 2.0 - side / 2.0, y + h / 2.0 - side / 2.0, side


def fit_square(square, frame_w, frame_h, margin=2):
    """Đưa ô vuông [x, y, cạnh] vào trong khung hình mà VẪN GIỮ NGUYÊN HÌNH VUÔNG.

    Quan trọng: nếu ô tràn ra ngoài khung thì ta DỜI nó vào trong chứ không thu nhỏ,
    nếu thu nhỏ thì ô sẽ méo thành hình chữ nhật -> mất đúng ý nghĩa "ô vuông".
    """
    x, y, side = float(square[0]), float(square[1]), max(8.0, float(square[2]))
    side = min(side, float(frame_w) - 2 * margin, float(frame_h) - 2 * margin)
    side = max(8.0, side)
    # dời tâm vào trong khung, giữ nguyên cạnh
    cx = min(max(x + side / 2.0, margin + side / 2.0), frame_w - margin - side / 2.0)
    cy = min(max(y + side / 2.0, margin + side / 2.0), frame_h - margin - side / 2.0)
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    side_i = int(round(side))
    return x0, y0, x0 + side_i, y0 + side_i


def _as_sface_input(img: np.ndarray) -> np.ndarray:
    """Đưa ảnh về đúng kích thước/kiểu SFace cần: 112x112x3 BGR uint8."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[:2] != SFACE_SIZE:
        img = cv2.resize(img, SFACE_SIZE, interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(img[:, :, :3])


# --- Cân bằng sáng (giảm phụ thuộc ánh sáng) --------------------------------
# CLAHE trên kênh L của LAB: tăng tương phản CỤC BỘ, giữ nguyên màu. Ảnh mẫu lúc
# đăng ký, ảnh lúc nhận diện và ảnh nạp lại từ đĩa đều đi qua hàm này nên vector
# đặc trưng SFace nhất quán -> cùng một người vẫn khớp khi ánh sáng thay đổi.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def normalize_light(img: np.ndarray, enable: bool = True) -> np.ndarray:
    """Chuẩn hoá ánh sáng trước khi trích đặc trưng (trả nguyên ảnh nếu tắt)."""
    if not enable or img is None or getattr(img, "size", 0) == 0:
        return img
    out = img
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _sharpness(gray: np.ndarray) -> float:
    """Độ nét = phương sai Laplacian, ĐÃ cân bằng sáng.

    Ảnh chụp trong lớp thiếu sáng có phương sai thấp một cách GIẢ TẠO. Nếu đo
    trên ảnh gốc thì mặt đúng người vẫn bị coi nhầm là "mờ" -> bị loại trước khi
    so khớp (đúng triệu chứng "đổi ánh sáng một chút là không nhận ra").
    Cân bằng sáng trước khi đo để ngưỡng chỉ còn phản ánh độ NÉT thật.
    """
    if gray is None or getattr(gray, "size", 0) < 64:
        return 0.0
    try:
        gray = cv2.equalizeHist(gray)
    except cv2.error:
        pass
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# chữ / vẽ
def draw_text(img, text, org, color, bg=(20, 20, 20), pad=4):
    """Vẽ text (tiếng Việt) lên ảnh BGR, có nền để dễ đọc. color là RGB."""
    draw_labels(img, [(text, org, color)], bg=bg, pad=pad)


def draw_labels(img, items, bg=(20, 20, 20), pad=4):
    """Vẽ nhiều nhãn chỉ trong MỘT lần chuyển ảnh BGR->PIL (tiết kiệm rõ ràng).

    `items` = [(text, (x, y), color_rgb) | (text, (x, y), color_rgb, anchor), ...]
    anchor "top" (mặc định): (x, y) là góc trên-trái.
    anchor "bottom":         (x, y) là mép DƯỚI của dòng chữ -> dùng để đặt nhãn
                            ngay TRÊN ô vuông mà không đè lên viền ô.
    """
    if not items:
        return
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _font(_font_size(img))
    for item in items:
        text, org, color = item[0], item[1], item[2]
        anchor = item[3] if len(item) > 3 else "top"
        x, y = int(org[0]), int(org[1])
        if anchor == "bottom":
            y -= _text_height(img, text)        # dời lên để mép dưới chạm y
        try:
            left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        except AttributeError:
            left, top, right, bottom = (x, y, x + 8 * len(text), y + font.size)
        if bg is not None:
            draw.rectangle((left - pad, top - pad, right + pad, bottom + pad), fill=bg)
        draw.text((x, y), text, font=font, fill=tuple(color))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# phát hiện mặt
@dataclass
class Detection:
    """Một khuôn mặt tìm thấy trong khung hình."""

    box: tuple                        # (x, y, w, h)
    score: float = 1.0                # độ chắc chắn của detector (0..1)
    landmarks: np.ndarray = None      # (5,2): mắt phải, mắt trái, mũi, khóe miệng

    @property
    def row(self) -> np.ndarray:
        """Hàng 15 số theo định dạng mà cv2.alignCrop của OpenCV mong đợi."""
        r = np.zeros(15, np.float32)
        r[0:4] = self.box
        if self.landmarks is not None:
            r[4:14] = np.asarray(self.landmarks, np.float32).reshape(-1)
        r[14] = self.score
        return r


class FaceDetector:
    """YuNet nếu có model; nếu không thì lùi về Haar cascade (bản cũ)."""

    def __init__(self, score_thresh: float = DEFAULTS["detect_score"],
                 min_face_size: int = DEFAULTS["min_face_size"]):
        self.score_thresh = float(score_thresh)
        self.min_face_size = int(min_face_size)
        self.name = "haar"
        self._det = None
        self._cascade = None
        self._lock = threading.Lock()
        try:
            path = _ensure_model("yunet")
            self._det = cv2.FaceDetectorYN.create(
                str(path), "", (320, 320), self.score_thresh, 0.3, 5000)
            self.name = "yunet"
        except Exception as exc:  # noqa: BLE001 - thiếu model/mạng -> dùng Haar
            print(f"[face] Khong dung YuNet ({exc}); chuyen sang Haar cascade")
        if self._det is None:
            self._cascade = self._load_cascade()

    @staticmethod
    def _load_cascade():
        path = Path(cv2.data.haarcascades) / CASCADE_NAME
        if not path.exists():
            local = MODEL_DIR / CASCADE_NAME
            if not local.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(CASCADE_URL, local)
            path = local
        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            raise RuntimeError(f"Khong tai duoc Haar cascade: {path}")
        return cascade

    @property
    def has_landmarks(self) -> bool:
        """Chỉ YuNet cho điểm mốc -> dùng được sf.alignCrop; Haar thì không."""
        return self._det is not None

    def detect(self, bgr: np.ndarray):
        """list[Detection] trong toạ độ của chính ảnh `bgr`."""
        if bgr is None or bgr.size == 0:
            return []
        h, w = bgr.shape[:2]
        try:
            with self._lock:  # detector KHÔNG thread-safe
                if self._det is not None:
                    self._det.setInputSize((w, h))
                    self._det.setScoreThreshold(self.score_thresh)
                    _, faces = self._det.detect(bgr)
                    return self._from_yunet(faces, w, h)
                gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
                faces = self._cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5,
                    minSize=(self.min_face_size,) * 2)
                return self._from_haar(faces)
        except cv2.error as exc:
            print(f"[detect] {exc}")
            return []

    def detect_raw(self, bgr: np.ndarray, score: float = 0.3):
        """Hộp mặt THÔ ở ngưỡng `score`, không lọc kích thước.

        Chỉ dùng để CHẨN ĐOÁN vì sao ảnh không làm mẫu được (mặt quá nhỏ/mờ
        vẫn hiện ở ngưỡng thấp). Mọi detect() thường đều tự đặt lại ngưỡng
        nên không ảnh hưởng luồng chính.
        """
        if bgr is None or bgr.size == 0:
            return []
        h, w = bgr.shape[:2]
        try:
            with self._lock:  # detector KHÔNG thread-safe
                if self._det is None:
                    return []
                self._det.setInputSize((w, h))
                self._det.setScoreThreshold(float(score))
                try:
                    _, faces = self._det.detect(np.ascontiguousarray(bgr))
                finally:
                    self._det.setScoreThreshold(float(self.score_thresh))
                out = []
                if faces is not None:
                    for f in faces:
                        x, y, bw, bh = (int(round(float(v))) for v in f[:4])
                        if min(bw, bh) > 0:
                            out.append((x, y, bw, bh))
                return out
        except cv2.error:
            return []

    def _from_yunet(self, faces, w, h):
        out = []
        if faces is None:
            return out
        for f in faces:
            fx, fy, fw, fh = (float(v) for v in f[:4])
            x = int(round(min(max(fx, 0.0), w - 1.0)))
            y = int(round(min(max(fy, 0.0), h - 1.0)))
            bw = int(round(min(max(fw, 1.0), w - x)))
            bh = int(round(min(max(fh, 1.0), h - y)))
            if min(bw, bh) < self.min_face_size:
                continue
            lm = np.asarray(f[4:14], np.float32).reshape(5, 2).copy()
            out.append(Detection((x, y, bw, bh), float(f[14]), lm))
        out.sort(key=lambda d: -d.score)
        return out

    @staticmethod
    def _from_haar(faces):
        out = [Detection((int(x), int(y), int(w), int(h)), 1.0, None)
               for (x, y, w, h) in faces]
        out.sort(key=lambda d: -d.box[2] * d.box[3])
        return out


# so khớp mặt
class FaceMatcher:
    """SFace (vector 128 chiều + khoảng cách cosin) nếu có model; không thì LBPH.

    Ngoài ngưỡng toàn cục, mỗi người có một ngưỡng riêng suy ra từ độ giống nhau
    giữa các mẫu của chính họ: người có mẫu sạch sẽ được siết chặt hơn, người có
    mẫu đa dạng (nhiều cự ly, nhiều ánh sáng) sẽ được nới ra -> giảm nhận nhầm.
    """

    def __init__(self, threshold: float = DEFAULTS["match_threshold"], adaptive: bool = True):
        self.adaptive = bool(adaptive)
        self.metric = "lbph"          # "cosine" | "lbph"
        self.name = "lbph"
        self.threshold = float(threshold)
        self._lock = threading.RLock()
        self._model_lock = threading.Lock()
        self._sf = None
        self._lbph = None
        self.light_norm = bool(DEFAULTS["light_norm"])   # cân bằng sáng trước khi trích
        self.gallery = {}              # sid -> ndarray (n, 128) đã chuẩn hoá
        self.gallery_gray = {}         # sid -> list ảnh xám (đường LBPH)
        self.per_person = {}           # sid -> ngưỡng riêng
        self.sample_counts = {}
        self._trained = False
        try:
            path = _ensure_model("sface")
            self._sf = cv2.FaceRecognizerSF.create(str(path), "")
            self.metric = "cosine"
            self.name = "sface"
        except Exception as exc:  # noqa: BLE001
            print(f"[face] Khong dung SFace ({exc}); chuyen sang LBPH")
            new_thr = _coerce_threshold("lbph", self.threshold)
            if new_thr != self.threshold:
                print(f"[face] Nguong {self.threshold:g} la thang cosin -> "
                      "doi sang 75 cho LBPH")
                self.threshold = new_thr
        if self._sf is None:
            self._lbph = _new_lbph()

    # trích đặc trưng
    def align(self, bgr: np.ndarray, landmarks) -> np.ndarray:
        """Căn mặt về 112x112 theo điểm mốc (sf.alignCrop của OpenCV)."""
        if self._sf is None or landmarks is None:
            return None
        row = Detection((0, 0, 1, 1), 1.0, landmarks).row.reshape(1, -1)
        with self._model_lock:
            return self._sf.alignCrop(bgr, row)

    def embed(self, aligned: np.ndarray) -> np.ndarray:
        """Vector 128 chiều chuẩn hoá L2 (khoảng cách cosin = 1 - tích lượng).

        Ảnh được CÂN BẰNG SÁNG trước khi trích đặc trưng (nếu bật light_norm):
        ảnh mẫu, ảnh nhận diện và ảnh nạp từ đĩa đều qua đây nên nhất quán.
        """
        if self._sf is None:
            return None
        img = normalize_light(aligned, self.light_norm)
        with self._model_lock:
            f = self._sf.feature(_as_sface_input(img)).flatten()
        return f / (np.linalg.norm(f) + 1e-9)

    def embed_frame(self, bgr, landmarks):
        """Tiện ích gộp: căn + trích vector, trả None nếu không làm được."""
        aligned = self.align(bgr, landmarks)
        return None if aligned is None else self.embed(aligned)

    # huấn luyện
    @property
    def trained(self) -> bool:
        return self._trained

    def retrain(self) -> None:
        """Đọc lại toàn bộ mẫu trong data/faces/<id>/ và dựng lại bộ so khớp.

        Thư mục của học sinh ĐÃ XOÁ (không còn trong DB) bị bỏ qua + xoá luôn:
        nếu không, người lạ có thể khớp vào mẫu của người đã xoá (điểm danh ma).
        """
        db.init_db()      # bảng phải tồn tại (để engine dùng được ngoài server.py)
        known = {int(s["id"]) for s in db.list_students()}
        vectors, grays, counts = {}, {}, {}
        if db.FACE_DIR.exists():
            for folder in sorted(db.FACE_DIR.iterdir()):
                if not folder.is_dir():
                    continue
                try:
                    sid = int(folder.name)
                except ValueError:
                    continue
                if sid not in known:
                    print(f"[face] Thu muc mao data/faces/{sid} (khong co hoc sinh"
                          " nay trong DB) -> bo qua, khong nap vao gallery")
                    continue
                vlist, glist = [], []
                for path in sorted(folder.glob("*.jpg")):
                    img = cv2.imread(str(path))
                    if img is None or img.size == 0:
                        continue
                    if self._sf is not None:
                        emb = self.embed(_as_sface_input(img))
                        if emb is not None:
                            vlist.append(emb)
                    else:
                        glist.append(cv2.resize(
                            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), LBPH_SIZE))
                if vlist:
                    vectors[sid] = np.asarray(vlist, np.float32)
                elif glist:
                    grays[sid] = glist
                counts[sid] = len(vlist or glist)

        with self._lock:
            self.gallery, self.gallery_gray = vectors, grays
            self.sample_counts = counts
            self._trained = bool(vectors or grays)
            if self._lbph is not None and grays:
                self._lbph = _new_lbph()
                self._lbph.update(
                    np.array([g for lst in grays.values() for g in lst]),
                    np.array([sid for sid, lst in grays.items() for _ in lst]))
            self.per_person = self._compute_adaptive()

        for st in db.list_students():          # đồng bộ số mẫu với DB
            db.set_sample_count(st["id"], counts.get(st["id"], 0))

    def _compute_adaptive(self) -> dict:
        """Ngưỡng riêng = trung bình + 2.5 * độ lệch của khoảng cách nội bộ."""
        if not self.adaptive or self.metric != "cosine":
            return {}
        out = {}
        for sid, mat in self.gallery.items():
            if len(mat) < 4:
                continue
            n = len(mat)
            iu = np.triu_indices(n, 1)
            d = 1.0 - (mat @ mat.T)[iu]
            d = d[np.isfinite(d)]
            if d.size < 6:
                continue
            thr = float(d.mean() + 2.5 * d.std())
            out[sid] = float(np.clip(thr, self.threshold * 0.75, self.threshold))
        return out

    def threshold_for(self, sid) -> float:
        """Ngưỡng áp dụng cho 1 người (có ngưỡng riêng nếu đủ mẫu)."""
        return float(self.per_person.get(sid, self.threshold))

    # so khớp
    def rank_for(self, emb: np.ndarray):
        """Xếp hạng mọi người theo khoảng cách nhỏ nhất tới mẫu của họ.

        Khoảng cách cosin = 1 - tích lượng; muốn lấy người GẦN NHẤT thì phải
        lấy 1 - MAX(tích lượng). Trả về 1 - MIN(...) sẽ ra khoảng cách tới mẫu
        XA NHẤT, khiến mọi khung đều bị coi là "không khớp".
        """
        if self.metric != "cosine" or emb is None:
            return []
        with self._lock:
            items = [(sid, float(1.0 - (emb @ mat.T).max()))
                     for sid, mat in self.gallery.items()]
        items.sort(key=lambda kv: kv[1])
        return items

    def dedup_distance(self, sid, emb) -> float:
        """Khoảng cách gần nhất tới mẫu đã có của `sid` (dùng khi đăng ký thêm)."""
        with self._lock:
            mat = self.gallery.get(int(sid))
        if mat is None or not len(mat) or emb is None:
            return 1.0
        return float(np.min(1.0 - mat @ emb))

    def match_gray(self, gray):
        """Đường LBPH: trả (sid, khoảng_cách)."""
        with self._lock:
            if not self._trained or self._lbph is None:
                return None, None
            label, dist = self._lbph.predict(cv2.resize(gray, LBPH_SIZE))
        return int(label), float(dist)

    # thống kê
    def info(self) -> dict:
        with self._lock:
            return {
                "recognizer": self.name,
                "metric": self.metric,
                "threshold": round(self.threshold, 3),
                "adaptive": bool(self.per_person),
                "identities": len(self.gallery) + len(self.gallery_gray),
                "faces_loaded": int(sum(self.sample_counts.values())),
            }


# theo dõi
@dataclass(eq=False)   # eq=False để track dùng được làm khoá dict (định danh theo object)
class _Track:
    """Một khuôn mặt đang được theo dõi qua nhiều khung hình."""

    tid: int
    box: tuple = (0.0, 0.0, 0.0, 0.0)     # hộp thẳng (x, y, w, h)
    sq: tuple = (0.0, 0.0, 0.0)           # ô vuông đã làm mượt (x, y, cạnh)
    vel: tuple = (0.0, 0.0)              # vận tốc tâm (px/khung)
    emb: np.ndarray = None               # vector đã trộn dần theo thời gian
    fresh: bool = False                  # khung vừa rồi có detect khớp không
    age: int = 0                         # số khung đã sống
    hits: int = 0                        # số lần được detect -> độ tin cậy
    det_score: float = 1.0
    label: int = None                    # student_id đã chốt
    label_hits: int = 0                  # số khung liên tiếp khớp
    alt_label: int = None                # ứng viên đang chờ xác nhận
    alt_hits: int = 0
    match_dist: float = None             # khoảng cách tới người khớp
    match_gap: float = None              # cách biệt so với người thứ hai
    trk: object = None                   # tracker nối frame hụt (tạo lazily)


# Factory tracker. MedianFlow nhanh nhất và đủ tốt cho ô mặt; KCF chắc hơn;
# CSRT chính xác nhất nhưng rất nặng (~90ms/khung/mặt) -> chỉ dùng khi thật cần.
def _tracker_factory(name: str):
    name = (name or "medianflow").lower()
    if name == "none":
        return None
    if name == "csrt":
        return getattr(cv2, "TrackerCSRT_create", None)
    if name == "kcf":
        return getattr(cv2, "TrackerKCF_create", None)
    legacy = getattr(getattr(cv2, "legacy", None), "TrackerMedianFlow_create", None)
    if legacy is not None:
        return legacy
    return getattr(cv2, "TrackerKCF_create", None)


class FaceTracker:
    """Ghép detect + vector thành các track ổn định, vẽ ô vuông mượt.

    Nguyên tắc chống nhầm: trong một khung hình, hai track khác nhau KHÔNG BAO
    GIỜ được gán cùng một student_id.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._next_id = 0
        self.tracks = []
        self._factory = _tracker_factory(cfg.get("track_backend", "medianflow"))

    def reset(self):
        self.tracks = []
        self._next_id = 0

    def set_backend(self, name):
        """Đổi loại tracker phụ và xoá track cũ (vì chúng dùng factory cũ)."""
        self.cfg["track_backend"] = (name or "medianflow").strip().lower()
        self._factory = _tracker_factory(self.cfg["track_backend"])
        self.reset()

    # tiện ích
    def _init_tracker(self, frame, box):
        """Tạo tracker khi cần. Không tạo sẵn: khởi tạo tracker khá tốn CPU."""
        if self._factory is None or frame is None:
            return None
        try:
            x, y, w, h = (int(round(v)) for v in box)
            if w < 12 or h < 12:
                return None
            trk = self._factory()
            return trk if trk.init(frame, (x, y, w, h)) else None
        except (cv2.error, AttributeError, TypeError):
            return None

    def _blend_embedding(self, t: _Track, emb):
        """Trộn vector mới vào vector của track (EMA) -> nhận dạng ổn định hơn."""
        if emb is None:
            return
        t.emb = emb if t.emb is None else t.emb * 0.6 + emb * 0.4
        n = np.linalg.norm(t.emb)
        if n > 1e-9:
            t.emb = (t.emb / n).astype(np.float32)

    def _predict(self, t: _Track):
        x, y, w, h = t.box
        return (x + t.vel[0], y + t.vel[1], w, h)

    def _apply(self, t: _Track, box, frame, snap: bool = False):
        """Cập nhật vị trí + làm mượt ô vuông; dựng lại CSRT khi lệch xa."""
        px, py, pw, ph = t.box
        nx, ny, nw, nh = (float(v) for v in box)
        if t.fresh:
            # giới hạn vận tốc để triệt tiêu rung của detector
            t.vel = (
                max(-80.0, min(80.0, nx + nw / 2.0 - (px + pw / 2.0))),
                max(-80.0, min(80.0, ny + nh / 2.0 - (py + ph / 2.0))),
            )
        t.box = (nx, ny, nw, nh)

        a = 1.0 if snap else max(0.0, min(1.0, float(self.cfg["track_smooth"])))
        sx, sy, side = t.sq
        nsx, nsy, nside = square_box(nx, ny, nw, nh, pad=float(self.cfg["square_pad"]))
        t.sq = (sx + (nsx - sx) * a, sy + (nsy - sy) * a, side + (nside - side) * a)

    # vòng lặp
    def update(self, dets, embs, frame) -> list:
        """Ghép detect mới vào track cũ. `embs[i]` có thể None (tiết kiệm CPU)."""
        cfg = self.cfg
        assigned_t, assigned_d = set(), set()

        # 1) điểm soát track <-> detect
        pairs = []
        for i, t in enumerate(self.tracks):
            pred = self._predict(t)
            for j, d in enumerate(dets):
                ov = _iou(pred, d.box)
                if ov < float(cfg["track_iou"]):
                    # mất dính hình học (người quay lưng, đi ngang nhau) -> dựa vào vector
                    emb = embs[j] if j < len(embs) else None
                    if t.emb is None or emb is None or ov < 0.05:
                        continue
                    if 1.0 - float(np.dot(t.emb, emb)) > float(cfg["track_emb"]):
                        continue
                pairs.append((-ov, i, j))
        pairs.sort()

        # 2) ghép tham lam: mỗi track nhận tối đa 1 detect và ngược lại
        for _, i, j in pairs:
            if i in assigned_t or j in assigned_d:
                continue
            assigned_t.add(i)
            assigned_d.add(j)
            t = self.tracks[i]
            t.fresh = True
            t.age = 0          # đếm số khung LIÊN TIẾP mất detect, không phải tổng
            t.hits += 1
            t.det_score = max(t.det_score * 0.7, dets[j].score)
            self._blend_embedding(t, embs[j])
            self._apply(t, dets[j].box, frame)

        # 3) track không khớp detect -> tracker nối tiếp (cầu nối lúc mất dặt)
        for i, t in enumerate(self.tracks):
            if i in assigned_t:
                continue
            t.fresh = False
            if t.trk is None:
                t.trk = self._init_tracker(frame, t.box)   # tạo lazily, chỉ khi thật cần
            newbox = None
            if t.trk is not None and frame is not None:
                try:
                    ok, box = t.trk.update(frame)
                    if ok:
                        bx, by, bw, bh = (float(v) for v in box)
                        if min(bw, bh) >= 16:
                            newbox = (bx, by, bw, bh)
                except cv2.error:
                    t.trk = None
            if newbox is None:
                newbox = self._predict(t)      # không có gì hơn thì quy đoán theo quán tính
            if _iou(newbox, t.box) < 0.03:
                t.trk = None                    # lệch quá xa -> bỏ tracker, dựng lại sau
            self._apply(t, newbox, frame)

        # 4) detect không khớp track nào -> track mới
        for j, d in enumerate(dets):
            if j in assigned_d:
                continue
            self._next_id += 1
            t = _Track(tid=self._next_id, box=tuple(float(v) for v in d.box),
                       det_score=d.score, hits=1, fresh=True)
            t.sq = square_box(*t.box, pad=float(self.cfg["square_pad"]))
            self._blend_embedding(t, embs[j])
            self.tracks.append(t)

        # 5) dọn track chết, cập nhật tuổi
        alive = []
        for t in self.tracks:
            t.age += 1
            if t.fresh or t.age <= int(cfg["track_max_age"]):
                alive.append(t)
        self.tracks = alive
        return self.tracks

    # quyết định
    @staticmethod
    def vote(t: _Track, sid, dist, gap, need: int):
        """Chốt tên ổn định theo thời gian để nhãn không nhấp nháy."""
        if sid is not None:
            if sid == t.label:
                t.label_hits = min(t.label_hits + 1, need + 2)
            else:
                if sid == t.alt_label:
                    t.alt_hits += 1
                else:
                    t.alt_label, t.alt_hits = sid, 1
                if t.alt_hits >= need:
                    t.label, t.label_hits = sid, need
                    t.alt_label, t.alt_hits = None, 0
            t.match_dist, t.match_gap = dist, gap
            return

        t.alt_label, t.alt_hits = None, 0
        if t.label is not None:
            # mất khớp -> phải xác nhận lại trước khi hiện lại tên
            t.label_hits -= 1
            if t.label_hits <= 0:
                t.label, t.label_hits = None, 0

    @staticmethod
    def confirmed(t: _Track, need: int) -> bool:
        return t.label is not None and t.label_hits >= need

    @staticmethod
    def displayed(t: _Track, need: int):
        """(student_id hiển thị, đã chốt chưa) — tên chỉ hiện khi đã xác nhận."""
        if FaceTracker.confirmed(t, need):
            return t.label, True
        if t.alt_label is not None:
            return t.alt_label, False
        return None, False


# engine
class FaceEngine:
    """Điểm vào cho server: nhận khung hình -> phát hiện -> nhận dạng -> vẽ."""

    def __init__(self, threshold: float = DEFAULTS["match_threshold"], **overrides):
        self.cfg = dict(DEFAULTS)
        self.cfg.update({k: v for k, v in overrides.items() if k in DEFAULTS})
        self.detector = FaceDetector(self.cfg["detect_score"], self.cfg["min_face_size"])
        self.matcher = FaceMatcher(float(threshold), self.cfg["adaptive_threshold"])
        self.tracker = FaceTracker(self.cfg)
        self.cfg["match_threshold"] = self.matcher.threshold
        self.frame_no = 0
        self.timing = {"detect_ms": 0.0, "feature_ms": 0.0, "total_ms": 0.0}
        # nạp sẵn mẫu đã đăng ký trong data/faces/, nếu không khởi động lại server
        # sẽ quên hết mọi người đã đăng ký
        self.retrain()

    # cấu hình
    @property
    def threshold(self) -> float:
        return self.matcher.threshold

    @threshold.setter
    def threshold(self, value):
        new = _coerce_threshold(self.matcher.metric, value)
        if new != float(value):
            print(f"[face] Nguong {float(value):g} la thang cosin nhung dang chay "
                  f"{self.matcher.metric} -> doi sang {new:g}")
        self.matcher.threshold = new
        self.cfg["match_threshold"] = new
        self.matcher.retrain()      # ngưỡng riêng từng người phụ thuộc ngưỡng toàn cục

    @property
    def trained(self) -> bool:
        return self.matcher.trained

    def configure(self, **kw) -> list:
        """Đổi cấu hình runtime. Trả về danh sách khoá đã đổi (đã ép kiểu)."""
        changed, need_retrain = [], False
        for key, val in kw.items():
            if key not in DEFAULTS or val is None or val == "":
                continue
            if key == "match_threshold":
                new = _coerce_threshold(self.matcher.metric, val)
                if new == self.matcher.threshold:
                    continue
                self.matcher.threshold = new
                self.cfg[key] = new
                changed.append(key)
                need_retrain = True
                continue
            new = self._cast(key, val)
            if new == self.cfg.get(key):
                continue
            self.cfg[key] = new
            if key == "adaptive_threshold":
                self.matcher.adaptive = new
                need_retrain = True
            elif key == "light_norm":
                self.matcher.light_norm = bool(new)
                need_retrain = True
            elif key == "detect_score":
                self.detector.score_thresh = new
            elif key == "min_face_size":
                self.detector.min_face_size = new
            elif key == "track_backend":
                self.tracker.set_backend(new)
            changed.append(key)
        if need_retrain:
            self.matcher.retrain()
        return changed

    @staticmethod
    def _cast(key, val):
        cur = DEFAULTS[key]
        if isinstance(cur, bool):
            if isinstance(val, str):
                return val.strip().lower() in ("1", "true", "yes", "on")
            return bool(val)
        if key in STR_KEYS:
            return str(val).strip().lower()
        if key in INT_KEYS:
            return int(float(val))
        return float(val)

    def reset_tracker(self):
        self.tracker.reset()

    def info(self) -> dict:
        return {
            "detector": self.detector.name,
            "landmark_align": self.detector.has_landmarks,
            "tracking": bool(self.cfg["track_enabled"]),
            "tracks": len(self.tracker.tracks),
            "threshold_margin": round(float(self.cfg["match_margin"]), 3),
            "adaptive_threshold": bool(self.matcher.per_person),
            "timing_ms": {k: round(v, 1) for k, v in self.timing.items()},
            **self.matcher.info(),
        }

    # phát hiện
    def _detect(self, frame, for_sample: bool = False):
        """Detector trên ảnh đã đưa về bề rộng chuẩn; kết quả trong toạ độ khung gốc.

        Phóng nhỏ ảnh (camera 320x240, 640x480) giúp YuNet bắt được mặt nhỏ hơn;
        thu lớn ảnh (camera 2K) giúp tiết kiệm thời gian. Cả hai đều giữ nguyên
        tỉ lệ, và toạ độ trả về luôn quy về khung gốc.
        """
        h, w = frame.shape[:2]
        target = 0 if for_sample else int(self.cfg["detect_width"])
        scale_x = scale_y = 1.0
        small = frame
        if target > 0 and w != target:
            k = target / float(w)
            interp = cv2.INTER_AREA if k < 1.0 else cv2.INTER_LINEAR
            small = cv2.resize(frame, (target, max(1, int(round(h * k)))),
                               interpolation=interp)
            scale_x, scale_y = w / float(small.shape[1]), h / float(small.shape[0])
        dets = self.detector.detect(small)
        if (scale_x, scale_y) != (1.0, 1.0):
            dets = [_rescale(d, scale_x, scale_y, w, h) for d in dets]
        return dets

    def _blur_of(self, gray, box) -> float:
        x, y, w, h = (int(v) for v in box)
        crop = gray[max(0, y):y + h, max(0, x):x + w]
        return _sharpness(crop)

    def face_quality(self, frame, box):
        """(cạnh_px_quy_ve_640, độ_nét) của vùng mặt.

        Cạnh được quy về khung rộng 640 để ngưỡng không phụ thuộc độ phân giải
        nguồn (ESP32-CAM VGA, webcam 720p, điện thoại 2K...).
        """
        x, y, w, h = (int(v) for v in box)
        frame_w = int(frame.shape[1]) if frame is not None else 640
        side = min(max(0, w), max(0, h)) * (640.0 / max(1, frame_w))
        if frame is None or w <= 0 or h <= 0:
            return side, 0.0
        crop = frame[max(0, y):y + h, max(0, x):x + w]
        if crop.size < 64:
            return side, 0.0
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return side, _sharpness(gray)

    def _quality_reject(self, frame, box):
        """Lý do KHÔNG đủ tin cậy để chốt tên: "small" | "blur" | None (đạt)."""
        side, blur = self.face_quality(frame, box)
        if side < float(self.cfg["recognize_min_face"]):
            return "small"
        if blur < float(self.cfg["recognize_blur_min"]):
            return "blur"
        return None

    # so khớp
    def _resolve_frame(self, per_track):
        """Chọn sao cho mỗi track trong khung: gần nhất thắng, mỗi người chỉ 1 ô.

        `per_track` = [(track, [(sid, khoảng_cách), ...]), ...]
        -> {(track): (sid, khoảng_cách, cách_biệt_người_thứ_2)}
        """
        margin = float(self.cfg["match_margin"])
        allowed = []
        for track, ranked in per_track:
            cands = [(sid, d) for sid, d in ranked if d <= self.matcher.threshold_for(sid)]
            allowed.append((track, cands))

        pool = sorted(((d, i, sid)
                       for i, (_track, cands) in enumerate(allowed)
                       for sid, d in cands),
                      key=lambda t: (t[0], t[1]))
        out, taken_sid, taken_row = {}, set(), set()
        for d, i, sid in pool:
            if sid in taken_sid or i in taken_row:
                continue
            taken_sid.add(sid)
            taken_row.add(i)
            out[i] = (sid, d)

        result = {}
        for i, (track, cands) in enumerate(allowed):
            if i not in out:
                continue
            sid, d = out[i]
            others = [dd for ss, dd in cands if ss != sid]
            gap = (min(others) - d) if others else None
            if margin > 0 and gap is not None and gap < margin:
                continue      # gần bằng ứng viên thứ hai -> dễ nhầm, không chốt
            result[track] = (sid, d, gap)
        return result

    @staticmethod
    def _student(sid):
        return db.get_student(sid) if sid else None

    @staticmethod
    def _pack(box, sq, score, sid, dist, match, extra=None):
        """`sq` là (x, y, cạnh); xuất ra JSON theo dạng [x, y, w, h] cho cả hai ô."""
        st = FaceEngine._student(sid) if match else None
        # Học sinh có thể vừa bị xoá lúc track đã chốt tên -> coi như KHÔNG nhận
        # diện được. Nếu không, kết quả sẽ mang student_id khong tồn tại và
        # full_name = None, làm ghi điểm danh hỏng (vi phạm NOT NULL).
        if match and st is None:
            match, sid, dist = False, None, None
        side = int(round(max(1.0, sq[2])))
        out = {
            "box": [int(round(v)) for v in box],
            "square": [int(round(sq[0])), int(round(sq[1])), side, side],
            "student_id": sid if match else None,
            "full_name": st["full_name"] if st else None,
            "date_of_birth": st["date_of_birth"] if st else None,
            "class_name": st["class_name"] if st else None,
            "confidence": round(float(dist), 3) if dist is not None else None,
            "score": round(float(score), 3),
            "match": bool(match),
        }
        if extra:
            out.update(extra)
        return out

    # xử lý
    def process(self, frame, draw: bool = True, track: bool = True):
        """Xử lý một khung hình. Trả về (kết quả, ảnh).

        `track=False` dùng cho ảnh đơn lẻ (/api/recognize, ESP32): không nhớ trạng
        thái nên kết quả luôn tái lập được với cùng một ảnh.
        """
        t_start = time.perf_counter()
        if frame is None:
            return [], frame
        frame = np.ascontiguousarray(frame)
        self.frame_no += 1
        use_track = bool(track and self.cfg["track_enabled"])
        need = max(1, int(self.cfg["track_confirm"]))
        cosine = self.matcher.metric == "cosine"
        gray = None if cosine else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) phát hiện (bỏ qua detector ở khung lẻ để giảm ~40% CPU; tracker cầu.
        # Không bỏ khi chưa có track nào — vừa reset mà bỏ thì khung trắng).
        t0 = time.perf_counter()
        skip_det = (use_track and int(self.cfg["detect_every"]) > 1
                    and self.frame_no % int(self.cfg["detect_every"])
                    and len(self.tracker.tracks) > 0)
        dets = [] if skip_det else self._detect(frame)
        t_det = (time.perf_counter() - t0) * 1000.0

        # 2) trích vector (giảm tần suất khi đã có tracking)
        t0 = time.perf_counter()
        embs = [None] * len(dets)
        if cosine and self.detector.has_landmarks:
            every = int(self.cfg["feature_every"]) if use_track else 1
            for i, d in enumerate(dets):
                if d.landmarks is None or (use_track and every > 1 and self.frame_no % every):
                    continue
                embs[i] = self.matcher.embed_frame(frame, d.landmarks)
        t_feat = (time.perf_counter() - t0) * 1000.0

        # 3) ghép track -> nhận dạng -> vẽ
        if use_track:
            results = self._results_from_tracks(self.tracker.update(dets, embs, frame), need)
        else:
            results = self._results_from_detections(dets, embs, gray, frame)

        total = (time.perf_counter() - t_start) * 1000.0
        a = 0.15                                    # trung bình trượt cho số đo
        self.timing["detect_ms"] += a * (t_det - self.timing["detect_ms"])
        self.timing["feature_ms"] += a * (t_feat - self.timing["feature_ms"])
        self.timing["total_ms"] += a * (total - self.timing["total_ms"])

        if draw:
            self.draw_results(frame, results)
        return results, frame

    def _results_from_tracks(self, tracks, need):
        per_track = []
        for t in tracks:
            ranked = self.matcher.rank_for(t.emb)
            if ranked:
                per_track.append((t, ranked))
        chosen = self._resolve_frame(per_track)

        results = []
        for t in tracks:
            sid, dist, gap = chosen.get(t, (None, None, None))
            FaceTracker.vote(t, sid, dist, gap, need)
            show_sid, locked = FaceTracker.displayed(t, need)
            results.append(self._pack(
                t.box, t.sq, t.det_score, show_sid, t.match_dist, locked,
                extra={
                    "track_id": int(t.tid),
                    "hits": int(t.hits),
                    "confirming": bool(show_sid is not None and not locked),
                    "match_margin": round(t.match_gap, 3) if t.match_gap is not None else None,
                }))
        return results

    def _results_from_detections(self, dets, embs, gray, frame=None):
        per_det = []
        for i, d in enumerate(dets):
            ranked = self.matcher.rank_for(embs[i])
            if ranked:
                per_det.append((i, ranked))
        chosen = self._resolve_frame(per_det)

        results = []
        for i, d in enumerate(dets):
            sid, dist, _gap = chosen.get(i, (None, None, None))
            match = sid is not None
            if not match and gray is not None:      # đường LBPH (không có model SFace)
                sid2, dist2 = self.matcher.match_gray(frame_slice(gray, d.box))
                if sid2 is not None and dist2 is not None and dist2 <= self.matcher.threshold:
                    sid, dist, match = sid2, dist2, True

            side, blur = self.face_quality(frame, d.box)
            extra = {"track_id": None, "confirming": False,
                     "face_size": int(round(side)),
                     "blur": round(blur, 1) if frame is not None else
                             (round(self._blur_of(gray, d.box), 1) if gray is not None else None)}
            # Chống nhận nhầm người lạ: mặt quá nhỏ/mờ thì KHÔNG chốt tên,
            # dù khoảng cách cosin có dưới ngưỡng. Giữ lại candidate_id + dist
            # để soi log, nhưng match=False nên không ghi điểm danh.
            reject = self._quality_reject(frame, d.box)
            if match and reject:
                match = False
                extra["reject"] = reject
                extra["candidate_id"] = sid
                extra["candidate_dist"] = round(float(dist), 3) if dist is not None else None

            sq = square_box(*d.box, pad=float(self.cfg["square_pad"]))
            results.append(self._pack(
                d.box, sq, d.score, sid if match else None, dist, match,
                extra=extra))
        return results

    # vẽ
    def draw_results(self, frame, results):
        """Vẽ ô vuông đã làm mượt + nhãn tiếng Việt."""
        h, w = frame.shape[:2]
        pad = 4
        labels = []
        for r in results:
            x0, y0, x1, y1 = fit_square(r["square"], w, h)
            if r["match"]:
                color_bgr, text_rgb = (80, 220, 80), (150, 245, 150)
            elif r.get("confirming"):
                color_bgr, text_rgb = (60, 200, 235), (160, 230, 250)
            else:
                color_bgr, text_rgb = (70, 70, 230), (255, 155, 155)

            thickness = max(2, int(round((x1 - x0) * 0.012)))
            cv2.rectangle(frame, (x0, y0), (x1, y1), color_bgr, thickness)
            # góc vuông đậm: dễ bám mắt khi khung có nhiều người
            tick = max(6, int(round((x1 - x0) * 0.24)))
            bold = thickness + 1
            for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                                   (x0, y1, 1, -1), (x1, y1, -1, -1)):
                cv2.line(frame, (cx, cy), (cx + dx * tick, cy), color_bgr, bold)
                cv2.line(frame, (cx, cy), (cx, cy + dy * tick), color_bgr, bold)

            if r["match"]:
                label = f"{r['full_name']} | {r['class_name']} | {r['date_of_birth']}"
                if r["confidence"] is not None:
                    label += f" ({r['confidence']:.2f})"
            elif r.get("confirming"):
                label = "Dang xac nhan..."
            else:
                label = "Chua nhan dien"
            if r.get("track_id") is not None:
                label = f"#{r['track_id']} {label}"
            # đặt nhãn ngoài ô vuông: ưu tiên bên trên, không đủ chỗ thì xuống dưới
            need = _text_height(frame, label) + 2 * pad + 2
            if y0 - need >= 0:
                labels.append((label, (x0 + 2, y0 - pad - 2), text_rgb, "bottom"))
            elif y1 + need <= h:
                labels.append((label, (x0 + 2, y1 + pad + 2), text_rgb, "top"))
            else:
                labels.append((label, (x0 + 2, max(0, y0 - pad - 2)), text_rgb, "bottom"))
        draw_labels(frame, labels, pad=pad)
        return frame

    # mẫu
    def _candidate_faces(self, bgr):
        """Mọi mặt trong ảnh, kèm điểm chọn.

        Nhiều người trong khung là tình huống dễ đăng ký nhầm người, nên thứ tự ưu
        tiên là: (1) mặt ở GIỮA khung - người đang đăng ký thường đứng chính giữa;
        (2) diện tích lớn hơn - tức người đứng gần camera hơn;
        (3) mặt nét hơn. Người dùng vẫn có thể chỉ định `face_index` để chọn tay.
        """
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        min_sz = int(self.cfg["sample_min"])
        out = []
        for d in self._detect(bgr, for_sample=True):
            if d.box[2] < min_sz or d.box[3] < min_sz:
                continue
            cx, cy = d.box[0] + d.box[2] / 2.0, d.box[1] + d.box[3] / 2.0
            center = 1.0 - 0.5 * (abs(cx - w / 2.0) / (w / 2.0)
                                  + abs(cy - h / 2.0) / (h / 2.0))
            center = max(0.0, min(1.0, center))
            sharp = 1.0 + min(1.0, self._blur_of(gray, d.box) / 250.0)
            score = (d.box[2] * d.box[3] * (0.25 + 0.75 * center) * (0.85 + 0.15 * sharp))
            out.append((score, d))
        out.sort(key=lambda t: -t[0])
        return out

    def _aligned_sample(self, bgr, det):
        """Cắt + căn mặt về ảnh mẫu 112x112; None nếu không căn được."""
        if self.detector.has_landmarks and det.landmarks is not None:
            aligned = self.matcher.align(bgr, det.landmarks)
            if aligned is not None:
                return _as_sface_input(aligned)
        x, y, w, h = det.box
        crop = bgr[max(0, y):y + h, max(0, x):x + w]
        return _as_sface_input(crop) if crop.size else None

    def add_samples(self, student_id: int, frames, face_index=None) -> dict:
        """Lưu mẫu mặt cho một người từ các khung hình.

        - Chọn đúng người cần đăng ký khi ảnh có nhiều mặt (ưu tiên vùng giữa khung).
        - Bỏ mẫu quá nhỏ, quá mờ, hoặc gần trùng mẫu đã có.

        Trả {"faces": k, "saved": n, "skipped": m}:
          faces   = số ảnh tìm thấy mặt dùng được
          saved   = số mẫu mới đã lưu
          skipped = số ảnh bị bỏ (không có mặt / quá mờ / trùng mẫu cũ)
        `faces > 0` mà `saved == 0` nghĩa là ảnh OK nhưng trùng mẫu đã có.
        """
        sid = int(student_id)
        folder = db.FACE_DIR / str(sid)
        folder.mkdir(parents=True, exist_ok=True)
        dedup, blur_min = float(self.cfg["sample_dedup"]), float(self.cfg["blur_min"])
        flip = bool(self.cfg["augment_flip"])
        saved = skipped = faces = 0

        for frame in frames:
            if frame is None:
                continue
            frame = np.ascontiguousarray(frame)
            cands = self._candidate_faces(frame)
            if not cands:
                skipped += 1
                continue
            idx = int(face_index) if face_index is not None else 0
            det = cands[idx][1] if 0 <= idx < len(cands) else cands[0][1]

            sample = self._aligned_sample(frame, det)
            if sample is None:
                skipped += 1
                continue
            if self._blur_of(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), det.box) < blur_min:
                skipped += 1
                continue
            faces += 1                       # đã thấy mặt dùng được trong ảnh này
            emb = self.matcher.embed(sample)
            if emb is not None and dedup > 0 \
                    and self.matcher.dedup_distance(sid, emb) < dedup:
                skipped += 1                      # gần như trùng mẫu cũ
                continue
            idx = len(list(folder.glob("*.jpg"))) // (2 if flip else 1)
            if not cv2.imwrite(str(folder / f"{idx:03d}.jpg"), sample):
                continue
            if flip:
                cv2.imwrite(str(folder / f"{idx:03d}_flip.jpg"), cv2.flip(sample, 1))
            saved += 1

        if saved:
            self.retrain()
        return {"faces": faces, "saved": saved, "skipped": skipped}

    def diagnose_sample(self, frame) -> dict:
        """Vì sao ảnh này không dùng làm mẫu được — CHỈ gọi khi đăng ký thất bại.

        Trả {"w","h","n","max_size","blur","need_size","need_blur"}:
          n         = số mặt thấy ở ngưỡng nới lỏng (0.3)
          max_size  = cạnh mặt lớn nhất (px, theo ảnh gốc)
          blur      = độ nét mặt lớn nhất (Laplacian, cần >= need_blur)
          need_size = cạnh tối thiểu để nhận làm mẫu (sample_min)
        """
        info = {"w": 0, "h": 0, "n": 0, "max_size": 0, "blur": 0.0,
                "need_size": int(self.cfg["sample_min"]),
                "need_blur": float(self.cfg["blur_min"])}
        if frame is None or getattr(frame, "size", 0) == 0:
            return info
        h, w = frame.shape[:2]
        info["w"], info["h"] = int(w), int(h)
        boxes = self.detector.detect_raw(np.ascontiguousarray(frame), 0.3)
        info["n"] = len(boxes)
        if not boxes:
            return info
        boxes.sort(key=lambda b: -min(b[2], b[3]))
        x, y, bw, bh = boxes[0]
        info["max_size"] = int(min(bw, bh))
        try:
            gray = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2GRAY)
            info["blur"] = round(float(self._blur_of(gray, (x, y, bw, bh))), 1)
        except Exception:
            pass
        return info

    def retrain(self):
        self.matcher.retrain()


def frame_slice(gray, box):
    """Cắt vùng ảnh theo hộp, luôn trả về mảng 2 chiều không rỗng."""
    x, y, w, h = (int(v) for v in box)
    crop = gray[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]
    return crop if crop.size else gray[:1, :1]


def _rescale(det: Detection, sx, sy, w, h) -> Detection:
    """Đưa hộp và điểm mốc từ ảnh đã thu nhỏ về toạ độ ảnh gốc."""
    x, y, bw, bh = det.box
    nx = int(round(min(max(x * sx, 0.0), w - 1.0)))
    ny = int(round(min(max(y * sy, 0.0), h - 1.0)))
    nw = int(round(min(max(bw * sx, 1.0), w - nx)))
    nh = int(round(min(max(bh * sy, 1.0), h - ny)))
    lm = None
    if det.landmarks is not None:
        lm = np.asarray(det.landmarks, np.float32) * np.float32([sx, sy])
    return Detection((nx, ny, nw, nh), det.score, lm)


# giữ tương thích với code cũ (trả ảnh xám 200x200)
def _largest_face_gray(self, bgr):  # pragma: no cover
    cands = self._candidate_faces(np.ascontiguousarray(bgr))
    if not cands:
        return None
    sample = self._aligned_sample(bgr, cands[0][1])
    return None if sample is None else cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)


FaceEngine._largest_face_gray = _largest_face_gray