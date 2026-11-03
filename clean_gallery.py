"""Kiểm tra và dọn thư viện mẫu mặt (data/faces/<id>/).

VẤN ĐỀ THẬT ĐÃ GẶP
------------------
Vài ảnh mẫu bị XOAY (90/180 độ) lọt vào thư viện (chụp lúc CAM còn bị ngược,
hoặc lúc đăng ký không căn được mặt). Với SFace, mặt bị xoay khác hẳn mặt
thẳng: khoảng cách cosin ~0.6-1.1 (bằng mức "người khác"), trong khi hai mẫu
bình thường của cùng người chỉ ~0.07-0.35. Hậu quả:

  * ngưỡng riêng của người đó bị tính sai -> nhận diện chập chờn;
  * đổi ánh sáng một chút là hết khớp.

CÁCH SỬA
--------
Gom nhóm các mẫu theo khoảng cách: nhóm ĐÔNG NHẤT = khuôn mặt chính (giữ lại),
các nhóm lẻ loi (bị xoay/lệch hướng/chụp nhầm) bị tách ra.

DÙNG
----
    python clean_gallery.py                 # chỉ báo cáo, KHÔNG đổi gì
    python clean_gallery.py --apply         # chuyển mẫu lạc sang <id>/_bad/
    python clean_gallery.py --apply --link 0.45   # đổi ngưỡng gộp nhóm

Sau khi --apply, khởi động lại server (engine tự nạp lại thư viện). Mẫu bị
chuyển vào <id>/_bad/ KHÔNG bị xoá - muốn khôi phục chỉ cần chuyển lại.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Windows cmd mặc định cp1252 -> in tiếng Việt sẽ lỗi; ép UTF-8 và thay ký tự lạ.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                        # noqa: BLE001
    pass

import database as db                                     # noqa: E402
import face_engine as fe                                  # noqa: E402
from face_engine import FaceEngine                        # noqa: E402


def _components(dist: np.ndarray, link: float) -> list[list[int]]:
    """Gom nhóm đơn giản: nối hai mẫu nếu khoảng cách < link (union-find)."""
    n = dist.shape[0]
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    iu = np.triu_indices(n, 1)
    for i, j in zip(*iu):
        if dist[i, j] < link:
            union(int(i), int(j))

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def scan_engine(engine: FaceEngine, link: float, apply: bool, min_keep: int,
                verbose: bool = True) -> dict:
    """Quét toàn bộ thư viện. Trả tổng kết."""
    db.init_db()
    m = engine.matcher
    if m.metric != "cosine":
        print("[!] Engine đang dùng LBPH (thiếu model SFace) -> không dọn được.")
        return {"students": 0, "moved": 0}

    total_moved = 0
    students = 0
    folders = sorted(db.FACE_DIR.iterdir()) if db.FACE_DIR.exists() else []
    for folder in folders:
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        try:
            sid = int(folder.name)
        except ValueError:
            continue
        files = sorted(folder.glob("*.jpg"))
        if len(files) < 3:
            continue

        embs, keep_paths = [], []
        for path in files:
            img = cv2.imread(str(path))
            if img is None or img.size == 0:
                continue
            emb = m.embed(fe._as_sface_input(img))
            if emb is None:
                continue
            embs.append(emb)
            keep_paths.append(path)
        if len(embs) < 3:
            continue

        mat = np.asarray(embs, np.float32)
        dist = 1.0 - (mat @ mat.T)
        np.fill_diagonal(dist, np.inf)
        nn = dist.min(axis=1)
        groups = _components(dist, link)
        main = groups[0]
        stray = [i for i in range(len(keep_paths)) if i not in set(main)]

        # Nếu nhóm chính còn quá ít thì giữ thêm nhóm lớn thứ hai
        if len(main) < min_keep and len(groups) > 1:
            main = sorted(main + groups[1])
            stray = [i for i in range(len(keep_paths)) if i not in set(main)]

        students += 1
        flag = "OK " if not stray else "CẦN DỌN"
        print(f"[{flag}] học sinh {sid}: {len(keep_paths)} mẫu, "
              f"{len(groups)} nhóm, giữ {len(main)}, lạc {len(stray)} "
              f"| NN TB={nn.mean():.3f} max={nn.max():.3f}")
        if stray and verbose:
            for i in stray:
                print(f"        lạc: {keep_paths[i].name} "
                      f"(NN={nn[i]:.3f})")

        if apply and stray:
            bad = folder / "_bad"
            bad.mkdir(exist_ok=True)
            for i in stray:
                try:
                    shutil.move(str(keep_paths[i]), str(bad / keep_paths[i].name))
                    total_moved += 1
                except OSError as exc:
                    print(f"        [!] không chuyển được {keep_paths[i].name}: {exc}")
            # Bản _flip đi kèm cũng dọn theo
            for i in stray:
                p = keep_paths[i]
                if not p.name.endswith("_flip.jpg"):
                    fp = p.with_name(p.stem + "_flip.jpg")
                    if fp.exists():
                        try:
                            shutil.move(str(fp), str(bad / fp.name))
                            total_moved += 1
                        except OSError:
                            pass

    return {"students": students, "moved": total_moved}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="thực sự chuyển mẫu lạc sang <id>/_bad/ (mặc định chỉ báo cáo)")
    ap.add_argument("--link", type=float, default=0.45,
                    help="ngưỡng gộp nhóm theo khoảng cách cosin (mặc định 0.45)")
    ap.add_argument("--min-keep", type=int, default=3,
                    help="giữ tối thiểu bao nhiêu mẫu mỗi người (mặc định 3)")
    args = ap.parse_args()

    engine = FaceEngine()
    print(f"engine: {engine.matcher.name} | light_norm={engine.matcher.light_norm} "
          f"| link={args.link:g} | apply={args.apply}")
    print("-" * 72)
    res = scan_engine(engine, args.link, args.apply, args.min_keep)
    print("-" * 72)
    if res["students"] == 0:
        print("Không tìm thấy thư viện mẫu nào (data/faces/<id>/).")
        return 0
    if args.apply:
        print(f"Đã chuyển {res['moved']} mẫu lạc sang <id>/_bad/.")
        print("=> Khởi động lại server để engine nạp lại thư viện.")
    else:
        print("Chỉ báo cáo. Thêm --apply để thực sự dọn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
