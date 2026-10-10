"""GeeTest v4「word」类验证码求解器（iKuuu 当前下发的交互验证码之一）。

流程：
1. RapidOCR 定位主图里的文字区域，顺带拿到识别文字用于约束搜索范围；
2. 题目小图（带透明通道的 PNG）取出字形掩码，并 OCR 出目标汉字；
3. 每个目标字在候选区域内做多尺度、多角度模板匹配（多种前景图取最优）；
4. 候选点池 + 互斥分配，避免两个题目落到同一个字上；
5. 输出归一化到 0~10000 的点击坐标。

只依赖 numpy / opencv / rapidocr-onnxruntime。
"""

import math

import cv2
import numpy as np
import requests
from rapidocr_onnxruntime import RapidOCR

STATIC_BASE = 'https://static.geetest.com/'
GLYPH_HEIGHT = 64
_ENGINE = None


def engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RapidOCR()
    return _ENGINE


def download(url, timeout=30):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def decode_image(raw, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.frombuffer(raw, np.uint8), flags)


def prompt_glyph(raw, target_h=GLYPH_HEIGHT):
    """题目小图 -> 归一化高度的字形掩码。"""
    img = decode_image(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        rgb = img[:, :, :3]
    else:
        alpha = np.full(img.shape[:2], 255, np.uint8)
        rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    mask = ((alpha > 40) & (gray < 160)).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = mask.shape
    scale = target_h / float(h)
    mask = cv2.resize(mask, (max(8, int(round(w * scale))), target_h), interpolation=cv2.INTER_AREA)
    return (mask > 127).astype(np.uint8)


def prompt_text(raw):
    """题目小图铺白底放大后 OCR，返回识别到的文字。"""
    img = decode_image(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        return ''
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = (img[:, :, :3].astype(np.float32) * alpha + 255 * (1 - alpha)).astype(np.uint8)
    else:
        rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    big = cv2.resize(rgb, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode('.png', big)
    if not ok:
        return ''
    result, _ = engine()(buf.tobytes())
    if not result:
        return ''
    return ''.join(item[1] for item in result).strip()


def foreground_maps(img):
    """多种前景图，匹配时逐个尝试取最优（每张验证码的字色和背景都不一样）。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    maps = {
        'gray': g,
        'inv_gray': 255.0 - g,
        'lab_a': lab[:, :, 1].astype(np.float32),
        'inv_lab_a': 255.0 - lab[:, :, 1].astype(np.float32),
        'lab_b': lab[:, :, 2].astype(np.float32),
        'inv_lab_b': 255.0 - lab[:, :, 2].astype(np.float32),
        'sat': hsv[:, :, 1].astype(np.float32),
        'inv_sat': 255.0 - hsv[:, :, 1].astype(np.float32),
        'tophat': cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel).astype(np.float32),
        'blackhat': cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32),
    }
    return {name: cv2.GaussianBlur(value, (3, 3), 0) for name, value in maps.items()}


def region_candidates(maps, glyph, shape, region, topk=8, scale_step=0.15, angle_step=15):
    """在指定区域内搜索字形，返回若干高分候选点。"""
    height, width = shape[:2]
    x1, y1, x2, y2 = region
    pool = {}
    for name, fmap in maps.items():
        src = fmap[y1:y2, x1:x2]
        if src.size == 0:
            continue
        for scale in np.arange(0.55, 1.36, scale_step):
            h = int(glyph.shape[0] * scale)
            w = int(glyph.shape[1] * scale)
            if h < 14 or w < 14 or h >= src.shape[0] or w >= src.shape[1]:
                continue
            base = cv2.resize(glyph * 255, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
            for angle in range(0, 360, angle_step):
                matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
                cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
                nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
                matrix[0, 2] += nw / 2.0 - w / 2.0
                matrix[1, 2] += nh / 2.0 - h / 2.0
                template = cv2.warpAffine(base, matrix, (nw, nh), flags=cv2.INTER_LINEAR)
                if template.shape[0] >= src.shape[0] or template.shape[1] >= src.shape[1]:
                    continue
                scores = cv2.matchTemplate(src, template, cv2.TM_CCOEFF_NORMED)
                local_max = cv2.dilate(scores, np.ones((7, 7), np.float32))
                peaks = np.argwhere((scores == local_max) & (scores > 0.35))
                for py, px in peaks[:6]:
                    cx = x1 + px + nw / 2.0
                    cy = y1 + py + nh / 2.0
                    if not (0 <= cx < width and 0 <= cy < height):
                        continue
                    key = (int(cx // 10), int(cy // 10))
                    score = float(scores[py, px])
                    if key not in pool or score > pool[key]['score']:
                        pool[key] = {'score': score, 'cx': cx, 'cy': cy,
                                     'scale': float(scale), 'angle': angle, 'map': name}
    return sorted(pool.values(), key=lambda item: -item['score'])[:topk]


def solve_from_bytes(main_raw, prompt_raws, verbose=False):
    """返回 [[x, y], ...]，坐标已归一化到 0~10000，顺序与题目一致。"""
    img = decode_image(main_raw)
    if img is None:
        raise ValueError('主图解码失败')
    height, width = img.shape[:2]
    maps = foreground_maps(img)

    ok, buf = cv2.imencode('.jpg', img)
    result, _ = engine()(buf.tobytes() if ok else main_raw)
    regions = []
    for box, text, score in (result or []):
        pts = np.array(box, dtype=np.int32)
        regions.append({'rect': (int(pts[:, 0].min()), int(pts[:, 1].min()),
                                 int(pts[:, 0].max()), int(pts[:, 1].max())),
                        'text': text, 'score': float(score)})

    targets = []
    for raw in prompt_raws:
        glyph = prompt_glyph(raw)
        if glyph is None:
            raise ValueError('题目图解码失败')
        targets.append({'text': prompt_text(raw), 'glyph': glyph, 'candidates': []})

    for target in targets:
        candidates = []
        for region in regions:
            pad = 12
            x1, y1, x2, y2 = region['rect']
            rect = (max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad))
            # 文字能对上的区域权重更高；对不上也保留，因为主图 OCR 不一定认得出
            weight = 1.0 if (target['text'] and target['text'] in region['text']) else 0.85
            for candidate in region_candidates(maps, target['glyph'], img.shape, rect):
                item = dict(candidate)
                item['score'] *= weight
                candidates.append(item)
        candidates.sort(key=lambda item: -item['score'])
        target['candidates'] = candidates

    # 互斥分配：按分数从高到低占位，同一个字不允许被两个题目同时选中
    assigned = {}
    used = []
    flat = sorted(((c['score'], index, c)
                   for index, target in enumerate(targets)
                   for c in target['candidates']), key=lambda item: -item[0])
    for score, index, candidate in flat:
        if index in assigned:
            continue
        separation = 0.55 * GLYPH_HEIGHT * candidate['scale']
        if any(math.hypot(candidate['cx'] - ux, candidate['cy'] - uy) < separation for ux, uy in used):
            continue
        assigned[index] = candidate
        used.append((candidate['cx'], candidate['cy']))

    points = []
    for index, target in enumerate(targets):
        candidate = assigned.get(index)
        if candidate is None:
            raise ValueError('第 %d 个题目字没有匹配到候选位置' % (index + 1))
        if verbose:
            print('    题目%d 文字=%s 位置=(%d,%d) 得分=%.3f' % (
                index + 1, target['text'] or '?', candidate['cx'], candidate['cy'], candidate['score']))
        points.append([candidate['cx'] / width * 10000.0, candidate['cy'] / height * 10000.0])
    return points


def solve_from_urls(imgs, ques, timeout=30, verbose=False):
    """imgs / ques 是 GeeTest 返回的相对路径。"""
    main_raw = download(STATIC_BASE + imgs, timeout)
    prompt_raws = [download(STATIC_BASE + path, timeout) for path in (ques or [])]
    if not prompt_raws:
        raise ValueError('题目为空')
    return solve_from_bytes(main_raw, prompt_raws, verbose=verbose)
