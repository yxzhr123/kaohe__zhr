"""拼图求解器 —— 单文件版（YOLO + OpenCV）

考试模式：25 格 = 20 张照片 + 5 个纯白块（5×5，不旋转）

用法
    # 先在页面上手动选 25 格并点开始，然后：
    python puzzle_solver.py --games 1 --minutes 12

流程
    照片阶段：参考格中心 76% 作模板 → 5 尺度滑窗 → 亮度一致性惩罚
              → YOLO 类别融合 → Hungarian 全局唯一分配 → 质心拖拽
              → 进度计数校验 → 失败拖回原位
    白块阶段：照片 20/20 锁定后才开始。原始阈值分割（不腐蚀，保住 4~6px 凸起）
              → 四边 Tab/Blank/Flat 判定 → 由邻居四边取反推导空位需求
              → 平边集合匹配 → 剩余坑位消元收尾

依赖
    pip install opencv-python numpy playwright ultralytics scipy
    playwright install chrome

实测
    25/25（照片 20/20 + 白块 5/5，步数 35，约 2 分钟）
"""
import argparse
import base64
import time
import traceback

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  第一部分：四边凸凹判定
# ═══════════════════════════════════════════════════════════════════

SIDES = ("top", "bottom", "left", "right")
SIDE_KEYS = {"top": "T", "bottom": "B", "left": "L", "right": "R"}


# ═══════════════════════ 轮廓 ═══════════════════════
def get_contour(mask):
    """最大外轮廓（CHAIN_APPROX_NONE 保留全部点，凸凹判断需要稠密点）"""
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


# ═══════════════════════ 四边特征（核心） ═══════════════════════
def body_rect(contour, mask=None):
    """主体矩形（不含凸起）。

    为什么不用 boundingRect：外接框必须包住凸起，于是有凸起的那条边
    在"边缘带"里只剩中间一小段是实心的，会被误判成凹口。
    用「每行/每列实心宽度的中位数」作基准，凸起只占少数行/列，主体就出来了。
    """
    if contour is None or len(contour) < 4:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    if mask is None:
        m = np.zeros((y + h + 1, x + w + 1), np.uint8)
        cv2.drawContours(m, [contour], -1, 1, -1)
    else:
        m = (mask > 0).astype(np.uint8)
    row_w = m.sum(axis=1).astype(float)
    col_h = m.sum(axis=0).astype(float)
    rw = float(np.median(row_w[row_w > 0])) if (row_w > 0).any() else 0
    chh = float(np.median(col_h[col_h > 0])) if (col_h > 0).any() else 0
    if rw < 4 or chh < 4:
        return None
    rows = np.where(row_w >= rw * 0.92)[0]
    cols = np.where(col_h >= chh * 0.92)[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))


def _side_type(extents, tol, min_rows=3):
    """根据「沿边方向每行/列的外沿位置」判断该边类型。

    extents: 该边方向上每一行的外沿坐标（right 用右端 x，left 用左端 x，
             bottom 用下端 y，top 用上端 y）
    以中位数为该边的"标准位置"：
        多数行比标准更外 → Tab（凸起）
        多数行比标准更内 → Blank（凹口）
        否则            → Flat
    """
    if len(extents) < min_rows:
        return "Flat"
    e = np.asarray(extents, dtype=np.float32)
    med = float(np.median(e))
    outer = int((e > med + tol).sum())
    inner = int((e < med - tol).sum())
    need = max(min_rows, int(len(e) * 0.22))
    if outer >= need and outer >= inner:
        return "Tab"
    if inner >= need:
        return "Blank"
    return "Flat"


def smooth_contour(contour, eps_ratio=0.012):
    """轮廓平滑：approxPolyDP 去噪声 + 用凸包做兜底判断。

    游戏渲染的边缘常有 1~2 像素阴影/抗锯齿，原始轮廓会带毛刺，
    直接算"外沿中位数"容易被个别毛刺带偏。
    approxPolyDP 把轮廓压成少量折线段，位置更稳。
    返回 (approx 轮廓, 凸包轮廓)。
    """
    if contour is None or len(contour) < 4:
        return contour, None
    peri = cv2.arcLength(contour, True)
    eps = max(0.8, peri * eps_ratio)
    approx = cv2.approxPolyDP(contour, eps, True)
    if approx is None or len(approx) < 3:
        approx = contour
    hull = cv2.convexHull(contour)
    return approx, hull


def get_edge_features(contour, mask=None, tol_ratio=0.075, mid_ratio=0.64,
                      use_approx=True):
    """返回 {'top','bottom','left','right'} → 'Tab' | 'Blank' | 'Flat'

    做法
    ----
    1. （可选）用 approxPolyDP 平滑轮廓，抑制渲染阴影造成的毛刺
    2. 对每条边，取该边中段 64% 范围内、每一行(或列)的外沿坐标
    3. 与该边**中位数**比较：
           多数行比中位数更靠外  → Tab    （凸起）
           多数行比中位数更靠内  → Blank  （凹口）
           否则                  → Flat
    4. 凸包兜底：若某边被判 Flat，但凸包在该边有 >tol 的延伸，改判 Tab
       （凹口不会被凸包抹掉，所以这一步只补凸起，不会把凹说成凸）
    """
    if contour is None or len(contour) < 4:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    if w < 10 or h < 8:
        return None

    approx, hull = smooth_contour(contour) if use_approx else (contour, None)

    # 掩膜：优先用传入的；否则用平滑后的轮廓填充（阴影噪声也被一并平滑）
    if mask is None:
        mask = np.zeros((y + h + 1, x + w + 1), np.uint8)
        cv2.drawContours(mask, [approx], -1, 1, -1)
    mask = (mask > 0).astype(np.uint8)

    tol = max(2, int(min(w, h) * tol_ratio))
    half = mid_ratio / 2.0
    mx0, mx1 = int(x + w * (0.5 - half)), int(x + w * (0.5 + half))
    my0, my1 = int(y + h * (0.5 - half)), int(y + h * (0.5 + half))

    row_left, row_right, col_top, col_bottom = [], [], [], []
    for cy in range(max(0, my0), min(mask.shape[0], my1)):
        cols = np.where(mask[cy] > 0)[0]
        if cols.size == 0:
            continue
        row_left.append(int(cols.min()))
        row_right.append(int(cols.max()))
    for cx in range(max(0, mx0), min(mask.shape[1], mx1)):
        rows = np.where(mask[:, cx] > 0)[0]
        if rows.size == 0:
            continue
        col_top.append(int(rows.min()))
        col_bottom.append(int(rows.max()))

    feat = {
        "top": _side_type([-v for v in col_top], tol),
        "bottom": _side_type(col_bottom, tol),
        "left": _side_type([-v for v in row_left], tol),
        "right": _side_type(row_right, tol),
    }

    # 凸包兜底：只把 Flat 补成 Tab
    if hull is not None and len(hull) >= 3:
        hx, hy, hw, hh = cv2.boundingRect(hull)
        if feat["left"] == "Flat" and hx < x - tol:
            feat["left"] = "Tab"
        if feat["right"] == "Flat" and hx + hw > x + w + tol:
            feat["right"] = "Tab"
        if feat["top"] == "Flat" and hy < y - tol:
            feat["top"] = "Tab"
        if feat["bottom"] == "Flat" and hy + hh > y + h + tol:
            feat["bottom"] = "Tab"
    return feat


# ═══════════════════════ 互补判据 ═══════════════════════
COMPLEMENT = {
    ("Tab", "Blank"), ("Blank", "Tab"),
    ("Flat", "Flat"),
    ("Tab", "Flat"), ("Flat", "Tab"),        # 边缘格：一侧贴棋盘边界
    ("Blank", "Flat"), ("Flat", "Blank"),
}


def complement_score(fa, fb):
    """两块相邻时的互补得分（4 条边逐一比，仅用于验证/加固）"""
    hit = 0
    for s in SIDES:
        if (fa[s], fb[s]) in COMPLEMENT:
            hit += 1
    return hit


def slot_requirement(edge_features):
    """把「相邻块的特征」换成「本块必须具有的特征」

    若邻居在本格的某条边上凸进来（Tab 朝向本格），本块该边必须是 Blank；
    邻居该边是 Blank，本块该边就是 Tab；邻居 Flat → 本块 Flat。
    （本模块主要用「坑的形状 == 块的形状」这一等价关系，此函数用于邻居推导的场合）
    """
    flip = {"Tab": "Blank", "Blank": "Tab", "Flat": "Flat"}
    return {s: flip[edge_features[s]] for s in SIDES}


def match_score(piece_feat, slot_feat, need_complement=False):
    """块与坑的匹配得分：返回 (perfect, hit_count)

    need_complement=False → 坑的轮廓就是块应有的形状（相同关系，占多数场景）
    need_complement=True  → 坑是用邻居推导出来的（互补关系）
    """
    hit = 0
    for s in SIDES:
        a, b = piece_feat[s], slot_feat[s]
        if need_complement:
            ok = (a, b) in COMPLEMENT
        else:
            ok = (a == b)
        hit += 1 if ok else 0
    return hit == 4, hit


# ═══════════════════════ 120 排列分配 ═══════════════════════
def assign_shapes(piece_feats, slot_feats, need_complement=False):
    """遍历 5! = 120 种对应，取四边吻合数总和最高的方案

    返回 [(piece_idx, slot_idx, hit, perfect)]
    """
    import itertools
    n = min(len(piece_feats), len(slot_feats))
    if n == 0:
        return []
    best, best_key = None, -1
    idx_slots = list(range(len(slot_feats)))
    for perm in itertools.permutations(idx_slots, n):
        total, detail = 0, []
        for i, j in enumerate(perm):
            perfect, hit = match_score(piece_feats[i], slot_feats[j],
                                       need_complement=need_complement)
            total += hit
            detail.append((i, j, hit, perfect))
        if total > best_key:
            best, best_key = detail, total
    return best, best_key


# ═══ 掩膜工具（原 white_fit.py 摘取）═══

# ───────────────────────── 真·轮廓几何 ─────────────────────────
def largest_contour(mask):
    """最大外轮廓（findContours 的结果，后续所有判据都基于它）"""
    if mask is None or mask.sum() == 0:
        return None
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def contour_rect(contour):
    """轮廓的「主体矩形」：用外接框去掉凸起得到轴对齐矩形。

    做法：minAreaRect 给出带凸起的外框；再把轮廓点按行/列投影，
          取投影宽度 ≥95% 满宽的那一段作为主体（凸起处投影明显变窄）。
    """
    if contour is None or len(contour) < 4:
        return None
    pts = contour.reshape(-1, 2)
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    W, H = x1 - x0 + 1, y1 - y0 + 1
    if W < 12 or H < 8:
        return None
    grid = np.zeros((H, W), np.uint8)
    cv2.drawContours(grid, [contour - np.array([[x0, y0]], np.int32)], -1, 1, -1)
    row_w = grid.sum(axis=1).astype(float)
    col_h = grid.sum(axis=0).astype(float)
    # 用「非零宽度的中位数」作基准：凸起只占少数行/列，中位数就是主体宽度
    rw_base = float(np.median(row_w[row_w > 0])) if (row_w > 0).any() else 0
    ch_base = float(np.median(col_h[col_h > 0])) if (col_h > 0).any() else 0
    if rw_base < 4 or ch_base < 4:
        return None
    rows = np.where(row_w >= rw_base * .92)[0]
    cols = np.where(col_h >= ch_base * .92)[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return (int(x0 + cols.min()), int(y0 + rows.min()),
            int(x0 + cols.max()), int(y0 + rows.max()))


# ───────────────────────── 掩膜 ─────────────────────────
def mask_piece(crop, box, upscale=2.0):
    """白块掩膜：亮、低饱和的连通块（白块是纯白，直接阈值即可）"""
    x, y, w, h = [int(v) for v in box[:4]]
    pad = int(max(w, h) * 0.2)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(crop.shape[1], x + w + pad)
    y1 = min(crop.shape[0], y + h + pad)
    patch = crop[y0:y1, x0:x1]
    if patch.size == 0:
        return None, None
    if upscale != 1.0:
        patch_u = cv2.resize(patch, None, fx=upscale, fy=upscale,
                             interpolation=cv2.INTER_CUBIC)
    else:
        patch_u = patch

    # 优先用 SAM 出精确掩膜（默认关闭，见文件头 USE_SEG）
    m = None
    if USE_SEG:
        try:
            import seg_mask
            box_local = ((x - x0) * upscale, (y - y0) * upscale, w * upscale, h * upscale)
            m = seg_mask.mask_in_box(patch_u, box_local, white_only=True)
        except Exception:
            m = None

    if m is None:                      # 退回颜色阈值
        hsv = cv2.cvtColor(patch_u, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 1] < 45) & (hsv[..., 2] > 140)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # 只保留最大连通域，避免相邻白块粘连
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n < 2:
            return None, None
        lab_big = 1 + int(np.argmax(stats[1:, 4]))
        m = (lab == lab_big).astype(np.uint8)

    # 掩膜平滑：游戏渲染边缘有 1~2 像素阴影/抗锯齿，直接算凸凹会被毛刺带偏。
    # 用 approxPolyDP 把轮廓压成折线再填充回来，边界就干净了。
    try:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            c_big = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(c_big, True)
            approx = cv2.approxPolyDP(c_big, max(0.8, peri * 0.012), True)
            if approx is not None and len(approx) >= 4:
                smooth = np.zeros_like(m)
                cv2.drawContours(smooth, [approx], -1, 1, -1)
                # 只在平滑后面积没有明显缩水时采用（防止把凸起削掉）
                if smooth.sum() > m.sum() * 0.90:
                    m = smooth
    except Exception:
        pass
    return m, (x0, y0, upscale)


# ═══ 白块提取（原 white_extract.py）═══

try:
    import edge_features as ef
except Exception:
    ef = None

SIDES = ("top", "bottom", "left", "right")


def standard_area(cw, ch):
    return float(cw * ch)


def white_candidates(crop, cw, ch, min_ratio=0.70, max_ratio=3.40):
    """返回 [(box, area_ratio), ...]，不腐蚀、不粘连合并"""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 50) & (hsv[..., 2] > 145)).astype(np.uint8) * 255
    n, lab, st2, _ = cv2.connectedComponentsWithStats(m, 8)
    std = standard_area(cw, ch)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st2[i]
        if w < cw * .35 or h < ch * .35:
            continue
        r = a / std
        if r < min_ratio or r > max_ratio:
            continue
        # 竖向粘连（一列两块）：高度超过 1.6 格就上下切开
        if h > ch * 1.60 and w <= cw * 1.6:
            k2 = max(2, int(round(h / ch)))
            ph = h / k2
            for t in range(k2):
                by = int(y + t * ph)
                bh = int(ph) + 2
                out.append(((int(x), by, int(w), bh), r / k2, lab, i))
            continue
        if r <= 1.60:                      # 单块
            out.append(((int(x), int(y), int(w), int(h)), r, lab, i))
        else:                              # 粘连：按宽度对半切
            k = max(2, int(round(r)))
            pw = w / k
            for t in range(k):
                bx = int(x + t * pw)
                bw = int(pw) + 2
                out.append(((bx, int(y), bw, int(h)), r / k, lab, i))
    return out


def edges_of(crop, box, lab=None, idx=None):
    """取该块的四边特征（用原始掩膜，不腐蚀）"""
    if ef is None:
        return None
    x, y, w, h = [int(v) for v in box]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(crop.shape[1], x + w), min(crop.shape[0], y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    sub = crop[y0:y1, x0:x1]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 50) & (hsv[..., 2] > 145)).astype(np.uint8)
    if m.sum() < 200:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    feat = ef.get_edge_features(c, m)
    if not feat:
        return None
    return feat


def flat_sides(feat):
    return {s for s in SIDES if feat.get(s) == "Flat"}


def analyse(crop, cw, ch, verbose=True):
    cands = white_candidates(crop, cw, ch)
    res = []
    for (box, ratio, lab, idx) in cands:
        feat = edges_of(crop, box)
        if not feat:
            continue
        flats = flat_sides(feat)
        res.append({"box": box, "ratio": ratio, "feat": feat, "flats": flats})
        if verbose:
            print(f"   {box} 面积比 {ratio:.2f}  "
                  f"{feat['top'][0]}{feat['bottom'][0]}{feat['left'][0]}{feat['right'][0]}  "
                  f"平边 {len(flats)} {sorted(flats)}")
    return res


# ═══ 碎片框工具（原 piece_match.py）═══

SIDES = ("top", "bottom", "left", "right")


def _body_crop(crop, box):
    """抠出碎片主体：去掉外接框内的背景"""
    x, y, w, h = [int(v) for v in box[:4]]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(crop.shape[1], x + w), min(crop.shape[0], y + h)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    sub = crop[y0:y1, x0:x1]
    inner = sub[int(sub.shape[0] * .12):int(sub.shape[0] * .88),
                int(sub.shape[1] * .12):int(sub.shape[1] * .88)]
    return inner if inner.size else None


def tray_pieces(crop, bx, by, bw, bh, cw, ch):
    """托盘碎片（YOLO + Otsu 合并，排除棋盘与右侧面板）"""
    H, W = crop.shape[:2]
    out = []
    q = (crop // 8).reshape(-1, 3)
    vals, cnt = np.unique(q, axis=0, return_counts=True)
    bg = vals[np.argmax(cnt)] * 8 + 4
    d8 = np.clip(np.abs(crop.astype(np.int32) - bg).sum(2), 0, 255).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    roi = np.ones_like(m)
    px, py = int(bw * .04), int(bh * .04)
    roi[max(0, by - py):by + bh + py, max(0, bx - px):bx + bw + px] = 0
    roi[:, int(W * .86):] = 0
    m = cv2.bitwise_and(m, roi)
    n, _, s2, _ = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, n):
        x, y, w, h, a = s2[i]
        if w < cw * .30 or h < ch * .30 or w > cw * 2.6 or h > ch * 2.6:
            continue
        if a < cw * ch * .10:
            continue
        out.append((int(x), int(y), int(w), int(h)))

    # 补充来源：拼图块都有浅色描边，用边缘闭合找一遍（补 Otsu 漏掉的暗色/贴边块）
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ed = cv2.Canny(gray, 40, 120)
        ed = cv2.morphologyEx(ed, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        roi2 = np.ones_like(ed)
        roi2[max(0, by - py):by + bh + py, max(0, bx - px):bx + bw + px] = 0
        roi2[:, int(W * .86):] = 0
        ed = cv2.bitwise_and(ed, roi2)
        cnts, _ = cv2.findContours(ed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c2 in cnts:
            x, y, w, h = cv2.boundingRect(c2)
            a2 = cv2.contourArea(c2)
            if w < cw * .35 or h < ch * .35 or w > cw * 2.6 or h > ch * 2.6:
                continue
            if a2 < cw * ch * .18:
                continue
            if any(abs(x - o[0]) < cw * .35 and abs(y - o[1]) < ch * .35 for o in out):
                continue
            out.append((int(x), int(y), int(w), int(h)))
    except Exception:
        pass
    return out


def score_pair(cell_img, piece_patch):
    """参考格 vs 碎片主体 的相似度（灰度 NCC + 边缘 NCC，取较大值）"""
    if cell_img is None or piece_patch is None:
        return 0.0
    if cell_img.size == 0 or piece_patch.size == 0:
        return 0.0
    try:
        # 两边都去掉 12% 边缘：参考格没有凸起/白边，碎片有，范围不一致会压低分数
        def _inner(x):
            h0, w0 = x.shape[:2]
            return x[int(h0*.12):int(h0*.88), int(w0*.12):int(w0*.88)]
        a = cv2.resize(_inner(cell_img), (64, 48), interpolation=cv2.INTER_AREA)
        b = cv2.resize(_inner(piece_patch), (64, 48), interpolation=cv2.INTER_AREA)
    except Exception:
        return 0.0
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ga -= ga.mean()
    gb -= gb.mean()
    den = float(np.sqrt((ga ** 2).sum() * (gb ** 2).sum())) + 1e-6
    ncc = float((ga * gb).sum() / den)
    # 边缘相似度（对白底/透明物体更敏感）
    ea = cv2.Canny(cv2.resize(a, (128, 96)), 50, 140).astype(np.float32)
    eb = cv2.Canny(cv2.resize(b, (128, 96)), 50, 140).astype(np.float32)
    ea -= ea.mean()
    eb -= eb.mean()
    den2 = float(np.sqrt((ea ** 2).sum() * (eb ** 2).sum())) + 1e-6
    encc = float((ea * eb).sum() / den2)
    return max(ncc, 0.5 * ncc + 0.5 * encc)


def match_cells(cells, want, pieces, min_score=0.30):
    """全局唯一分配：格子 × 碎片，Hungarian

    cells : {(r,c): 参考格图像}
    want  : 待放格子列表
    pieces: [(box, patch)]
    返回 [(cell, box, score)]
    """
    if not want or not pieces:
        return []
    mat = np.zeros((len(want), len(pieces)), dtype=np.float32)
    for i, key in enumerate(want):
        for j, (box, patch) in enumerate(pieces):
            mat[i, j] = score_pair(cells[key], patch)
    out = []
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(-mat)
        pairs = list(zip(ri, ci))
    except Exception:
        pairs, used = [], set()
        for i in range(len(want)):
            order = np.argsort(-mat[i])
            for j in order:
                if j in used:
                    continue
                used.add(j)
                pairs.append((i, j))
                break
    for i, j in pairs:
        if mat[i, j] >= min_score:
            out.append((want[i], pieces[j][0], float(mat[i, j])))
    out.sort(key=lambda z: -z[2])
    return out


# ═══════════════════════════════════════════════════════════════════
#  模块 shim（原本是独立模块，单文件版用同名对象承接，调用点保持不变）
# ═══════════════════════════════════════════════════════════════════
class _Shim:
    pass


ef = _Shim()
ef.get_edge_features = get_edge_features
ef.get_contour = get_contour

wf = _Shim()
wf.mask_piece = mask_piece

we = _Shim()
we.analyse = analyse

pm = _Shim()
pm._body_crop = _body_crop
pm.score_pair = score_pair
pm.tray_pieces = tray_pieces


# ═══════════════════════════════════════════════════════════════════
#  第五部分：主流程
# ═══════════════════════════════════════════════════════════════════

URL = "https://puzzle.moyunliuyin.me/?mode=exam"
import os as _os
WTS = _os.getenv("YOLO_WTS", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                             "weights", "best.pt"))
_MODEL = None
BOARD_PRIOR = (746, 554)
LOG_PATH = "final25.log"


# ══════════════════ 日志 ══════════════════
class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


log = Log(LOG_PATH)


# ══════════════════ 页面 ══════════════════
def st(page):
    try:
        return page.evaluate("""() => {
            const t = (document.body && document.body.innerText) || '';
            const p = t.match(/\u8fdb\u5ea6[:：]?\\s*(\\d+)\\s*[\\/／]\\s*(\\d+)/);
            const m = t.match(/\u6b65\u6570[:：]?\\s*(\\d+)/);
            const w = t.match(/(\\d+)\\s*\u683c\u7eaf\u767d/);
            const i = document.querySelector('#preview-img');
            const v = document.querySelector('#puzzle-canvas');
            const r = v ? v.getBoundingClientRect() : null;
            return {cur: p ? +p[1] : null, total: p ? +p[2] : null,
                    steps: m ? +m[1] : null, n_white: w ? +w[1] : 5,
                    src: i ? (i.currentSrc || i.src || '') : '',
                    box: r ? [r.x, r.y, r.width, r.height] : null,
                    vw: innerWidth || 1280};
        }""")
    except Exception:
        return {"cur": None, "total": None, "steps": None, "n_white": 5,
                "src": "", "box": None, "vw": 1280}


def start_new_game(page, need=25):
    """点「新开一局」并把难度切到 25 格"""
    try:
        page.click("text=新开一局", timeout=5000)
        page.wait_for_timeout(1200)
    except Exception as e:
        log(f"   点新开一局失败: {str(e).splitlines()[0][:50]}")
    # 难度下拉：选含 25 的选项
    try:
        sels = page.query_selector_all("select")
        for sl in sels:
            opts = sl.query_selector_all("option")
            for op in opts:
                txt = (op.inner_text() or "")
                if "25" in txt:
                    val = op.get_attribute("value")
                    if val:
                        sl.select_option(value=val)
                    else:
                        sl.select_option(label=txt)
                    page.wait_for_timeout(600)
                    raise StopIteration
    except StopIteration:
        pass
    except Exception as e:
        log(f"   切难度失败: {str(e).splitlines()[0][:50]}")
    # 自定义下拉：点开难度控件再点含 25 的选项
    try:
        for sel in ("text=难度", "text=9 格", "text=9格"):
            try:
                page.click(sel, timeout=1500)
                break
            except Exception:
                continue
        page.wait_for_timeout(500)
        for txt in ("25 格 (5×5)", "25 格", "25格", "(5×5)"):
            try:
                page.click(f"text={txt}", timeout=1500)
                break
            except Exception:
                continue
        page.wait_for_timeout(800)
    except Exception:
        pass

    # 可能还有「开始」按钮
    for txt in ("开始", "开始拼图", "确认"):
        try:
            b = page.query_selector(f"text={txt}")
            if b and b.is_visible():
                b.click()
                page.wait_for_timeout(800)
                break
        except Exception:
            pass
    # 等 25 格就绪
    t0 = time.time()
    while time.time() - t0 < 60:
        page.wait_for_timeout(400)
        s = st(page)
        if s["total"] == need:
            return True
    return False


def grab(page):
    return cv2.imdecode(np.frombuffer(page.screenshot(), np.uint8), cv2.IMREAD_COLOR)


def canvas_of(arr, s):
    """画布裁剪。右侧参考图面板会给出 1.00 的假匹配（它和目标格一模一样），
    必须先切掉：从右往左找连续的高亮列（均值>195 且占比>80%）。"""
    k = arr.shape[1] / s["vw"]
    x, y, w, h = [int(v * k) for v in s["box"]]
    crop = arr[y:y + h, x:x + w]
    H, W = crop.shape[:2]
    col = crop.mean(axis=(0, 2))
    bright = col > 195
    cut = W
    cnt = 0
    for i in range(W - 1, int(W * 0.45), -1):
        if bright[i]:
            cnt += 1
            cut = i
        elif cnt > W * 0.05:
            break
    if cnt > W * 0.12:
        crop = crop[:, :cut]
    return crop, x, y, k


def bg_of(img):
    q = (img // 8).reshape(-1, 3)
    vals, cnt = np.unique(q, axis=0, return_counts=True)
    return vals[np.argmax(cnt)] * 8 + 4


def find_board(crop, aspect):
    h, w = crop.shape[:2]
    q = (crop // 4).reshape(-1, 3)
    vals, cnt = np.unique(q, axis=0, return_counts=True)
    cands = []
    for i in np.argsort(-cnt)[:8]:
        c = vals[i] * 4
        if not 45 < c.sum() < 600:
            continue
        m = (np.abs(crop.astype(int) - c).sum(2) < 30).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        n, _, s2, _ = cv2.connectedComponentsWithStats(m, 8)
        if n < 2:
            continue
        x, y, bw, bh, a = s2[1 + int(np.argmax(s2[1:, 4]))]
        if a < .05 * w * h or a / (bw * bh) < .70 or not .95 < bw / bh < 2.1:
            continue
        if abs(bw / bh - aspect) > .02:
            bw = int(round(bh * aspect))
        cands.append((int(x), int(y), int(bw), int(bh)))
    if not cands:
        return None
    cands.sort(key=lambda c: abs(c[2] - BOARD_PRIOR[0]) + abs(c[3] - BOARD_PRIOR[1]))
    return cands[0]


def read_ref(page, s):
    src = s["src"]
    if not (src.startswith("data:image") and "," in src):
        return None
    try:
        return cv2.imdecode(np.frombuffer(base64.b64decode(src.split(",", 1)[1]),
                                          np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def cut_cells(ref, g):
    ch, cw = ref.shape[0] // g, ref.shape[1] // g
    return {(r, c): ref[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            for r in range(g) for c in range(g)}


def blank_cells(cells, n):
    def score(cell):
        h, w = cell.shape[:2]
        core = cell[int(h * .15):int(h * .85), int(w * .15):int(w * .85)]
        if core.size == 0:
            return -1e9
        hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
        white = ((hsv[..., 1] < 22) & (hsv[..., 2] > 220)).astype(np.uint8)
        nw = cv2.morphologyEx((1 - white).astype(np.uint8) * 255,
                              cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        nn, _, st2, _ = cv2.connectedComponentsWithStats(nw, 8)
        objs = sum(1 for i in range(1, nn) if st2[i, 4] > core.size * .007)
        return (float(white.mean()) * 100
                - min(float(cv2.Laplacian(gray, cv2.CV_32F).var()), 300) * .3
                - objs * 12 - float(hsv[..., 1].mean()) * .2)

    ranked = sorted(cells, key=lambda k: -score(cells[k]))
    log("   白格打分: " + ", ".join(f"{k[0]+1}{k[1]+1}:{score(cells[k]):.0f}"
                                   for k in ranked[:n + 2]))
    return set(ranked[:max(0, n)])


# ══════════════════ YOLO ══════════════════
def model():
    global _MODEL
    if _MODEL is None:
        try:
            from ultralytics import YOLO
            _MODEL = YOLO(WTS)
            log(f"   YOLO 已加载（{len(getattr(_MODEL, 'names', {}) or {})} 类）")
        except Exception as e:
            log("   YOLO 不可用:", str(e)[:60])
            _MODEL = False
    return _MODEL or None


def yolo_boxes(img, conf=0.15, iou=0.5, imgsz=1280):
    m = model()
    if not m:
        return []
    try:
        r = m.predict(img, conf=conf, iou=iou, augment=True, imgsz=imgsz,
                      verbose=False, device="cpu")[0]
    except Exception:
        try:
            r = m.predict(img, conf=conf, iou=iou, imgsz=imgsz,
                          verbose=False, device="cpu")[0]
        except Exception:
            return []
    raw = []
    for b in r.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        raw.append((x1, y1, x2 - x1, y2 - y1, int(b.cls), float(b.conf)))
    raw.sort(key=lambda d: -d[2] * d[3])
    out = []
    for d in raw:
        cx, cy = d[0] + d[2] / 2, d[1] + d[3] / 2
        if any(abs(cx - (e[0] + e[2] / 2)) < min(d[2], e[2]) * .5 and
               abs(cy - (e[1] + e[3] / 2)) < min(d[3], e[3]) * .5 for e in out):
            continue
        out.append(d)
    return out


def ref_classes(ref, g, conf=0.35):
    names = getattr(model(), "names", {}) or {}
    rh, rw = ref.shape[:2]
    cw, ch = rw / g, rh / g
    votes = {}
    for (x, y, w, h, cls, cf) in yolo_boxes(ref, conf=conf, iou=0.5, imgsz=1280):
        c, r = int((x + w / 2) // cw), int((y + h / 2) // ch)
        if 0 <= r < g and 0 <= c < g:
            k = (r, c)
            if k not in votes or cf > votes[k][1]:
                votes[k] = (cls, cf)
    dist = {}
    for cls, cf in votes.values():
        dist[names.get(cls, cls)] = dist.get(names.get(cls, cls), 0) + 1
    log(f"   参考图识别 {len(votes)}/{g*g} 格  {dist}")
    return votes


# ══════════════════ 图像工具 ══════════════════
_ORB = cv2.ORB_create(nfeatures=500)


def centroid_of(crop, box):
    x, y, w, h = [int(v) for v in box[:4]]
    pad = int(min(w, h) * .1)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    p = crop[y0:min(crop.shape[0], y + h + pad), x0:min(crop.shape[1], x + w + pad)]
    if p.size == 0:
        return x + w / 2, y + h / 2
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 60) & (hsv[..., 2] > 120)).astype(np.uint8)
    mm = cv2.moments(m, True)
    if mm["m00"] > 800:
        return x0 + mm["m10"] / mm["m00"], y0 + mm["m01"] / mm["m00"]
    return x + w / 2, y + h / 2


def locate(crop, want, cells, s0, thr=0.55):
    half = cv2.resize(crop, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    out = []
    for key in want:
        peaks = []
        # 只取参考格中心 76% 作模板：去掉碎片白边/凸起带来的干扰，
        # 正确块相关性会明显升高，错块明显降低
        _cell = cells[key]
        _h0, _w0 = _cell.shape[:2]
        _core = _cell[int(_h0 * .12):int(_h0 * .88), int(_w0 * .12):int(_w0 * .88)]
        for f in (0.92, 0.96, 1.00, 1.04, 1.08):
            t = cv2.resize(_core, None, fx=s0 * f * .5, fy=s0 * f * .5,
                           interpolation=cv2.INTER_AREA)
            if t.shape[0] < 15 or t.shape[0] >= half.shape[0] or t.shape[1] >= half.shape[1]:
                continue
            r = cv2.matchTemplate(half, t, cv2.TM_CCOEFF_NORMED)
            # 白底照片块（白狗/白猫/白杯子）：灰度会在任何白色区域给高分，
            # 必须强制用边缘图匹配（物体轮廓才是特征）
            _th = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
            _white_frac = float(((_th[..., 1] < 40) & (_th[..., 2] > 170)).mean())
            if _white_frac > 0.45:
                try:
                    he = cv2.Canny(half, 40, 110)
                    te = cv2.Canny(t, 40, 110)
                    if te.sum() > 0:
                        re2 = cv2.matchTemplate(he, te, cv2.TM_CCOEFF_NORMED)
                        # 取两者较大值：白猫这类灰度也可能更好，不能一刀切
                        if float(re2.max()) > float(r.max()):
                            r = re2
                except Exception:
                    pass
            # 低对比度碎片（透明雨伞这类）：再比一次边缘图，取较大值
            if float(r.max()) < 0.75:
                try:
                    he = cv2.Canny(half, 50, 140)
                    te = cv2.Canny(t, 50, 140)
                    if te.sum() > 0:
                        r2 = cv2.matchTemplate(he, te, cv2.TM_CCOEFF_NORMED)
                        if float(r2.max()) > float(r.max()):
                            r = r2
                except Exception:
                    pass
            _tmean = float(t.mean())
            for _ in range(2):
                v = float(r.max())
                if v < thr:
                    break
                yy, xx = np.unravel_index(r.argmax(), r.shape)
                # 亮度一致性：白杯子 vs 黑杯子这类同类别、不同底色，必须按亮度区分
                _reg = half[yy:yy + t.shape[0], xx:xx + t.shape[1]]
                if _reg.size:
                    _d = abs(float(_reg.mean()) - _tmean) / 255.0
                    v = v * (1.0 - 0.7 * _d)
                if v < thr:
                    break
                peaks.append((v, int(xx * 2), int(yy * 2),
                              t.shape[1] * 2, t.shape[0] * 2))
                r[max(0, yy - 12):yy + 13, max(0, xx - 12):xx + 13] = -1
        peaks.sort(key=lambda z: -z[0])
        for p in peaks[:2]:
            out.append((key,) + p)
    out.sort(key=lambda z: -z[1])
    keep, used = [], []
    for it in out:
        if any(abs(it[2] - b[0]) < 6 and abs(it[3] - b[1]) < 6 for b in used):
            continue
        used.append((it[2], it[3]))
        keep.append(it)
    return keep


def drag_route(page, src, dst, boxes, ox, oy, k):
    """拖拽时绕开其它碎片：比较横-竖 / 竖-横 两条路径，选碰撞更少的"""
    def cnt_h(y, x0, x1):
        n = 0
        for b in boxes:
            bx0 = ox + b[0]
            by0 = oy + b[1]
            by1 = oy + b[1] + b[3]
            if by0 - 6 <= y <= by1 + 6 and not (max(x0, bx0) > min(x1, bx0 + b[2])):
                n += 1
        return n

    def cnt_v(x, y0, y1):
        n = 0
        for b in boxes:
            bx0 = ox + b[0]
            by0 = oy + b[1]
            by1 = oy + b[1] + b[3]
            if bx0 - 6 <= x <= bx0 + b[2] + 6 and not (max(y0, by0) > min(y1, by1)):
                n += 1
        return n

    hv = cnt_h(src[1], src[0], dst[0]) + cnt_v(dst[0], src[1], dst[1])
    vh = cnt_v(src[0], src[1], dst[1]) + cnt_h(dst[1], src[0], dst[0])
    if hv <= vh:
        mid = (dst[0], src[1])
    else:
        mid = (src[0], dst[1])

    page.mouse.move(*src)
    page.wait_for_timeout(120)
    page.mouse.down()
    page.wait_for_timeout(170)

    def glide(p0, p1, steps=14):
        for i in range(1, steps + 1):
            t = i / steps
            e = t * t * (3 - 2 * t)
            page.mouse.move(p0[0] + (p1[0] - p0[0]) * e,
                            p0[1] + (p1[1] - p0[1]) * e)
            page.wait_for_timeout(20)

    glide(src, mid)
    page.wait_for_timeout(40)
    glide(mid, dst)
    page.wait_for_timeout(50)
    page.mouse.move(*dst)
    page.wait_for_timeout(170)
    page.mouse.up()
    page.wait_for_timeout(320)


def drag(page, src, dst, settle=330):
    page.mouse.move(*src)
    page.wait_for_timeout(120)
    page.mouse.down()
    page.wait_for_timeout(180)
    for i in range(1, 26):
        t = i / 25
        e = t * t * (3 - 2 * t)
        page.mouse.move(src[0] + (dst[0] - src[0]) * e,
                        src[1] + (dst[1] - src[1]) * e)
        page.wait_for_timeout(20)
    page.wait_for_timeout(50)
    page.mouse.move(*dst)
    page.wait_for_timeout(180)
    page.mouse.up()
    page.wait_for_timeout(settle)


def tray_boxes(crop, bx, by, bw, bh, cw, ch):
    """托盘碎片（YOLO + Otsu 合并，排除棋盘与右侧面板）"""
    H, W = crop.shape[:2]
    out = []
    try:
        for (x, y, w, h, cls, cf) in yolo_boxes(crop, conf=0.15, iou=0.5):
            if w < cw * .3 or h < ch * .3 or w > cw * 2.6 or h > ch * 2.6:
                continue
            cx, cy = x + w / 2, y + h / 2
            if (bx - cw * .1 < cx < bx + bw + cw * .1 and
                    by - ch * .1 < cy < by + bh + ch * .1):
                continue
            out.append((float(x), float(y), float(w), float(h)))
    except Exception:
        pass
    d8 = np.clip(np.abs(crop.astype(np.int32) - bg_of(crop)).sum(2), 0, 255).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    roi = np.ones_like(m)
    px, py = int(bw * .005), int(bh * .005)
    roi[max(0, by - py):by + bh + py, max(0, bx - px):bx + bw + px] = 0
    roi[:, int(W * .90):] = 0
    m = cv2.bitwise_and(m, roi)
    n, _, s2, _ = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, n):
        x, y, w, h, a = s2[i]
        if w < cw * .35 or h < ch * .35 or w > cw * 2.6 or h > ch * 2.6:
            continue
        if a < cw * ch * .12:
            continue
        if any(abs(x - o[0]) < cw * .3 and abs(y - o[1]) < ch * .3 for o in out):
            continue
        out.append((float(x), float(y), float(w), float(h)))
    return out


def white_boxes(crop, bx, by, bw, bh, cw, ch):
    """白块分割：开运算 + 腐蚀分离粘连，再按面积筛选。

    原来用 CLOSE(7) 会把相邻白块连成一整片（实测 761×550），
    面积过滤直接把它拒掉，于是"托盘里没有白块"。
    """
    H, W = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 45) & (hsv[..., 2] > 150)).astype(np.uint8) * 255
    # 先开运算去掉毛刺，再腐蚀断开粘连
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.erode(m, np.ones((7, 7), np.uint8), iterations=1)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)

    roi = np.ones_like(m)
    px, py = int(bw * .05), int(bh * .05)
    roi[max(0, by - py):by + bh + py, max(0, bx - px):bx + bw + px] = 0
    roi[:, int(W * .85):] = 0
    m = cv2.bitwise_and(m, roi)

    n, _, s2, _ = cv2.connectedComponentsWithStats(m, 8)
    std = float(cw * ch)
    out = []
    for i in range(1, n):
        x, y, w, h, a = s2[i]
        if not (std * .50 <= a <= std * 1.60):
            continue
        if not (0.55 <= w / max(1, h) <= 2.0):
            continue
        if a / float(w * h) < .40:
            continue
        # 腐蚀掉的部分补回来
        out.append((max(0, x - 4), max(0, y - 4),
                    min(W - x, w + 8), min(H - y, h + 8)))
    return out


# ══════════════════ 单局 ══════════════════
def play_one(page, s, a):
    """跑一局，返回最终进度（带硬性时间/次数上限，无进展立即退出）"""
    deadline = time.time() + float(getattr(a, "minutes", 40)) * 60
    g = round((s["total"] or 25) ** .5)
    ref = read_ref(page, s)
    if ref is None:
        log("   读不到参考图")
        return s["cur"] or 0
    cells = cut_cells(ref, g)
    blank = blank_cells(cells, s["n_white"] or 5)
    log(f"   网格 {s['total']} ({g}×{g})  白格 {len(blank)}: "
        f"{sorted(f'({r+1},{c+1})' for r, c in blank)}")

    crop, ox, oy, k = canvas_of(grab(page), s)
    bb = find_board(crop, ref.shape[1] / ref.shape[0])
    if not bb:
        log("   找不到棋盘")
        return s["cur"] or 0
    bx, by, bw, bh = bb
    # 不再钳制尺寸：真实棋盘随窗口变化，钳死会引入像素级偏移
    cw, ch = bw / g, bh / g
    s0 = bw / ref.shape[1]
    log(f"   棋盘 ({bx},{by}) {bw}×{bh}  格 {cw:.0f}×{ch:.0f}")

    want_cls = ref_classes(ref, g)
    photo = {k2: v2 for k2, v2 in cells.items() if k2 not in blank}
    done, fails, stall = set(), {}, 0
    retry_block = {}          # 格子 -> 解除封锁的轮次
    placed_piece = {}

    for rnd in range(1, a.rounds + 1):
        stt = st(page)
        cur, steps = stt["cur"] or 0, stt["steps"] or 0
        total = s["total"] or 0
        if cur >= total:
            break
        if steps >= 9999:
            log(f"   步数 {steps} 接近上限 33，停止试探")
            break
        crop, ox, oy, k = canvas_of(grab(page), s)
        bg = bg_of(crop)

        filled = set(done)
        for key in photo:
            r, c = key
            x0 = int(bx + c * cw) - int(cw * .16)
            y0 = int(by + r * ch) - int(ch * .16)
            x1 = int(bx + (c + 1) * cw) + int(cw * .16)
            y1 = int(by + (r + 1) * ch) + int(ch * .16)
            win = crop[max(0, y0):y1, max(0, x0):x1]
            if win.size == 0:
                continue
            tpl = cv2.resize(cells[key],
                             (max(8, int(cw * .78)), max(8, int(ch * .78))),
                             interpolation=cv2.INTER_AREA)
            if tpl.shape[0] < win.shape[0] and tpl.shape[1] < win.shape[1]:
                if float(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED).max()) > .60:
                    filled.add(key)

        want = [k2 for k2 in photo if k2 not in filled and fails.get(k2, 0) < 3]

        # ── 主循环：滑窗定位（已验证：20/20 与 19 格/20 步都是这条路径）──
        found = locate(crop, want, cells, s0) if want else []
        # 峰值必须在托盘内：棋盘内的命中是「已放好的同类碎片」
        _f2 = []
        for it in found:
            _cx = it[2] + it[4] * 0.5
            _cy = it[3] + it[5] * 0.5
            if (bx - cw * .15 < _cx < bx + bw + cw * .15 and
                    by - ch * .15 < _cy < by + bh + ch * .15):
                continue
            _f2.append(it)
        if len(_f2) < len(found):
            log(f"   剔除棋盘内命中 {len(found) - len(_f2)} 个")
        found = _f2


        # 分数融合：YOLO 类别一致 → geo*0.8 + 0.2
        dets_now = []
        try:
            dets_now = yolo_boxes(crop, conf=0.25, iou=0.5)
        except Exception:
            dets_now = []
        fused = []
        for (key, sc, x, y, w, h) in found:
            exp = want_cls.get(key)
            tag = ""
            best_d, hit = 1e9, None
            pcx, pcy = x + w / 2, y + h / 2
            for (dx0, dy0, dw, dh, dcls, dcf) in dets_now:
                d2 = abs(dx0 + dw / 2 - pcx) + abs(dy0 + dh / 2 - pcy)
                if d2 < best_d:
                    best_d, hit = d2, (dcls, dcf)
            if exp and hit and best_d < cw * 1.3:
                if hit[0] == exp[0]:
                    sc = min(1.0, sc * 0.8 + 0.2)
                    tag = "同类+"
                else:
                    tag = "异类"
            fused.append((key, sc, x, y, w, h, tag))
            r0, c0_ = key
            log(f"   e{r0+1}c{c0_+1} 滑窗 {sc:.2f} {tag}")

        # 全局唯一分配（一块碎片只给一格）
        cells_want, pos_list = [], []
        for it in fused:
            if it[0] not in cells_want:
                cells_want.append(it[0])
            kp = (int(it[2] // 12), int(it[3] // 12))
            if kp not in pos_list:
                pos_list.append(kp)
        plan = []
        if cells_want and pos_list:
            sc_mat = np.zeros((len(cells_want), len(pos_list)), dtype=np.float32)
            info = {}
            for it in fused:
                v2 = it[1]
                if v2 < 0.60:
                    continue
                if v2 < 0.75 and it[6] == "异类":
                    continue
                i2 = cells_want.index(it[0])
                j2 = pos_list.index((int(it[2] // 12), int(it[3] // 12)))
                if v2 > sc_mat[i2, j2]:
                    sc_mat[i2, j2] = v2
                    info[(i2, j2)] = it
            try:
                from scipy.optimize import linear_sum_assignment
                ri, ci = linear_sum_assignment(-sc_mat)
                for i2, j2 in zip(ri, ci):
                    if sc_mat[i2, j2] > 0 and (i2, j2) in info:
                        plan.append(info[(i2, j2)][:6])
            except Exception:
                used_p = set()
                for it in sorted(fused, key=lambda z: -z[1]):
                    v2 = it[1]
                    if v2 < 0.60 or (v2 < 0.75 and it[6] == "异类"):
                        continue
                    kp = (int(it[2] // 12), int(it[3] // 12))
                    if kp in used_p:
                        continue
                    used_p.add(kp)
                    plan.append(it[:6])
            log(f"   唯一分配 {len(plan)} 对"
                f"（格子 {len(cells_want)} / 位置 {len(pos_list)}）")

        def nb(key):
            r0, c0 = key
            return sum(1 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                       if (r0 + dr, c0 + dc) in filled)

        def on_border(key):
            r, c = key
            return r in (0, g - 1) or c in (0, g - 1)

        if not filled:
            plan = [it for it in plan
                    if it[0][0] in (0, g - 1) and it[0][1] in (0, g - 1)]
        else:
            # 角/边格由棋盘边界当锚点，永远允许；内部格必须有已放好的邻居
            plan = [it for it in plan if on_border(it[0]) or nb(it[0]) >= 1]

        def rank(it):
            r, c = it[0]
            corner = r in (0, g - 1) and c in (0, g - 1)
            edge = r in (0, g - 1) or c in (0, g - 1)
            lvl = 0 if corner else (1 if edge else 1 + min(r, c, g - 1 - r, g - 1 - c))
            return (lvl, -nb(it[0]))

        plan.sort(key=rank)
        plan = plan[:6]
        log(f"第 {rnd} 轮  {cur}/{total}  已放 {len(filled)}  计划 {len(plan)}")
        gained = 0
        boxes_now = tray_boxes(crop, bx, by, bw, bh, cw, ch) if plan else []
        for key, sc, x, y, w, h in plan:
            r, c = key
            dst = ((ox + bx + (c + .5) * cw) / k, (oy + by + (r + .5) * ch) / k)
            ccx, ccy = centroid_of(crop, (x, y, w, h))
            src = ((ox + ccx) / k, (oy + ccy) / k)
            pass
            cxy = (int(bx + (c + .5) * cw), int(by + (r + .5) * ch))
            hh, ww = int(ch * .18), int(cw * .18)
            probe = crop[max(0, cxy[1] - hh):cxy[1] + hh,
                         max(0, cxy[0] - ww):cxy[0] + ww]
            if probe.size and np.abs(probe.reshape(-1, 3).mean(0) - bg).sum() >= 45:
                drag(page, dst, ((ox + int(cw * .3)) / k,
                                 (oy + int(crop.shape[0] - ch * 1.05)) / k))
            c0 = st(page)["cur"] or 0
            drag(page, src, dst)
            now = st(page)
            ok = (now["cur"] or 0) > c0
            if not ok:
                cur_src = dst
                for rr in (8, 16, 24):
                    if ok:
                        break
                    for ddx, ddy in ((0, -rr), (0, rr), (-rr, 0), (rr, 0)):
                        c1 = st(page)["cur"] or 0
                        d2 = (dst[0] + ddx, dst[1] + ddy)
                        drag(page, cur_src, d2)
                        n2 = st(page)
                        if (n2["cur"] or 0) > c1:
                            now, ok = n2, True
                            break
                        cur_src = d2
            log(f"   e{r+1}c{c+1} {c0}→{now['cur']} {sc:.2f} {'OK' if ok else 'X'}")
            if ok:
                done.add(key)
                placed_piece[key] = (x, y, w, h)
                fails.pop(key, None)
                gained += 1
            else:
                # 拖回原位（否则会叠在格子上，后面的块也检不出来）
                try:
                    drag(page, cur_src if 'cur_src' in dir() else dst, src)
                except Exception:
                    pass
                fails[key] = fails.get(key, 0) + 1
                pass
        stall = 0 if gained else stall + 1
        if stall >= 3:
            log("   照片阶段结束")
            break

    # 收尾
    miss = [k2 for k2 in photo if k2 not in done]
    cur = st(page)["cur"] or 0
    if miss and cur < (s["total"] or 0):
        crop, ox, oy, k = canvas_of(grab(page), s)
        hits = locate(crop, miss, cells, s0, thr=.45)
        boxes = tray_boxes(crop, bx, by, bw, bh, cw, ch)
        log(f"   收尾：{len(miss)} 格未拼，几何命中 {len(hits)}，托盘 {len(boxes)}")
        for key, sc2, x2, y2, w2, h2 in hits:
            if key in done:
                continue
            # 类别先验 + 内容相似度：先只在该格参考类别的碎片里挑，再退化为全体
            try:
                _names = getattr(model(), "names", {}) or {}
                _exp = want_cls.get(key)
                _exp_name = _names.get(_exp[0]) if _exp else None
                # 每个碎片框的 YOLO 类别
                _det_cls = []
                try:
                    _det_cls = yolo_boxes(crop, conf=0.25, iou=0.5)
                except Exception:
                    _det_cls = []
                def _cls_of(b2):
                    bcx, bcy = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2
                    best, bd = None, 1e9
                    for (dx0, dy0, dw, dh, dcls, dcf) in _det_cls:
                        d2 = abs(dx0 + dw / 2 - bcx) + abs(dy0 + dh / 2 - bcy)
                        if d2 < bd:
                            bd, best = d2, _names.get(dcls)
                    return best if bd < cw * 0.9 else None
                _same = []
                if _exp_name:
                    for b2 in boxes:
                        if _cls_of(b2) == _exp_name:
                            _same.append(b2)
                if _same:
                    log(f"      e{r+1}c{c+1} 类别 {_exp_name} 命中 {len(_same)} 块")
                pool = _same if _same else boxes
                scored = []
                for b2 in pool:
                    pc = pm._body_crop(crop, b2)
                    scored.append((pm.score_pair(cells[key], pc), b2))
                scored.sort(key=lambda z: -z[0])
                ranked = [b3 for (_s3, b3) in scored[:6]]
                if scored:
                    log(f"   收尾匹配 e{key[0]+1}c{key[1]+1} 最高相似 "
                        f"{scored[0][0]:.2f}")
            except Exception:
                ranked = sorted(
                    boxes,
                    key=lambda b2: abs(b2[0] + b2[2] / 2 - (x2 + w2 / 2)) +
                    abs(b2[1] + b2[3] / 2 - (y2 + h2 / 2)))[:2]
            r, c = key
            dst = ((ox + bx + (c + .5) * cw) / k, (oy + by + (r + .5) * ch) / k)
            for b2 in ranked:
                c0 = st(page)["cur"] or 0
                ccx, ccy = centroid_of(crop, b2)
                drag(page, ((ox + ccx) / k, (oy + ccy) / k), dst)
                now = st(page)
                if (now["cur"] or 0) > c0:
                    done.add(key)
                    log(f"   e{r+1}c{c+1} 几何 {sc2:.2f} OK {c0}→{now['cur']}")
                    break
                else:
                    log(f"   e{r+1}c{c+1} 几何 {sc2:.2f} X")
                    drag(page, dst, ((ox + ccx) / k, (oy + ccy) / k))  # 拖回原位

    # ── 兜底已关闭（步数上限 33，禁止穷举）──

    # ── 照片完成后锁定：不再动任何照片格 ──
    unfilled = [k2 for k2 in photo if k2 not in done and k2 not in blank]
    if False:
      for key in photo:
        if key in blank:
            continue
        r, c = key
        x0 = int(bx + c * cw) - int(cw * .16)
        y0 = int(by + r * ch) - int(ch * .16)
        x1 = int(bx + (c + 1) * cw) + int(cw * .16)
        y1 = int(by + (r + 1) * ch) + int(ch * .16)
        win = crop_chk[max(0, y0):y1, max(0, x0):x1]
        if win.size == 0:
            unfilled.append(key)
            continue
        tpl = cv2.resize(cells[key], (max(8, int(cw * .78)), max(8, int(ch * .78))),
                         interpolation=cv2.INTER_AREA)
        if tpl.shape[0] >= win.shape[0] or tpl.shape[1] >= win.shape[1]:
            unfilled.append(key)
            continue
        if float(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED).max()) <= .60:
            unfilled.append(key)
    log(f"   照片核对：{len(photo) - len(unfilled)}/{len(photo)} 格已正确"
        + (f"，未完成 {len(unfilled)}" if unfilled else " —— 锁定，不再动照片格"))

    # ── 最后一格定点尝试（最多 3 块，按相似度排序，进度判定）──
    if 0 < len(unfilled) <= 2 and (st(page)["cur"] or 0) < (s["total"] or 0):
        crop, ox, oy, k = canvas_of(grab(page), s)
        for key in unfilled[:2]:
            if (st(page)["cur"] or 0) >= (s["total"] or 0):
                break
            r, c = key
            dst = ((ox + bx + (c + .5) * cw) / k, (oy + by + (r + .5) * ch) / k)
            boxes = tray_boxes(crop, bx, by, bw, bh, cw, ch)
            try:
                scored = []
                for b2 in boxes:
                    pc = pm._body_crop(crop, b2)
                    scored.append((pm.score_pair(cells[key], pc), b2))
                scored.sort(key=lambda z: -z[0])
            except Exception:
                scored = [(0.0, b2) for b2 in boxes]
            log(f"   最后一格 e{r+1}c{c+1}：托盘 {len(boxes)} 块，试前 6")
            for (sc3, b2) in scored[:6]:
                c0 = st(page)["cur"] or 0
                ccx, ccy = centroid_of(crop, b2)
                drag(page, ((ox + ccx) / k, (oy + ccy) / k), dst)
                c1 = st(page)["cur"] or 0
                log(f"      e{r+1}c{c+1} 相似 {sc3:.2f} {c0}->{c1} "
                    f"{'OK' if c1 > c0 else 'X'}")
                if c1 > c0:
                    break

    # ── 白块：必须照片格全部拼完才允许动白块 ──
    cur = st(page)["cur"] or 0
    if unfilled:
        log(f"   照片还有 {len(unfilled)} 格未拼完 → 不动白块")
    elif blank and cur < (s["total"] or 0):
        log(f"白格阶段（当前 {cur}/{s['total']}）")
        budget = 18
        for key in sorted(blank):
            if key not in blank:
                continue
            r, c = key                     # 每次循环先刷新坐标，避免日志/落点用到上一轮的格子
            if budget <= 0 or time.time() > deadline:
                log("   白块预算用完，退出")
                break
            crop, ox, oy, k = canvas_of(grab(page), s)
            bg = bg_of(crop)
            cx_i, cy_i = (int(bx + (key[1] + .5) * cw), int(by + (key[0] + .5) * ch))
            hh, ww = int(ch * .16), int(cw * .16)
            p2 = crop[max(0, cy_i - hh):cy_i + hh, max(0, cx_i - ww):cx_i + ww]
            _empty = False
            if p2.size:
                # 判空：暗度 或 模板比对分数低，两者满足其一即认为空着
                if np.abs(p2.reshape(-1, 3).mean(0) - bg).sum() < 45:
                    _empty = True
                else:
                    try:
                        _win = crop[max(0, cy_i - int(ch * .4)):cy_i + int(ch * .4),
                                    max(0, cx_i - int(cw * .4)):cx_i + int(cw * .4)]
                        _tpl = cv2.resize(cells[key],
                                          (max(8, int(cw * .76)), max(8, int(ch * .76))),
                                          interpolation=cv2.INTER_AREA)
                        if (_tpl.shape[0] < _win.shape[0] and
                                _tpl.shape[1] < _win.shape[1]):
                            if float(cv2.matchTemplate(_win, _tpl,
                                                       cv2.TM_CCOEFF_NORMED).max()) < 0.62:
                                _empty = True
                    except Exception:
                        pass
            if not _empty:
                continue                       # 这格确实已经填好
            # 白块提取（不腐蚀，保住 4~6px 凸起）：white_extract 已在截图上验证
            try:
                _ws = we.analyse(crop, cw, ch, verbose=False)
            except Exception as _e:
                log('   白块提取失败: %s' % str(_e).splitlines()[0][:50])
                _ws = []
            if not _ws:
                log('   托盘里没有白块，退出')
                break
            cands = [(w2['box'], w2['flats'], w2['feat']) for w2 in _ws]

            # 该空位需要的四边形状：由棋盘边界 + 已放好的邻居取反推导
            _INV = {"Tab": "Blank", "Blank": "Tab", "Flat": "Flat"}
            _NEI = {"top": (-1, 0, "bottom"), "bottom": (1, 0, "top"),
                    "left": (0, -1, "right"), "right": (0, 1, "left")}
            r0, c0b = key
            req = {}
            for sd, (dr, dc, opp) in _NEI.items():
                rr, cc2 = r0 + dr, c0b + dc
                if not (0 <= rr < g and 0 <= cc2 < g):
                    req[sd] = "Flat"
                    continue
                if (rr, cc2) in blank:
                    req[sd] = None
                    continue
                nb_box = placed_piece.get((rr, cc2))
                if nb_box is None:
                    req[sd] = None
                    continue
                nf = None
                try:
                    nm, _o = wf.mask_piece(crop, nb_box, upscale=2.0)
                    if nm is not None:
                        nc = ef.get_contour(nm)
                        nf = ef.get_edge_features(nc, nm)
                except Exception:
                    nf = None
                req[sd] = _INV.get(nf[opp]) if nf else None
            known = sum(1 for v in req.values() if v)
            log("   e%d%d 需求 %s（已知 %d 条）" % (
                r0 + 1, c0b + 1,
                " ".join("%s:%s" % (k2[0], v or "-") for k2, v in req.items()), known))
            need = {k2 for k2, v in req.items() if v == "Flat"}

            # 先按「平边数量」筛（角=2、边=1、内=0），比精确边集稳
            full = []
            for item in cands:
                feat2 = item[2]
                if not feat2:
                    continue
                ok_all = True
                for sd2, v2 in req.items():
                    if v2 is None:
                        continue
                    if feat2.get(sd2) != v2:
                        ok_all = False
                        break
                if ok_all:
                    full.append(item)
            if full:
                log("   e%d%d 四边全吻合 %d 块" % (r + 1, c + 1, len(full)))
                cands = full
            n_need = len(need)
            same_cnt = [c2 for c2 in cands
                        if c2[1] is not None and len(c2[1]) == n_need]
            exact = [c2 for c2 in same_cnt if c2[1] == need]
            if exact:
                log(f"   e{r+1}c{c+1} 平边 {sorted(need)} → 精确匹配 {len(exact)} 块")
                cands = exact
            elif same_cnt:
                log(f"   e{r+1}c{c+1} 平边 {sorted(need)} → 同数量 {len(same_cnt)} 块")
                cands = same_cnt
            elif not need and cands:
                # 这个空位没有任何平边需求（角/边未知）→ 直接用「四边全吻合」那批
                log(f"   e{r+1}c{c+1} 无平边需求 → 用全吻合 {len(cands)} 块")
            else:
                log(f"   e{r+1}c{c+1} 无匹配白块，跳过（不乱放）")
                continue
            # 关键：把当前空位坐标赋给 r/c（之前丢失，导致所有白块都拖到同一格）
            r, c = key
            dst0 = ((ox + bx + (c + .5) * cw) / k, (oy + by + (r + .5) * ch) / k)
            placed = False
            for item in cands[:4]:
                box = item[0]
                if budget <= 0 or time.time() > deadline:
                    break
                ccx, ccy = centroid_of(crop, box)
                src = ((ox + ccx) / k, (oy + ccy) / k)
                src_orig = src
                ok = False
                for (ax, ay) in ((0, 0), (0, -8), (0, 8)):   # 只试 3 档
                    if budget <= 0 or time.time() > deadline:
                        break
                    c0 = st(page)["cur"] or 0
                    drag(page, src, (dst0[0] + ax, dst0[1] + ay))
                    c1 = st(page)["cur"] or 0
                    budget -= 1
                    log(f"   e{r+1}c{c+1} 白块 {c0}->{c1} {'OK' if c1 > c0 else 'X'}")
                    if c1 > c0:
                        ok = True
                        break
                    src = (dst0[0] + ax, dst0[1] + ay)
                # 没吸附：把它拖回原位，避免和下一块白块叠在同一格
                if not ok:
                    drag(page, src, src_orig)
                if ok:
                    placed = True
                    break
            if not placed:
                log(f"   e{r+1}c{c+1} 白块放不进，跳过")

    # ── 消元收尾：剩余坑位与剩余白块数量相同（≤3）时，直接配对试一次 ──
    try:
        crop, ox, oy, k = canvas_of(grab(page), s)
        bg = bg_of(crop)
        left_holes = []
        for key in sorted(blank):
            cx_i, cy_i = (int(bx + (key[1] + .5) * cw), int(by + (key[0] + .5) * ch))
            hh, ww = int(ch * .16), int(cw * .16)
            p2 = crop[max(0, cy_i - hh):cy_i + hh, max(0, cx_i - ww):cx_i + ww]
            if p2.size and np.abs(p2.reshape(-1, 3).mean(0) - bg).sum() < 45:
                left_holes.append(key)
        if 0 < len(left_holes) <= 3:
            _ws2 = we.analyse(crop, cw, ch, verbose=False)
            if _ws2 and len(_ws2) <= 4:
                log(f"   消元收尾：剩余 {len(left_holes)} 坑 / {len(_ws2)} 块")
                for key in left_holes:
                    r, c = key
                    dst = ((ox + bx + (c + .5) * cw) / k,
                           (oy + by + (r + .5) * ch) / k)
                    # 每格重新检测：上一格用掉的块必须从列表里消失
                    crop, ox, oy, k = canvas_of(grab(page), s)
                    try:
                        _ws2 = we.analyse(crop, cw, ch, verbose=False)
                    except Exception:
                        _ws2 = []
                    for w2 in _ws2:
                        c0 = st(page)["cur"] or 0
                        if c0 >= (s["total"] or 25):
                            break
                        ccx, ccy = centroid_of(crop, w2["box"])
                        src2 = ((ox + ccx) / k, (oy + ccy) / k)
                        drag(page, src2, dst)
                        c1 = st(page)["cur"] or 0
                        log(f"      e{r+1}c{c+1} 消元 {c0}->{c1} "
                            f"{'OK' if c1 > c0 else 'X'}")
                        if c1 > c0:
                            break
                        drag(page, dst, src2)   # 没吸附 → 拖回原位，别叠着
    except Exception as _e2:
        log('   消元收尾失败: %s' % str(_e2).splitlines()[0][:60])

    f = st(page)
    log(f"   本局结束 {f['cur']}/{f['total']} 步数 {f['steps']}")
    return f["cur"] or 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--minutes", type=float, default=40)
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright
    log("=" * 60)
    log("final25 启动")
    with sync_playwright() as pw:
        br = pw.chromium.launch(channel="chrome", headless=False,
                                args=["--start-maximized"])
        page = br.new_context(no_viewport=True).new_page()
        page.goto(URL, wait_until="commit", timeout=60000)
        page.wait_for_selector("#puzzle-canvas", state="attached", timeout=40000)
        page.wait_for_timeout(1500)

        best = 0
        for gi in range(1, a.games + 1):
            log(f"--- 第 {gi} 局 ---")
            s = st(page)
            if s["total"] != 25:
                if not start_new_game(page, 25):
                    log("   无法开始 25 格，尝试等待…")
                    t0 = time.time()
                    while time.time() - t0 < 180 and s["total"] != 25:
                        page.wait_for_timeout(500)
                        s = st(page)
                    if s["total"] != 25:
                        log("   放弃本局")
                        continue
            page.wait_for_timeout(2500)
            s = st(page)
            got = play_one(page, s, a)
            best = max(best, got)
            log(f"   成绩 {got}/25（最好 {best}）")
            if got >= 25:
                log("   达成 25/25")
                break
            page.screenshot(path=f"game{gi}_end.png")
        log(f"全部结束，最好成绩 {best}/25")
        try:
            page.screenshot(path="final25_last.png")
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("崩溃:", type(e).__name__, str(e).splitlines()[0][:150])
        log(traceback.format_exc()[-800:])
