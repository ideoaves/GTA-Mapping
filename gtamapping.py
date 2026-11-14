import bisect
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import torch

TARGETS = 'data/targets.json'
MAP_IMAGE = 'data/map.jpg'
STEP = 2
WALK_STEP = 10
MAP_ZOOM = 1.5
NEAR_M = 600
NEAR_SPEED = 40
LOST_SEC = 60
MIN_NEAR = 6
NEAR_RATIO = 0.9
MIN_GLOBAL = 15
RATIO = 0.8
MATCH_M = 6
FAST_SPEED = 10
WALK_SCALE_RATE = (0.8, 1.38)
CAR_SCALE_RATE = (0.53, 4.2)
SHAPE_SCALE = {'丸': 0.43, '四角': 0.59}
TUNE_RANGE = 200
TUNE_GAIN = 0.7
TUNE_MIN = 100
CAR_TUNE_MIN = 30
MAX_SPEED = 200 / 3.6
TELEPORT_SEC = 40
WARP_SPOTS = [(1748, 2487)]
WARP_M = 1000
STILL_M = 50
LONE_SEC = 60
SPIKE_SEC = 30
SPIKE_SPEED = 20
SMOOTH_SEC = 10
THIN_M = 15
OFFSET_MAX = 600
START_STEP = 600

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
sift = cv2.SIFT_create()


def rim_shape(n, power):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bend = lambda a: np.sign(a) * np.abs(a) ** (2 / power)
    return bend(np.cos(th)), bend(np.sin(th))


FRAME = (1280, 720)
REF_HUD = (4, 622)
SHAPES = {
    '丸': {'REF_CROP': (14, 450, 210, 614), 'REF_PLAYER': (108, 573), 'REF_ELLIPSE': (111.5, 529.5, 86.5, 71.5),
          'SQUASH': 1.12, 'CAR_TILT': (1.62, 0.00105), 'RIM_POWER': 2, 'RIM_K': (0.97, 0.99, 1.01, 1.03), 'RIM_IN': 0.92, 'CAR_RIM': -5},
    '四角': {'REF_CROP': (15, 531, 153, 618), 'REF_PLAYER': (83.7, 592), 'REF_ELLIPSE': (84, 574.3, 68.7, 43.7),
           'SQUASH': 0.885, 'CAR_TILT': (1.6, 0.003), 'RIM_POWER': 16, 'RIM_K': (0.99, 1.0, 1.01), 'RIM_IN': 0.94, 'CAR_RIM': 10},
}


class Stream:
    def __init__(self, name, target):
        self.name, self.target, self.video, self.path = name, target, f'video/{name}.mp4', f'probes/{name}.json'
        self.state = json.load(open(self.path, encoding='utf-8')) if os.path.exists(self.path) else {}
        self.dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', self.video], capture_output=True, text=True).stdout)

    def save(self):
        json.dump(self.state, open(self.path, 'w', encoding='utf-8'), ensure_ascii=False)

    def grab(self, t):
        return np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', self.video, '-frames:v', '1', '-vf', f'scale={FRAME[0]}:{FRAME[1]}',
                                             '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True).stdout, np.uint8).reshape(FRAME[1], FRAME[0], 3)

    def find_hud(self, samples=24):
        if 'hud' in self.state:
            return True
        tpl, hits = cv2.imread('data/hud.png'), []
        for k in range(samples):
            fr = self.grab(self.dur * (k + 0.5) / samples)
            best = max(((cv2.minMaxLoc(cv2.matchTemplate(fr, cv2.resize(tpl, None, fx=s, fy=s), cv2.TM_CCOEFF_NORMED))[1:4:2], s) for s in np.arange(0.5, 2.05, 0.05)), key=lambda b: b[0][0])
            if best[0][0] > 0.4:
                hits.append((best[0][1], best[1]))
        if len(hits) < 3:
            return False
        self.state['hud'] = {'scale': round(float(np.median([h[1] for h in hits])), 3), 'at': [round(float(np.median([h[0][i] for h in hits]))) for i in (0, 1)]}
        self.save()
        return True

    def fit_minimap(self, force=None, samples=40):
        if 'map' in self.state and force in (None, self.state['map']['shape']):
            return self.state['map']
        si, (ix, iy) = self.state['hud']['scale'], self.state['hud']['at']

        def rim_search(G, sh):
            u, v = rim_shape(72, sh['RIM_POWER'])
            ex, ey, rx, ry = sh['REF_ELLIPSE']

            def score(cx, cy, s):
                ring = lambda k: G[:, np.clip((cy + ry * s * k * v).astype(int), 0, G.shape[1] - 1), np.clip((cx + rx * s * k * u).astype(int), 0, G.shape[2] - 1)].astype(np.float32)
                on = np.max([ring(k) for k in sh['RIM_K']], axis=0)
                margin = np.minimum.reduce([np.percentile(on, 20, axis=1) - 80, np.percentile(on - ring(sh['RIM_IN']), 20, axis=1) - sh['CAR_RIM'],
                                            np.percentile(on - ring(1.08), 20, axis=1) - 15])
                return (margin > 0).sum() + np.clip(margin, 0, None).sum() / 1e4
            at = lambda s, dx, dy: (ix + (ex - REF_HUD[0]) * s + dx, iy + (ey - REF_HUD[1]) * s + dy, s)
            best = max(((score(*at(s, dx, dy)), *at(s, dx, dy)) for s in np.arange(0.7, 2.21, 0.05) for dx in range(-60, 61, 4) for dy in range(-60, 61, 4)), key=lambda b: b[0])
            best = max(((score(best[1] + dx, best[2] + dy, best[3] + ds), best[1] + dx, best[2] + dy, best[3] + ds)
                        for ds in np.arange(-0.05, 0.051, 0.01) for dx in range(-4, 5) for dy in range(-4, 5)), key=lambda b: b[0])
            return int(best[0]), float(best[1]), float(best[2]), float(best[3])
        for count in (samples, samples * 3):
            G = np.stack([cv2.cvtColor(self.grab(self.dur * (k + 0.5) / count), cv2.COLOR_BGR2GRAY) for k in range(count)])
            found = {shape: rim_search(G, SHAPES[shape]) for shape in ([force] if force else SHAPES)}
            shape = max(found, key=lambda k: found[k][0])
            if found[shape][0] >= 2:
                break
        n, cx, cy, s = found[shape]
        if n < 2:
            shape, s = force or '四角', si
            cx, cy = ix + (SHAPES[shape]['REF_ELLIPSE'][0] - REF_HUD[0]) * s, iy + (SHAPES[shape]['REF_ELLIPSE'][1] - REF_HUD[1]) * s
        ex, ey = SHAPES[shape]['REF_ELLIPSE'][:2]
        self.state['map'] = {'shape': shape, 'scale': round(s, 3), 'hud': [round(cx - (ex - REF_HUD[0]) * s, 1), round(cy - (ey - REF_HUD[1]) * s, 1)],
                             'walk': round(SHAPE_SCALE[shape] * s, 3), 'rims': f'{n}/{count}（{"・".join(f"{k}{v[0]}" for k, v in found.items())}）'}
        self.save()
        return self.state['map']

    def guess_start(self):
        import easyocr
        reader, jst, found = easyocr.Reader(['en'], gpu=False, verbose=False), timezone(timedelta(hours=9)), []
        for t in np.arange(60, self.dur - 60, START_STEP):
            m = re.search(r'(20\d\d)\D(\d\d)\D(\d\d)\s*(\d{1,2}):(\d{2})', ' '.join(reader.readtext(cv2.resize(self.grab(t)[0:40, 640:1280], None, fx=2, fy=2), detail=0, allowlist='0123456789-: ')))
            try:
                found.append(datetime(*map(int, m.groups()), tzinfo=jst).timestamp() - t)
            except (AttributeError, ValueError):
                pass
        return round(max(found, key=lambda c: sum(abs(c - o) <= 90 for o in found))) if found else None

    def measure_offset(self):
        if 'offset' in self.state:
            return f'時刻のずれ {self.state["offset"]:+d}秒'
        import easyocr
        reader, start = easyocr.Reader(['en'], gpu=False, verbose=False), self.target['start']
        base = datetime.fromtimestamp(start, timezone(timedelta(hours=9)))

        def real(hm):
            r = base.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
            return (r + (timedelta(days=1) if r < base - timedelta(hours=3) else timedelta(0))).timestamp()

        def clock(t, dated):
            text = ' '.join(reader.readtext(cv2.resize(self.grab(t)[0:40, 640:1280], None, fx=2, fy=2), detail=0, allowlist='0123456789-: '))
            m = re.search(r'20\d\d\D\d\d\D\d\d\s*(\d{1,2}):(\d{2})' if dated else r'(?<![\d:-])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])', text)
            hm = (int(m[1]), int(m[2])) if m and int(m[1]) < 24 and int(m[2]) < 60 else None
            return hm if hm and abs(real(hm) - (start + t)) <= 1800 else None

        def flip(found, dated, k=0, prev=None):
            if k >= 70:
                return None
            hm = clock(found + k, dated)
            if hm and prev and hm != prev[1] and k - prev[0] <= 3:
                return round(real(hm) - (start + found + k))
            return flip(found, dated, k + 1, (k, hm) if hm else prev)
        raw = next((r for dated in (True, False) for found in [t for t in np.linspace(60, self.dur - 120, 20).astype(int) if clock(t, dated)][:5]
                    if (r := flip(found, dated)) is not None), None)
        fixed = raw if raw is not None and abs(raw) <= OFFSET_MAX else min((raw - sign * h * 3600 for h in (9, 12, 24) for sign in (1, -1)), key=abs) if raw is not None else 0
        self.state['offset'] = fixed if abs(fixed) <= OFFSET_MAX else 0
        self.save()
        return f'時刻のずれ {raw}秒' + ('' if raw == self.state['offset'] else f' → {self.state["offset"]:+d}秒に修正') if raw is not None else '時刻を読めないので0秒'

    def setup(self):
        m, hd = self.state['map'], self.state['hud']
        sh, s, (mx, my), si, (hx, hy) = SHAPES[m['shape']], m['scale'], m['hud'], hd['scale'], hd['at']
        at = lambda px, py: (mx + (px - REF_HUD[0]) * s, my + (py - REF_HUD[1]) * s)
        c = [int(round(v / 2) * 2) for v in (*at(*sh['REF_CROP'][:2]), *at(*sh['REF_CROP'][2:]))]
        up = 3 / s
        to_big = lambda px, py: np.array([(px - c[0]) * up * sh['SQUASH'], (py - c[1]) * up])
        P = to_big(*at(*sh['REF_PLAYER']))
        T = np.array([[1, 0, 0], [0, sh['CAR_TILT'][0], 0], [0, sh['CAR_TILT'][1], 1]]) @ np.array([[1, 0, -P[0]], [0, 1, -P[1]], [0, 0, 1]])
        ex, ey, rx, ry = sh['REF_ELLIPSE']
        u, v = rim_shape(72, sh['RIM_POWER'])
        flat = cv2.perspectiveTransform(np.float32([[to_big(*at(ex + rx * a, ey + ry * b)) for a, b in zip(u, v)]]), T)[0]
        (fx0, fy0), (fx1, fy1) = flat.min(0), flat.max(0)
        x0, y0 = max(0, min(c[0], hx - 4)) // 2 * 2, max(0, min(c[1], int(hy - 40 * si))) // 2 * 2
        x1, y1 = min(FRAME[0], max(c[2] + int(70 * s), int(hx + 210 * si))) // 2 * 2, min(FRAME[1], max(c[3], int(hy + 26 * si))) // 2 * 2
        self.geo = dict(sh=sh, crop=c, up=up, player=P, car_warp=np.array([[1, 0, -fx0], [0, 1, -fy0], [0, 0, 1]]) @ T, car_size=(int(fx1 - fx0), int(fy1 - fy0)),
                        car_player=np.array([-fx0, -fy0]), region=(x0, y0, x1 - x0, y1 - y0),
                        speedos=[cv2.resize(cv2.imread('data/speedo.png', 0), None, fx=k / 1.5, fy=k / 1.5) for k in {si, s}],
                        speedo_at=(min(int(hx + 140 * si), c[2] - int(30 * s)) - x0, min(int(hy - 40 * si), c[3] - int(40 * s)) - y0),
                        rim=(at(ex, ey)[0] - x0, at(ex, ey)[1] - y0, rx * s, ry * s),
                        walk=[r * m['walk'] for r in WALK_SCALE_RATE], car=[r * m['walk'] for r in CAR_SCALE_RATE])

    def locate(self, crop, car, P, D, center, radius, need, wide=False, ratio=RATIO):
        g = self.geo
        big = cv2.resize(crop, None, fx=g['up'] * g['sh']['SQUASH'], fy=g['up'], interpolation=cv2.INTER_CUBIC)
        mask = np.full(big.shape[:2], 255, np.uint8)
        if car:
            big, mask = (cv2.warpPerspective(a, g['car_warp'], g['car_size']) for a in (big, mask))
        kp, d = sift.detectAndCompute(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY), cv2.erode(mask, np.ones((15, 15), np.uint8)))
        if d is None or len(kp) < need:
            return None
        if center is not None:
            idx = torch.nonzero(((P - torch.tensor(center, device=dev)) ** 2).sum(1) < radius ** 2).squeeze(1)
            if len(idx) < 2:
                return None
            P, D = P[idx], D[idx]
        q = torch.tensor(d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6), device=dev).half()
        src, dst = [], []
        for k in range(0, len(q), 32):
            top = (q[k:k + 32] @ D.T).topk(2, dim=1)
            dist = torch.sqrt(torch.clamp(2 - 2 * top.values.float(), min=0))
            for j in torch.nonzero(dist[:, 0] < ratio * dist[:, 1]).squeeze(1).tolist():
                src.append(kp[k + j].pt)
                dst.append(P[top.indices[j, 0]].tolist())
        if len(src) < need:
            return None
        M, inl = cv2.estimateAffinePartial2D(np.float32(src), np.float32(dst), method=cv2.RANSAC, ransacReprojThreshold=MATCH_M)
        lo, hi = g['car'] if car or wide else g['walk']
        if M is None or inl.sum() < need or not lo < math.hypot(M[0, 0], M[1, 0]) < hi:
            return None
        x, y = M @ [*(g['car_player'] if car else g['player']), 1]
        return round(float(x), 1), round(float(-y), 1), int(inl.sum()), round(math.degrees(math.atan2(M[1, 0], M[0, 0])), 1), M

    def match(self, report):
        if os.path.exists('probes/map_features.npz'):
            z = np.load('probes/map_features.npz')
            p, d = z['pts'], z['desc']
        else:
            S, X0, Y0 = 0.6818144931769898, 2826.1523153159696, 5735.495139738179
            img = cv2.resize(cv2.imread(MAP_IMAGE, cv2.IMREAD_GRAYSCALE), None, fx=MAP_ZOOM, fy=MAP_ZOOM, interpolation=cv2.INTER_CUBIC)
            tile, pad, pts, desc = 2048, 64, [], []
            for y in range(0, img.shape[0], tile):
                for x in range(0, img.shape[1], tile):
                    ox, oy = max(0, x - pad), max(0, y - pad)
                    kp, dd = sift.detectAndCompute(img[oy:y + tile + pad, ox:x + tile + pad], None)
                    if dd is None:
                        continue
                    q = np.float32([k.pt for k in kp]) + [ox, oy]
                    core = (q[:, 0] >= x) & (q[:, 0] < x + tile) & (q[:, 1] >= y) & (q[:, 1] < y + tile)
                    pts.append(q[core])
                    desc.append(dd[core])
            p, d = np.concatenate(pts), np.concatenate(desc)
            p = np.stack([(p[:, 0] / MAP_ZOOM - X0) / S, (p[:, 1] / MAP_ZOOM - Y0) / S], 1).astype(np.float32)
            d = (d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)).astype(np.float16)
            np.savez('probes/map_features.npz', pts=p, desc=d)
        P, D = torch.tensor(p, device=dev), torch.tensor(d, device=dev)
        g = self.geo
        (x0, y0, w, h), c = g['region'], g['crop']
        (cx, cy, rx, ry), (sx, sy), sh = g['rim'], g['speedo_at'], g['sh']
        u, v = rim_shape(180, sh['RIM_POWER'])
        ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', self.video, '-vf', f'fps=1/{STEP},scale={FRAME[0]}:{FRAME[1]},crop={w}:{h}:{x0}:{y0}',
                               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
        cands, last, t, fast, tried = [], None, -STEP, False, 0
        while buf := ff.stdout.read(w * h * 3):
            t += STEP
            img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ring = lambda k: gray[np.clip((cy + ry * k * v).astype(int), 0, gray.shape[0] - 1), np.clip((cx + rx * k * u).astype(int), 0, gray.shape[1] - 1)].astype(float)
            on = np.max([ring(k) for k in sh['RIM_K']], axis=0)
            seen = bool(np.percentile(on, 20) > 80 and np.percentile(on - ring(sh['RIM_IN']), 20) > sh['CAR_RIM']) or \
                max(cv2.matchTemplate(gray[sy:, sx:], sp, cv2.TM_CCOEFF_NORMED).max() for sp in g['speedos']) >= 0.6
            car = seen or bool(self.state['map'].get('blind'))
            if not car and not fast and t % WALK_STEP:
                continue
            tried += 1
            crop = img[c[1] - y0:c[3] - y0, c[0] - x0:c[2] - x0]
            search = lambda car, wide: (last and self.locate(crop, car, P, D, (last[1], -last[2]), NEAR_M + NEAR_SPEED * (t - last[0]), MIN_NEAR, wide, NEAR_RATIO)) or \
                (None if last and t - last[0] <= LOST_SEC else self.locate(crop, car, P, D, None, 0, MIN_GLOBAL, wide))
            r, mode = search(car, fast), 'c' if car else 'p' if fast else 'w'
            if car and (flat := search(False, False)) and (not r or flat[2] > r[2]):
                r, mode = flat, 'p' if seen else 'w'
            if r:
                cands.append([t, r[0], r[1], r[2], mode, r[3], r[4]])
                fast = bool(last) and t - last[0] <= LOST_SEC and math.hypot(r[0] - last[1], r[1] - last[2]) > FAST_SPEED * (t - last[0])
                last = (t, r[0], r[1])
            if t % 60 == 0:
                report(t=t, total=int(self.dur), found=len(cands), tried=tried)
        return cands

    def build(self, cands):
        def tune(Ms, ts, player, span, need, anchor=None):
            if len(ts) < 3:
                return player
            if anchor is not None:
                j = np.clip(np.searchsorted(anchor[0], ts), 1, len(anchor[0]) - 1)
                j = np.where(np.abs(anchor[0][j - 1] - ts) < np.abs(anchor[0][j] - ts), j - 1, j)
                use = np.abs(anchor[0][j] - ts) <= 4
                n, score = int(use.sum()), lambda o: np.median(np.hypot(*(np.einsum('nij,j->ni', Ms[use], [player[0] + o[0], player[1] + o[1], 1]) - anchor[1][j[use]]).T))
            else:
                i = np.where(ts[2:] - ts[:-2] <= span)[0] + 1
                k = ((ts[i] - ts[i - 1]) / (ts[i + 1] - ts[i - 1]))[:, None]
                n = len(i)

                def score(o):
                    q = np.einsum('nij,j->ni', Ms, [player[0] + o[0], player[1] + o[1], 1])
                    return np.median(np.hypot(*(q[i] - q[i - 1] - (q[i + 1] - q[i - 1]) * k).T))
            if n < need:
                return player
            best = min(((dx, dy) for dx in range(-TUNE_RANGE, TUNE_RANGE + 1, 6) for dy in range(-TUNE_RANGE, TUNE_RANGE + 1, 6)), key=score)
            best = min(((best[0] + dx, best[1] + dy) for dx in range(-6, 7) for dy in range(-6, 7)), key=score)
            return player + best if score(best) < TUNE_GAIN * score((0, 0)) and max(map(abs, best)) <= TUNE_RANGE else player
        g = self.geo
        walk, car = [c for c in cands if c[4] == 'w'], [c for c in cands if c[4] == 'c']
        Mw, tw, Mc, tc = np.array([c[6] for c in walk]), np.array([c[0] for c in walk], float), np.array([c[6] for c in car]), np.array([c[0] for c in car], float)
        g['player'] = tune(Mw, tw, g['player'], 25, TUNE_MIN)
        anchor = (tw, np.einsum('nij,j->ni', Mw, [*g['player'], 1])) if len(tw) > 1 else None
        j = np.clip(np.searchsorted(tw, tc), 1, max(1, len(tw) - 1))
        pairs = int((np.minimum(np.abs(tw[j] - tc), np.abs(tw[j - 1] - tc)) <= 4).sum()) if anchor and len(tc) else 0
        g['car_player'] = tune(Mc, tc, g['car_player'], 6, CAR_TUNE_MIN, anchor) if pairs >= CAR_TUNE_MIN else tune(Mc, tc, g['car_player'], 6, TUNE_MIN)
        pts = [[c[0], *[round(float(v) * s, 1) for v, s in zip(c[6] @ [*(g['car_player'] if c[4] == 'c' else g['player']), 1], (1, -1))], *c[3:6]] for c in cands]
        kept = []
        for p in smooth(scattered(lone(warp(select_track(pts))))):
            q = kept[-1] if kept else None
            if not (q and p[4] in ('c', 'p') and len(p) <= 6 and p[0] - q[0] < WALK_STEP and math.hypot(p[1] - q[1], p[2] - q[2]) < THIN_M):
                kept.append(p)
        json.dump({'member': self.target['member'], 'video': self.target['video_id'], 'start': self.target['start'], 'offset': self.state['offset'], 'interval': [WALK_STEP, STEP], 'points': kept},
                  open(f'site/tracks/{self.name}.json', 'w', encoding='utf-8'))
        return len(kept)


def select_track(cands, window=150):
    reach = lambda a, b: math.hypot(a[1] - b[1], a[2] - b[2]) <= MAX_SPEED * (b[0] - a[0]) + 30
    score, prev, jump, best = [], [], [], -1
    for i, c in enumerate(cands):
        w = WALK_STEP if c[4] == 'w' else STEP
        s, p, tp = w, -1, False
        for j in {*range(max(0, i - window), i), best} - {-1}:
            if cands[j][0] < c[0]:
                far = not reach(cands[j], c)
                v = score[j] + w - TELEPORT_SEC * far
                if v > s:
                    s, p, tp = v, j, far
        score.append(s)
        prev.append(p)
        jump.append(tp)
        if best < 0 or s >= score[best]:
            best = i
    path = []
    while best >= 0:
        path.append(cands[best] + [1] * jump[best])
        best = prev[best]
    return path[::-1]


def warp(pts):
    near = lambda p: any(math.hypot(p[1] - x, p[2] - y) < 100 for x, y in WARP_SPOTS)
    return [p + [1] if len(p) == 6 and i and near(p) and math.hypot(p[1] - pts[i - 1][1], p[2] - pts[i - 1][2]) > WARP_M else p for i, p in enumerate(pts)]


def lone(pts):
    dist = lambda p, q: math.hypot(p[1] - q[1], p[2] - q[2])
    out, i, ts = [], 0, [p[0] for p in pts]
    while i < len(pts):
        a = pts[i]
        out.append(a)
        j = next((j for j in range(i + 1, len(pts)) if pts[j][0] - a[0] > LONE_SEC or dist(a, pts[j]) <= STILL_M), len(pts))
        back = j < len(pts) and pts[j][0] - a[0] <= LONE_SEC and all(dist(a, q) > 2 * STILL_M for q in pts[i + 1:j])
        if back:
            lo, hi = bisect.bisect_left(ts, a[0] - LONE_SEC), bisect.bisect_right(ts, pts[j][0] + LONE_SEC)
            back = sum(dist(a, q) <= STILL_M for q in pts[lo:hi]) > j - i - 1
        i = j if back else i + 1
    return out


def scattered(pts):
    def stray(i):
        p = pts[i]
        nb = [q for q in pts[max(0, i - 30):i + 31] if q is not p and abs(q[0] - p[0]) <= SPIKE_SEC]
        if len(nb) < 3:
            return False
        mx, my = np.median([q[1] for q in nb]), np.median([q[2] for q in nb])
        calm = sum(math.hypot(q[1] - mx, q[2] - my) <= STILL_M for q in nb) >= len(nb) / 2
        alone = not any(math.hypot(q[1] - p[1], q[2] - p[2]) <= STILL_M for q in nb)
        return calm and alone and math.hypot(p[1] - mx, p[2] - my) > STILL_M + SPIKE_SPEED * min(abs(q[0] - p[0]) for q in nb)
    return [p for i, p in enumerate(pts) if not stray(i)]


def smooth(pts):
    seg, ts, out = np.cumsum([len(p) > 6 for p in pts]), [p[0] for p in pts], []
    for i, p in enumerate(pts):
        lo, hi = bisect.bisect_left(ts, p[0] - SMOOTH_SEC), bisect.bisect_right(ts, p[0] + SMOOTH_SEC)
        nb = [pts[k] for k in range(lo, hi) if seg[k] == seg[i]]
        out.append(p if len(nb) < 3 else [p[0], round(float(np.median([q[1] for q in nb])), 1), round(float(np.median([q[2] for q in nb])), 1), *p[3:]])
    return out


def progress(state, name, **kw):
    state['streams'][name].update(kw, now=time.time())
    try:
        disk = json.load(open('probes/progress.json', encoding='utf-8'))
    except (OSError, ValueError):
        disk = state
    disk['streams'][name] = state['streams'][name]
    json.dump(disk, open('probes/progress.json', 'w', encoding='utf-8'), ensure_ascii=False)


def log(msg):
    print(time.strftime('%H:%M:%S ') + msg, flush=True)


PROBE_KEYS = ('hud', 'map', 'offset')


def pending(arg, failed, finished):
    targets = json.load(open(TARGETS, encoding='utf-8')) if os.path.exists(TARGETS) else {}
    probe = lambda n: json.load(open(f'probes/{n}.json', encoding='utf-8')) if os.path.exists(f'probes/{n}.json') else {}
    same = lambda n: (s := probe(n)).get('used') == {k: s.get(k) for k in PROBE_KEYS}
    redo = lambda n: n == arg and not same(n)
    wanted = lambda n: (arg is None or n == arg or n.endswith(f'_day{arg}')) and (redo(n) or not os.path.exists(f'site/tracks/{n}.json')) and n not in failed + finished
    if arg in targets and arg not in failed + finished and os.path.exists(f'site/tracks/{arg}.json') and same(arg):
        log(f'{arg} は probes/{arg}.json が前回と同じなので回し直さないよ')
    for name in sorted(f[:-4] for f in os.listdir('video') if re.fullmatch(r'\w+_day\d+\.mp4', f)):
        if name in targets or not wanted(name):
            continue
        info = json.load(open(f'video/{name}.info.json', encoding='utf-8')) if os.path.exists(f'video/{name}.info.json') else {}
        target = {'member': re.sub(r'\d+$', '', name.rsplit('_day', 1)[0]), 'video_id': info.get('id', ''), 'start': info.get('release_timestamp') or info.get('timestamp')}
        target['start'] = target['start'] or Stream(name, target).guess_start()
        if not target['start']:
            failed.append(name)
            log(f'{name} 開始時刻がわからないので飛ばすよ（info.json も時計も読めない）')
            continue
        targets[name] = target
        json.dump(targets, open(TARGETS, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log(f'{name} を登録 {datetime.fromtimestamp(target["start"], timezone(timedelta(hours=9))):%Y-%m-%d %H:%M:%S} 開始' + ('' if info else '（時計から逆算）'))
    return targets, [n for n in targets if wanted(n) and os.path.exists(f'video/{n}.mp4')]


def analyze(name, target, state, force):
    t0, st = time.time(), Stream(name, target)
    progress(state, name, status='HUD', begin=t0)
    if not st.find_hud():
        progress(state, name, status='失敗', note='HUDが見当たらないよ')
        log(f'{name} キャラの裏にあるなど特定不可だよ')
        return False
    progress(state, name, status='ミニマップ')
    m = st.fit_minimap(force)
    log(f'{name} ミニマップ {m["shape"]} {m["scale"]}倍 縁{m["rims"]} 徒歩の縮尺{m["walk"]}')
    progress(state, name, status='時刻')
    log(f'{name} {st.measure_offset()}')
    st.setup()
    progress(state, name, status='照合', match_begin=time.time())
    cands = st.match(lambda **kw: progress(state, name, **kw))
    if os.path.exists(f'site/tracks/{name}.json'):
        os.replace(f'site/tracks/{name}.json', f'probes/{name}_old_{next(k for k in range(10 ** 4) if not os.path.exists(f"probes/{name}_old_{k}.json"))}.json')
    n = st.build(cands)
    st.state['used'] = {k: st.state[k] for k in PROBE_KEYS}
    st.save()
    index, targets = {}, json.load(open(TARGETS, encoding='utf-8'))
    ev = json.load(open('site/event.json', encoding='utf-8')) if os.path.exists('site/event.json') else {'title': '', 'days': [], 'groupColors': {}, 'members': [], 'places': []}
    for key, t in targets.items():
        day = int(key.rsplit('day', 1)[1])
        ev['days'] += [''] * (day - len(ev['days']))
        ev['days'][day - 1] = ev['days'][day - 1] or f'{datetime.fromtimestamp(t["start"], timezone(timedelta(hours=9))):%Y-%m-%d}'
        if os.path.exists(f'site/tracks/{key}.json'):
            index.setdefault(t['member'], {}).setdefault(day, []).append(f'tracks/{key}.json')
    json.dump({mem: {d: f[0] if len(f) == 1 else sorted(f) for d, f in sorted(days.items())} for mem, days in index.items()},
              open('site/tracks.json', 'w', encoding='utf-8'), ensure_ascii=False)
    ids = {m['id'] for m in ev['members']}
    ev['members'] += [{'id': mem, 'name': mem, 'color': '#000000', 'group': ''} for mem in dict.fromkeys(t['member'] for t in targets.values()) if mem not in ids]
    open('site/event.json', 'w', encoding='utf-8').write(re.sub(r'\{\n\s+([^{}\[\]]*?)\n\s+\}', lambda k: '{' + re.sub(r',\n\s+', ', ', k[1]) + '}', json.dumps(ev, ensure_ascii=False, indent=1)) + '\n')
    progress(state, name, status='完了', found=len(cands), kept=n)
    log(f'{name} 完了 {int(time.time() - t0) // 60}分 {len(cands)}点 → 採用{n}点')
    return True


def main(arg, force):
    failed, finished = [], []
    probe = f'probes/{arg}.json'
    if force and os.path.exists(probe) and (s := json.load(open(probe, encoding='utf-8'))).get('map', {}).get('shape') != force:
        s.pop('map', None)
        json.dump(s, open(probe, 'w', encoding='utf-8'), ensure_ascii=False)
    targets, names = pending(arg, failed, finished)
    log(f'開始 {len(names)}本: {" ".join(names)}')
    old = json.load(open('probes/progress.json', encoding='utf-8')) if os.path.exists('probes/progress.json') else {'start': time.time(), 'streams': {}}
    state = {'start': old['start'], 'streams': {**old['streams'], **{n: {'status': '待ち'} for n in names}}}
    while names:
        (finished if analyze(names[0], targets[names[0]], state, force) else failed).append(names[0])
        targets, names = pending(arg, failed, finished)
        for n in names:
            if state['streams'].get(n, {}).get('status') != '待ち':
                state['streams'][n] = {'status': '待ち'}
                log(f'{n} を一覧に追加')
    log(f'完了 {len(finished)}本 失敗{len(failed)}本 {" ".join(failed)}')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs('probes', exist_ok=True)
    os.makedirs('site/tracks', exist_ok=True)
    words = [a for a in sys.argv[1:] if not a.startswith('--')]
    shapes = [{'--round': '丸', '--rect': '四角'}[a] for a in sys.argv[1:] if a in ('--round', '--rect')]
    server = None
    if not os.environ.get('GTAMAPPING_NO_BROWSER'):
        import edit
        server = edit.open_page('progress.html', False)
    main(words[0] if words else None, shapes[0] if shapes else None)
    if server:
        input('Enter で閉じます')
