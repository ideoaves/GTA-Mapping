import itertools, json, math, os, sys, time
import numpy as np, torch

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
import gtamapping as G

STEP = 0.5
OUT_STEP = 2
RADIUS = 1000
NEED = 4
RATIO = 0.9
NEAR_SEC = 3
NEAR_M = 70


def main(name):
    path = f'site/tracks/{name}.json'
    data = json.load(open(path, encoding='utf-8'))
    pts = data['points']
    runs = [(g[0][0] - 1, g[-1][0] + 1) for red, g in ((k, list(g)) for k, g in itertools.groupby(enumerate(pts), lambda e: e[1][4] == 'f'))
            if red and 0 < g[0][0] and g[-1][0] + 1 < len(pts)]
    if not runs:
        return G.log(f'{name} 赤い点がないよ')
    if not os.path.exists(f'video/{name}.mp4'):
        return G.log(f'{name} video/{name}.mp4 が無いので照合できないよ')
    st = G.Stream(name, data)
    if 'map' not in st.state and not (st.find_hud() and st.fit_minimap()):
        return G.log(f'{name} HUDが見つからないので照合できないよ')
    st.setup()
    z = np.load('probes/map_features.npz')
    P, D, c = torch.tensor(z['pts'], device=G.dev), torch.tensor(z['desc'], device=G.dev), st.geo['crop']
    out, last = [], 0
    for a, b in runs:
        t0, guide, cand, best = time.time(), pts[a:b + 1], [], {}
        ts = np.array([p[0] for p in guide], float)
        for t in np.arange(ts[0] + STEP, ts[-1], STEP):
            crop = st.grab(t)[c[1]:c[3], c[0]:c[2]]
            cx, cy = np.interp(t, ts, [p[1] for p in guide]), np.interp(t, ts, [p[2] for p in guide])
            hits = [r + ('c' if car else 'w',) for car in (True, False) if (r := st.locate(crop, car, P, D, (cx, -cy), RADIUS, NEED, True, RATIO))]
            if hits:
                r = max(hits, key=lambda r: r[2])
                cand.append([round(float(t), 1), r[0], r[1], r[2], r[5], r[3]])
        for p in cand:
            n = [q for q in cand if q is not p and abs(q[0] - p[0]) <= NEAR_SEC]
            k = round(p[0] / OUT_STEP)
            if len(n) >= 2 and math.hypot(p[1] - np.median([q[1] for q in n]), p[2] - np.median([q[2] for q in n])) < NEAR_M and (k not in best or p[3] > best[k][3]):
                best[k] = p
        kept, hand = sorted(best.values()), pts[a + 1:b]
        out += pts[last:a + 1] + (kept or hand)
        last = b
        G.log(f'{name} {pts[a][0]}〜{pts[b][0]}秒 赤い点{len(hand)}個 → 照合{len(cand)}点 採用{len(kept)}点 {time.time() - t0:.0f}秒' + ('（赤い点のまま）' if not kept else ''))
    data['points'] = out + pts[last:]
    os.replace(path, f'probes/{name}_hand.json')
    json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    G.log(f'{name} {len(pts)}点 → {len(data["points"])}点')


if __name__ == '__main__':
    for name in sys.argv[1:]:
        main(name)
