#!/usr/bin/env python3
"""
BindCORE — Graphical Abstract SVG Generator
Illustration: IDP conformational ensemble with predicted binding residues.
Optimized for: White background, small-scale visibility (thumbnails).
Fix: Adjusted random walk parameters to prevent unnaturally straight chains.
Update: All text, legend, and annotation lines removed.
"""

import numpy as np
from scipy.interpolate import splprep, splev

# ─── Canvas ────────────────────────────────────────────────────────────────
W, H   = 1400, 900
CX, CY = W // 2, H // 2

# ─── IDP generation helpers ────────────────────────────────────────────────
def generate_idp(n=75, spread=210, cx=CX, cy=CY, seed=0, wiggliness=1.25):
    """Persistent random walk (IDP backbone in 2-D)."""
    rng = np.random.default_rng(seed)
    # 'wiggliness' controls the standard deviation of angle changes. 
    # Higher value = more coiling, fewer straight lines.
    angles = np.cumsum(rng.normal(0, wiggliness, n))
    step   = spread / np.sqrt(n) * 1.65
    x = np.cumsum(np.cos(angles) * step)
    y = np.cumsum(np.sin(angles) * step)
    x = x - x.mean() + cx
    y = y - y.mean() + cy
    return x, y

def smooth_spline(x, y, n_out=500, s_factor=1.6):
    try:
        tck, _ = splprep([x, y], s=len(x) * s_factor, k=3)
        return splev(np.linspace(0, 1, n_out), tck)
    except Exception:
        return x, y

def to_path(x, y, step=2):
    pts = list(zip(x[::step], y[::step]))
    d   = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for px, py in pts[1:]:
        d += f" L{px:.1f},{py:.1f}"
    return d

# ─── Color palette (Optimized for White Background) ────────────────────────
PALETTE = [
    "#03045e", "#0077b6", "#00b4d8", "#005f73",
    "#0a9396", "#94d2bd", "#ee9b00", "#ca6702",
    "#bb3e03", "#ae2012", "#9b2226", "#6a040f",
    "#3c096c", "#5a189a", "#7b2cbf", "#2d00f7",
    "#3d405b", "#81b29a",
]

HERO_COLOR  = "#5a189a"   # hero chain: deep purple
BIND_COLOR  = "#fb8500"   # binding residues: bold orange
N_COLOR     = "#0077b6"   # N-terminus: deep cyan/blue
C_COLOR     = "#d90429"   # C-terminus: crimson red
# TEXT_COLOR removed

# ─── SVG builder ───────────────────────────────────────────────────────────
out = []

out.append(
    f'<svg xmlns="http://www.w3.org/2000/svg"'
    f' width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
)

# ── DEFS ───────────────────────────────────────────────────────────────────
out.append(f"""  <defs>

    <filter id="glow-xs" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>

    <filter id="glow-md" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>

    <filter id="glow-lg" x="-100%" y="-100%" width="300%" height="300%">
      <feGaussianBlur in="SourceGraphic" stdDeviation="15" result="b"/>
      <feMerge>
        <feMergeNode in="b"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>

    <radialGradient id="gold-sphere" cx="30%" cy="30%" r="70%">
      <stop offset="0%"   stop-color="#ffea00"/>
      <stop offset="50%"  stop-color="#ff9100"/>
      <stop offset="100%" stop-color="#cc5500"/>
    </radialGradient>
    <radialGradient id="n-sphere" cx="30%" cy="30%" r="70%">
      <stop offset="0%"  stop-color="#48cae4"/>
      <stop offset="100%" stop-color="#023e8a"/>
    </radialGradient>
    <radialGradient id="c-sphere" cx="30%" cy="30%" r="70%">
      <stop offset="0%"  stop-color="#ff758f"/>
      <stop offset="100%" stop-color="#a4161a"/>
    </radialGradient>

    <clipPath id="cvs">
      <rect x="0" y="0" width="{W}" height="{H}"/>
    </clipPath>

  </defs>""")

# Solid white background
out.append(f'  <rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
out.append('  <g clip-path="url(#cvs)">')

# ── STAR FIELD  (Subtle particles retained for visual depth) ──────────────
rng_s = np.random.default_rng(2024)
for _ in range(130):
    px = float(rng_s.uniform(10, W - 10))
    py = float(rng_s.uniform(10, H - 10))
    pr = float(rng_s.uniform(0.4, 2.2))
    po = float(rng_s.uniform(0.04, 0.28))
    pc = PALETTE[int(rng_s.integers(0, len(PALETTE)))]
    out.append(
        f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="{pr:.1f}"'
        f' fill="{pc}" opacity="{po:.2f}"/>'
    )

# ── ENSEMBLE CHAINS ────────────────────────────────────────────────────────
N_CONF = 20
ensemble = []
for i in range(N_CONF):
    # 'wiggliness=1.25' to force coiling and eliminate straight lines
    x, y   = generate_idp(n=75, spread=225, cx=CX, cy=CY, seed=i * 37 + 11, wiggliness=1.25)
    xs, ys = smooth_spline(x, y, n_out=500, s_factor=1.7)
    ensemble.append((xs, ys))

for i, (xs, ys) in enumerate(ensemble):
    color  = PALETTE[i % len(PALETTE)]
    # Opacity and width optimized for visibility against white background
    alpha  = round(0.25 + 0.10 * (i % 4), 2)
    width  = round(3.0 + 1.5 * (i % 3), 1)
    d = to_path(xs, ys, step=2)
    glow_attr = ' filter="url(#glow-xs)"' if (i % 5 == 0) else ""
    out.append(
        f'    <path d="{d}" fill="none" stroke="{color}"'
        f' stroke-width="{width}" stroke-opacity="{alpha}"'
        f' stroke-linecap="round"{glow_attr}/>'
    )

# ── HERO CHAIN ─────────────────────────────────────────────────────────────
# Retained original seed for the hero chain so the binding spheres stay mapped correctly
x_h, y_h   = generate_idp(n=75, spread=200, cx=CX, cy=CY + 8, seed=42, wiggliness=1.1)
xs_h, ys_h = smooth_spline(x_h, y_h, n_out=600, s_factor=1.5)
d_h = to_path(xs_h, ys_h, step=1)

# Layer 1: outer aura
out.append(
    f'    <path d="{d_h}" fill="none" stroke="{HERO_COLOR}"'
    f' stroke-width="35" stroke-opacity="0.15" stroke-linecap="round"'
    f' filter="url(#glow-lg)"/>'
)
# Layer 2: inner thick glow
out.append(
    f'    <path d="{d_h}" fill="none" stroke="{HERO_COLOR}"'
    f' stroke-width="14" stroke-opacity="0.4" stroke-linecap="round"'
    f' filter="url(#glow-md)"/>'
)
# Layer 3: crisp heavy line
out.append(
    f'    <path d="{d_h}" fill="none" stroke="{HERO_COLOR}"'
    f' stroke-width="6.5" stroke-opacity="1.0" stroke-linecap="round"/>'
)

# ── BINDING RESIDUES on hero chain ─────────────────────────────────────────
N_PTS  = len(xs_h)
BIND_S = int(0.32 * N_PTS)
BIND_E = int(0.64 * N_PTS)
STEP_B = 28          # sample spacing

for idx in range(BIND_S, BIND_E + 1, STEP_B):
    if idx >= N_PTS:
        break
    bx, by = float(xs_h[idx]), float(ys_h[idx])
    # Outer halo (larger)
    out.append(
        f'    <circle cx="{bx:.1f}" cy="{by:.1f}" r="32"'
        f' fill="{BIND_COLOR}" opacity="0.15" filter="url(#glow-lg)"/>'
    )
    # 3D sphere body (larger)
    out.append(
        f'    <circle cx="{bx:.1f}" cy="{by:.1f}" r="14"'
        f' fill="url(#gold-sphere)" filter="url(#glow-md)"/>'
    )
    # Specular highlight
    out.append(
        f'    <circle cx="{bx - 4.5:.1f}" cy="{by - 4.5:.1f}" r="4.5"'
        f' fill="white" opacity="0.85"/>'
    )

# ── N-TERMINUS sphere ──────────────────────────────────────────────────────
nx, ny = float(xs_h[0]), float(ys_h[0])
out.append(
    f'    <circle cx="{nx:.1f}" cy="{ny:.1f}" r="18"'
    f' fill="url(#n-sphere)" filter="url(#glow-md)"/>'
)
out.append(
    f'    <circle cx="{nx - 5:.1f}" cy="{ny - 5:.1f}" r="6"'
    f' fill="white" opacity="0.8"/>'
)
# N label removed

# ── C-TERMINUS sphere ──────────────────────────────────────────────────────
cx_t, cy_t = float(xs_h[-1]), float(ys_h[-1])
out.append(
    f'    <circle cx="{cx_t:.1f}" cy="{cy_t:.1f}" r="18"'
    f' fill="url(#c-sphere)" filter="url(#glow-md)"/>'
)
out.append(
    f'    <circle cx="{cx_t - 5:.1f}" cy="{cy_t - 5:.1f}" r="6"'
    f' fill="white" opacity="0.8"/>'
)
# C label removed

# ── BINDING REGION ANNOTATION removed ───
# mid_idx and direction logic removed
# dotted line and annotation text removed

# ── LEGEND removed ───

out.append('  </g>')
out.append('</svg>')

# ── Write output ───────────────────────────────────────────────────────────
svg_text = "\n".join(out)
out_path = "ensemble_unannotated.svg"
with open(out_path, "w") as f:
    f.write(svg_text)

print(f"✓ SVG written → {out_path}")
print(f"  Canvas : {W} × {H} px")
print(f"  Chains : {N_CONF} ensemble + 1 hero")
print(f"  File   : {len(svg_text) // 1024} KB")