import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from mplsoccer import Pitch
from tqdm import tqdm

# ── CONFIG ─────────────────────────────────────────────────────────────────────
PITCH_W, PITCH_H = 105, 68
FRAME_RATE       = 10
HOME_COLOR       = "#00A8E0"
AWAY_COLOR       = "#E05200"
BG_COLOR         = "white"
LINE_COLOR       = "#aaaaaa"
ALPHA_FILL       = 0.35
DPI              = 72
ARROW_SCALE      = 1.5   # tune this: larger = longer arrows


# ── LOAD MATCH ─────────────────────────────────────────────────────────────────
def load_match(match_id):
    cols = [
        "match_id", "frame", "player_id", "x", "y",
        "ball_x", "ball_y", "possession_group",
        "team_id", "vx_kmh", "vy_kmh",
        "end_type", "event_id", "frame_end", "frame_start"
    ]
    chunks = pd.read_csv('../data/full_data.csv', usecols=cols,
                         chunksize=500_000, low_memory=False)
    data = pd.concat(
        [c[c["match_id"] == match_id] for c in tqdm(chunks, desc="Loading CSV")],
        ignore_index=True
    )
    home_id = data.loc[data["possession_group"] == "home team", "team_id"].dropna()
    if home_id.empty:
        home_id = data["team_id"].dropna().mode()
    data["team_id_plot"] = (data["team_id"] == home_id.iloc[0]).astype(int)
    return data


# ── INTERPOLATE MISSING FRAMES ─────────────────────────────────────────────────
def interpolate_frames(clip):
    """Fill in missing frames by linearly interpolating player positions,
    then recompute velocity from the interpolated positions."""
    clip = clip.drop_duplicates(subset=["player_id", "frame"], keep="first")
    all_frames  = np.arange(clip["frame"].min(), clip["frame"].max() + 1)
    interp_rows = []

    non_num_cols = [c for c in clip.select_dtypes(exclude=np.number).columns
                    if c not in ["player_id", "frame"]]

    for pid in clip["player_id"].unique():
        pdf      = clip[clip["player_id"] == pid].set_index("frame").sort_index()
        num_cols = pdf.select_dtypes(include=np.number).columns
        pdf_i    = pdf[num_cols].reindex(all_frames).interpolate(method="index").reset_index()
        pdf_i.rename(columns={"index": "frame"}, inplace=True)

        # Recompute velocity from interpolated x/y instead of interpolating vx/vy
        pdf_i["vx_f"] = pdf_i["x"].diff().fillna(0)   # m/frame
        pdf_i["vy_f"] = pdf_i["y"].diff().fillna(0)

        for col in non_num_cols:
            if col in pdf.columns:
                pdf_i[col] = pdf[col].reindex(all_frames).ffill().bfill().values

        pdf_i["player_id"] = pid
        interp_rows.append(pdf_i)

    return pd.concat(interp_rows, ignore_index=True)


# ── CLIPPED VORONOI ────────────────────────────────────────────────────────────
def clipped_voronoi(points):
    x0, x1 = -PITCH_W/2, PITCH_W/2
    y0, y1 = -PITCH_H/2, PITCH_H/2
    pitch_poly = Polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])
    pad = max(PITCH_W, PITCH_H) * 2
    mirror = np.vstack([
        points,
        points + [ pad,  0  ], points + [-pad,  0  ],
        points + [ 0,    pad], points + [ 0,   -pad],
        points + [ pad,  pad], points + [-pad,  pad],
        points + [ pad, -pad], points + [-pad, -pad],
    ])
    vor = Voronoi(mirror)
    polys = []
    for i in range(len(points)):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            polys.append(None); continue
        poly = Polygon(vor.vertices[region]).intersection(pitch_poly)
        polys.append(poly if not poly.is_empty else None)
    return polys


# ── MAKE VORONOI CLIP ──────────────────────────────────────────────────────────
def make_voronoi_clip(
    data: pd.DataFrame,
    match_id: int,
    start_frame: int,
    end_frame: int,
    out_path: str = None,
    frame_step: int = 1,
):
    if out_path is None:
        out_path = f"voronoi_{match_id}_{start_frame}.gif"

    clip = data[
        (data["match_id"] == match_id) &
        (data["frame"] >= start_frame) &
        (data["frame"] <= end_frame)
    ].copy()

    if clip.empty:
        raise ValueError(f"No data for match_id={match_id}, frames {start_frame}–{end_frame}.")

    clip = interpolate_frames(clip)

    if "team_id_plot" not in clip.columns:
        home_id = clip.loc[clip["possession_group"] == "home team", "team_id"].dropna()
        if home_id.empty:
            raise ValueError("Could not determine home team")
        clip["team_id_plot"] = (clip["team_id"] == home_id.iloc[0]).astype(int)

    # vx_f/vy_f already computed in interpolate_frames (m/frame)
    # if not present (no interpolation), compute from vx_kmh
    if "vx_f" not in clip.columns:
        clip["vx_f"] = clip["vx_kmh"] / 3.6
        clip["vy_f"] = clip["vy_kmh"] / 3.6

    color_map = {1: HOME_COLOR, 0: AWAY_COLOR}
    frames    = sorted(clip["frame"].unique())[::frame_step]

    # ── Pre-compute Voronoi ────────────────────────────────────────────────────
    frame_data = {}
    for fn in tqdm(frames, desc="Pre-computing Voronoi"):
        fdf = clip[clip["frame"] == fn].drop_duplicates("player_id", keep="last")
        fdf = fdf[fdf["x"].notna() & fdf["y"].notna()]
        if len(fdf) < 4:
            continue

        def_fdf, att_fdf = fdf.copy(), pd.DataFrame()
        def_color, att_color = HOME_COLOR, AWAY_COLOR

        pg = fdf["possession_group"].dropna()
        if not pg.empty:
            g = pg.iloc[0]
            if g == "home team":
                poss_tid = 1
            elif g == "away team":
                poss_tid = 0
            else:
                poss_tid = None
            if poss_tid is not None:
                att_fdf   = fdf[fdf["team_id_plot"] == poss_tid]
                def_fdf   = fdf[fdf["team_id_plot"] != poss_tid]
                att_color = color_map[poss_tid]
                def_color = color_map[1 - poss_tid]

        all_pts    = fdf[["x","y"]].values
        def_ids    = set(def_fdf["player_id"].values)
        player_ids = fdf["player_id"].values
        polys      = clipped_voronoi(all_pts)

        cells = []
        for j, poly in enumerate(polys):
            if poly is None or poly.is_empty or player_ids[j] not in def_ids:
                continue
            try:
                xs, ys = poly.exterior.xy
                cells.append((list(xs), list(ys), def_color))
            except Exception:
                pass

        frame_data[fn] = {
            "cells":     cells,
            "def":       def_fdf[["x","y","vx_f","vy_f"]].values,
            "att":       att_fdf[["x","y","vx_f","vy_f"]].values if not att_fdf.empty else np.empty((0,4)),
            "def_color": def_color,
            "att_color": att_color,
            "ball":      fdf[["ball_x","ball_y"]].dropna().values,
        }

    valid_frames = [fn for fn in frames if fn in frame_data]
    if not valid_frames:
        raise ValueError("No valid frames — all frames had fewer than 4 detected players.")

    # ── Draw pitch once ────────────────────────────────────────────────────────
    pitch = Pitch(pitch_type="skillcorner", pitch_color=BG_COLOR,
                  line_color=LINE_COLOR, pitch_length=PITCH_W, pitch_width=PITCH_H)
    fig, ax = pitch.draw(figsize=(10, 6.5))
    fig.set_facecolor(BG_COLOR)

    added = []

    def animate(i):
        nonlocal added
        for a in added:
            try: a.remove()
            except: pass
        added.clear()

        fn = valid_frames[i]
        fd = frame_data[fn]

        for xs, ys, c in fd["cells"]:
            added += ax.fill(xs, ys, color=c, alpha=ALPHA_FILL, zorder=2)
            added += ax.plot(xs, ys, color=c, lw=0.8, alpha=0.8, zorder=3)

        dc = fd["def_color"]
        if len(fd["def"]):
            added += ax.plot(fd["def"][:,0], fd["def"][:,1], "o", color=dc,
                             ms=8, markeredgecolor="black", markeredgewidth=0.5, zorder=5)
            for row in fd["def"]:
                dx = row[2] * ARROW_SCALE
                dy = row[3] * ARROW_SCALE
                if np.sqrt(dx**2 + dy**2) > 0.1:
                    added.append(ax.annotate("",
                        xy=(row[0]+dx, row[1]+dy), xytext=(row[0], row[1]),
                        arrowprops=dict(arrowstyle="-|>", color=dc, lw=1.5), zorder=6))

        ac = fd["att_color"]
        if len(fd["att"]):
            added += ax.plot(fd["att"][:,0], fd["att"][:,1], "o", color=ac,
                             ms=8, markeredgecolor="black", markeredgewidth=0.5, zorder=5)
            for row in fd["att"]:
                dx = row[2] * ARROW_SCALE
                dy = row[3] * ARROW_SCALE
                if np.sqrt(dx**2 + dy**2) > 0.1:
                    added.append(ax.annotate("",
                        xy=(row[0]+dx, row[1]+dy), xytext=(row[0], row[1]),
                        arrowprops=dict(arrowstyle="-|>", color=ac, lw=1.5), zorder=6))

        if len(fd["ball"]):
            added += ax.plot(fd["ball"][0,0], fd["ball"][0,1], "o",
                             color="black", ms=7, markeredgecolor="#555",
                             markeredgewidth=0.5, zorder=7)

        label = f"Frame {fn}" + (" ← SHOT" if fn == end_frame else "")
        added.append(ax.set_title(label, color="red" if fn == end_frame else "#555555",
                                  fontsize=9, pad=4))
        return added

    anim = animation.FuncAnimation(
        fig, animate, frames=len(valid_frames),
        interval=1000/FRAME_RATE, blit=False, repeat=False
    )
    anim.save(out_path, writer="pillow", dpi=DPI,
              savefig_kwargs={"facecolor": BG_COLOR, "pad_inches": 0},
              progress_callback=lambda i, n: print(f"\rRendering {i+1}/{n}", end=""))
    plt.close(fig)
    print(f"\nDone! → {out_path}")
    return out_path


# ── SHOT CLIPS ─────────────────────────────────────────────────────────────────
def make_all_shot_clips(match_id, pre_sec=9, post_sec=3):
    match_data = load_match(match_id)

    shot_events = (
        match_data[match_data["end_type"] == "shot"]
        .drop_duplicates(subset=["event_id"])[["event_id", "frame_end"]]
        .dropna()
    )

    if shot_events.empty:
        print(f"No shot events found for match {match_id}")
        return

    print(f"Found {len(shot_events)} shots for match {match_id}")

    for _, event in shot_events.iterrows():
        shot_frame  = int(event["frame_end"])
        start_frame = max(0, shot_frame - int(pre_sec * FRAME_RATE))
        end_frame   = shot_frame + int(post_sec * FRAME_RATE)
        out_path    = f"voronoi_{match_id}_shot_{shot_frame}.gif"

        print(f"\nShot at frame {shot_frame} → {start_frame}–{end_frame}")
        try:
            make_voronoi_clip(match_data, match_id, start_frame, end_frame,
                              out_path=out_path)
        except Exception as e:
            print(f"  ✗ skipped: {e}")


# ── USAGE ──────────────────────────────────────────────────────────────────────
make_all_shot_clips(1996435)