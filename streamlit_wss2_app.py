import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import math
from scipy.signal import correlate
import time

# ---- パラメータ ----
mu = 0.0035
pixel_size_m = 1e-4
frame_rate = 30.0

# ---- キャッシュ付き読み込み ----
@st.cache_data
def load_frames(video_bytes, skip=3, max_frames=100):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video_bytes.read())
    tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and len(frames) >= max_frames):
            break
        if idx % skip == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames

# ---- マスク抽出 ----
def red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 70, 50])
    upper2 = np.array([180, 255, 255])
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    return (m1 | m2) > 0

def blue_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([100, 100, 50])
    upper = np.array([140, 255, 255])
    m = cv2.inRange(hsv, lower, upper)
    return m > 0

# ---- WSS / Pressure ----
def compute_wss(frames, resize_scale=0.5):
    gray = [
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (0, 0), fx=resize_scale, fy=resize_scale)
        for f in frames
    ]
    wss_list = []
    for i in range(len(gray) - 1):
        flow = cv2.calcOpticalFlowFarneback(gray[i], gray[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        du = cv2.Sobel(flow[..., 0], cv2.CV_32F, 1, 0, 3)
        dv = cv2.Sobel(flow[..., 1], cv2.CV_32F, 0, 1, 3)
        wss = mu * np.sqrt(du**2 + dv**2) / pixel_size_m
        wss_list.append(np.nanmean(wss))
    return np.array(wss_list)

def compute_pressure(frames, vmax=50.0):
    reds = []
    for frame in frames:
        mask = red_mask(frame)
        if mask.any():
            reds.append(frame[..., 0][mask].mean())
        else:
            reds.append(np.nan)
    M = max([r for r in reds if not np.isnan(r)], default=1)
    pressures = [(r / M) * vmax * np.pi * (0.25 ** 2) if not np.isnan(r) else np.nan for r in reds]
    return np.array(pressures)

# ---- 特徴量 ----
def detect_peaks(series):
    arr = np.array(series)
    peaks = []
    for i in range(1, len(arr) - 1):
        if not math.isnan(arr[i]) and arr[i] >= arr[i-1] and arr[i] >= arr[i+1]:
            peaks.append(i)
    return peaks

def compute_features(pressure, wss, time):
    valid = ~np.isnan(pressure) & ~np.isnan(wss)
    p, w, t = pressure[valid], wss[valid], time[valid]
    if len(p) < 3:
        return {'corr': np.nan, 'lag': np.nan, 'sim_peaks': 0}
    corr = np.corrcoef(p, w)[0,1]
    cc = correlate(p - np.mean(p), w - np.mean(w), mode='full')
    lag_idx = np.argmax(cc) - (len(p)-1)
    dt = t[1] - t[0] if len(t) > 1 else 0
    lag = lag_idx * dt
    peaks_w = detect_peaks(w)
    peaks_p = detect_peaks(p)
    sim = 0
    for pw in peaks_w:
        for pp in peaks_p:
            if abs(pw-pp) <= 1:
                sim += 1
                break
    return {'corr': corr, 'lag': lag, 'sim_peaks': sim}

# ---- ベクトル描画（代表フレームのみ） ----
def draw_vectors(base, next_frame, mask, color=(255,0,0), threshold=0.5, step=15):
    img = base.copy()
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray, gray_next, None, 0.5,3,15,3,5,1.2,0)
    h, w = flow.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            if not mask[y, x]:
                continue
            fx, fy = flow[y, x]
            mag = np.hypot(fx, fy)
            if mag < threshold:
                continue
            pt1 = (x, y)
            pt2 = (int(x + fx * 5), int(y + fy * 5))
            cv2.arrowedLine(img, pt1, pt2, color, 1, tipLength=0.3)
    return img

# ---- UI セットアップ ----
st.set_page_config(page_title="簡易 Flow Analyzer", layout="wide")
st.title("軽量版 赤・青マスク + WSS/Pressure 解析")

# 入力
video = st.file_uploader("MP4動画をアップロード", type="mp4")
vmax = st.slider("速度レンジ（cm/s）", 10.0, 100.0, 50.0)
skip = st.sidebar.slider("間引き (skip)", 1, 6, 3)
resize_scale = st.sidebar.slider("WSS縮小率", 0.3, 1.0, 0.5, step=0.1)
arrow_threshold = st.sidebar.slider("矢印しきい値", 0.0, 3.0, 0.5, step=0.1)
arrow_step = st.sidebar.slider("矢印間隔", 5, 25, 15)

if video:
    st.video(video)
    if st.button("解析実行"):
        t0 = time.time()
        frames = load_frames(video, skip=skip, max_frames=120)
        if len(frames) < 2:
            st.error("フレームが足りません")
            st.stop()

        # WSS / Pressure
        mean_wss = compute_wss(frames, resize_scale=resize_scale)
        pressures = compute_pressure(frames, vmax=vmax)
        time_axis = np.arange(len(mean_wss)) / frame_rate * skip

        # 特徴量
        feat = compute_features(pressures[:len(mean_wss)], mean_wss, time_axis)

        # 代表フレーム（WSS最大）
        idx_wss = int(np.nanargmax(mean_wss)) if len(mean_wss) > 0 else 0
        base = frames[idx_wss]
        next_frame = frames[idx_wss+1] if idx_wss+1 < len(frames) else base

        # マスク
        r_mask = red_mask(base)
        b_mask = blue_mask(base)

        # 矢印付き画像
        arrow_red = draw_vectors(base, next_frame, r_mask, color=(255,0,0),
                                 threshold=arrow_threshold, step=arrow_step)
        arrow_blue = draw_vectors(base, next_frame, b_mask, color=(0,0,255),
                                  threshold=arrow_threshold, step=arrow_step)
        combined_mask = r_mask | b_mask
        arrow_combined = draw_vectors(base, next_frame, combined_mask, color=(0,255,0),
                                      threshold=arrow_threshold, step=arrow_step)

        # プロット
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            ax1.plot(time_axis, mean_wss, label="WSS", color="orange")
            ax1.set_title("WSS Trend")
            ax1.set_xlabel("Time (s)")
            ax1.legend()
            st.pyplot(fig1)
        with col2:
            fig2, ax2 = plt.subplots()
            ax2.plot(time_axis, pressures[:len(mean_wss)], label="Pressure", color="blue")
            ax2.set_title("Pressure Trend")
            ax2.set_xlabel("Time (s)")
            ax2.legend()
            st.pyplot(fig2)

        st.subheader("代表フレームの流れ（矢印）")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(arrow_red, caption="赤領域流れ", use_column_width=True)
        with c2:
            st.image(arrow_blue, caption="青領域流れ", use_column_width=True)
        with c3:
            st.image(arrow_combined, caption="合成流れ", use_column_width=True)

        st.subheader("特徴量")
        st.markdown(f"- Correlation (WSS vs Pressure): {feat['corr']:.2f}")
        st.markdown(f"- Lag time: {feat['lag']:.2f} 秒")
        st.markdown(f"- Simultaneous peaks: {feat['sim_peaks']} 回")

        # 簡易示唆
        corr_val = feat['corr']
        lag = feat['lag']
        sim = feat['sim_peaks']
        if sim >= 80 and abs(lag) >= 2.0:
            suggestion = "高度狭窄疑い"
        elif sim >= 50 or abs(lag) >= 0.8 or abs(corr_val) >= 0.3:
            suggestion = "軽度〜中等度狭窄疑い"
        else:
            suggestion = "狭窄なし寄り"
        st.markdown(f"### 示唆: {suggestion}")
        st.markdown(f"- 同時ピーク数: {sim}, ラグ: {lag:.2f}, 相関: {corr_val:.2f}")

        # CSV ダウンロード
        df = pd.DataFrame({
            "Frame": np.arange(len(mean_wss)),
            "Time (s)": time_axis,
            "WSS": mean_wss,
            "Pressure": pressures[:len(mean_wss)]
        })
        st.download_button("CSV 保存", data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="simple_flow.csv", mime="text/csv")

        st.success(f"完了 (処理時間 {time.time()-t0:.2f}s)")
