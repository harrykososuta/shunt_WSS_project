import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import math
from scipy.signal import correlate
import time

# ---------- パラメータ初期値 ----------
mu = 0.0035
pixel_size_m = 1e-4
resize_scale = 0.5  # 表示/演算前の縮小率
frame_rate = 30.0  # 元動画のfps想定

# ---------- キャッシュ付きヘルパー ----------
@st.cache_data(show_spinner=False)
def load_and_sample_frames(video_bytes, skip=3, max_frames=120):
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

@st.cache_data(show_spinner=False)
def compute_wss_maps_cached(frames, resize_scale_local):
    gray = [
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY),
                   (0, 0), fx=resize_scale_local, fy=resize_scale_local)
        for f in frames
    ]
    wss_maps = []
    for i in range(len(gray) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray[i], gray[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        du = cv2.Sobel(flow[..., 0], cv2.CV_32F, 1, 0, 3)
        dv = cv2.Sobel(flow[..., 1], cv2.CV_32F, 0, 1, 3)
        wss = mu * np.sqrt(du ** 2 + dv ** 2) / pixel_size_m
        wss_maps.append(wss)
    return wss_maps

# ---------- マスク/特徴量/描画関数 ----------
def extract_red_mask_dynamic(img, l1, u1, l2, u2):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, np.array(l1), np.array(u1))
    m2 = cv2.inRange(hsv, np.array(l2), np.array(u2))
    return (m1 | m2) > 0

def extract_blue_mask_dynamic(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, np.array(lower), np.array(upper))
    return m > 0

def calculate_pressure(frames, red_lower1, red_upper1, red_lower2, red_upper2, vmax):
    reds = []
    for frame in frames:
        mask = extract_red_mask_dynamic(frame, red_lower1, red_upper1, red_lower2, red_upper2)
        if mask.any():
            reds.append(frame[..., 0][mask].mean())
        else:
            reds.append(np.nan)
    M = max([r for r in reds if not np.isnan(r)], default=1)
    pressures = [(r / M) * vmax * np.pi * (0.25 ** 2) if not np.isnan(r) else np.nan for r in reds]
    return pressures

def detect_local_peaks(series):
    data = np.array(series)
    peaks = []
    for i in range(1, len(data) - 1):
        if not math.isnan(data[i]) and data[i] >= data[i - 1] and data[i] >= data[i + 1]:
            peaks.append(i)
    return peaks

def compute_feature_from_trends(pressure, mean_wss, time):
    valid = ~np.isnan(pressure) & ~np.isnan(mean_wss)
    p, w, t = pressure[valid], mean_wss[valid], time[valid]
    if len(p) < 3:
        return {'corr_pressure_wss': np.nan, 'lag_sec_wss_after_pressure': np.nan, 'simultaneous_peak_counts': 0}
    corr = np.corrcoef(p, w)[0, 1]
    cc = correlate(p - np.mean(p), w - np.mean(w), mode='full')
    lag = (np.argmax(cc) - (len(p) - 1)) * (t[1] - t[0] if len(t) > 1 else 0)
    peaks_wss = detect_local_peaks(w)
    peaks_p = detect_local_peaks(p)
    sim = sum(any(abs(pw - pp) <= 1 for pp in peaks_p) for pw in peaks_wss)
    return {'corr_pressure_wss': corr, 'lag_sec_wss_after_pressure': lag, 'simultaneous_peak_counts': sim}

def draw_flow_arrows(base, flow, mask, color, threshold=0.5, step=15):
    img = base.copy()
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

def bullseye_map_highlight(vals, title, cmap='jet'):
    sectors = 12
    arr = np.array(vals)
    if arr.size < sectors:
        arr = np.pad(arr, (0, sectors - arr.size), constant_values=np.nan)
    thr = np.nanmean(arr) + np.nanstd(arr)
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(4, 4))
    width = 2 * np.pi / sectors
    for i, v in enumerate(arr):
        theta = i * width
        if np.isnan(v) or v < thr:
            color = 'white'
        else:
            norm = (v - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-6)
            color = plt.get_cmap(cmap)(norm)
        ax.bar(theta, 0.2, width=width, bottom=0.8, color=color, edgecolor='black', linewidth=0.8)
    ax.set_xticks(np.linspace(0, 2 * np.pi, sectors, endpoint=False))
    ax.set_xticklabels([f"{i*30}°" for i in range(sectors)])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    return fig, arr

def get_high_sectors(arr, label):
    thr = np.nanmean(arr) + np.nanstd(arr)
    idx = np.where(arr >= thr)[0]
    if idx.size:
        degs = ", ".join(f"{i*30}°" for i in idx)
        return f"- **{label} 集中部位**: {degs}"
    return f"- **{label} 集中部位**: なし"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="軽量化版赤青流れ解析", layout="wide")
st.title("軽量化版 Blood Flow / WSS / Pressure Analyzer")

# サイドバー：設定
st.sidebar.header("マスク閾値 (HSV)")
red_h1 = st.sidebar.slider("赤低域 H1", 0, 30, 0)
red_s1 = st.sidebar.slider("赤低域 S1", 0, 255, 70)
red_v1 = st.sidebar.slider("赤低域 V1", 0, 255, 50)
red_h2 = st.sidebar.slider("赤高域 H2", 150, 180, 160)
red_s2 = st.sidebar.slider("赤高域 S2", 0, 255, 70)
red_v2 = st.sidebar.slider("赤高域 V2", 0, 255, 50)

blue_h_lower = st.sidebar.slider("青下限 H", 80, 140, 100)
blue_s_lower = st.sidebar.slider("青下限 S", 50, 255, 100)
blue_v_lower = st.sidebar.slider("青下限 V", 0, 255, 50)
blue_h_upper = st.sidebar.slider("青上限 H", 80, 180, 140)
blue_s_upper = st.sidebar.slider("青上限 S", 50, 255, 255)
blue_v_upper = st.sidebar.slider("青上限 V", 0, 255, 255)

st.sidebar.header("矢印表示調整")
arrow_threshold = st.sidebar.slider("矢印しきい値", 0.0, 5.0, 0.5, step=0.1)
arrow_step = st.sidebar.slider("矢印間隔", 5, 30, 20)

st.sidebar.header("代表フレーム選択")
rep_choice = st.sidebar.selectbox("基準", ["WSS最大", "Pressure最大", "WSS変化量最大"])

st.sidebar.header("パフォーマンス調整")
skip = st.sidebar.slider("フレーム間引き skip", 1, 6, 3)
resize_scale_local = st.sidebar.slider("解析用縮小率 (WSS)", 0.3, 1.0, 0.5, step=0.1)

# 動画アップロード
video = st.file_uploader("動画をアップロード（MP4）", type="mp4")
vmax = st.slider("速度レンジ（cm/s）", 10.0, 120.0, 50.0, step=1.0)

if video:
    st.video(video)
    if st.button("特徴量を計算＆代表フレーム選出"):
        t_start = time.time()
        frames = load_and_sample_frames(video, skip=skip, max_frames=120)
        st.write(f"読み込んだ間引きフレーム数: {len(frames)}")
        if len(frames) < 2:
            st.error("フレーム不足")
            st.stop()

        wss_maps = compute_wss_maps_cached(frames, resize_scale_local)
        pressures = calculate_pressure(frames,
                                       red_lower1=(red_h1, red_s1, red_v1),
                                       red_upper1=(red_h2, red_s2, red_v2),
                                       red_lower2=(red_h1, red_s1, red_v1),
                                       red_upper2=(red_h2, red_s2, red_v2),
                                       vmax=vmax)
        mean_wss = np.array([np.nanmean(w) for w in wss_maps])
        time_axis = np.arange(len(mean_wss)) / frame_rate * skip  # 補正

        feat = compute_feature_from_trends(np.array(pressures[:len(mean_wss)]), mean_wss, time_axis)

        # 代表フレーム決定
        if rep_choice == "WSS最大":
            rep_idx = int(np.nanargmax(mean_wss))
        elif rep_choice == "Pressure最大":
            rep_idx = int(np.nanargmax(pressures[:len(mean_wss)]))
        else:
            diff = np.abs(np.diff(mean_wss, prepend=mean_wss[0]))
            rep_idx = int(np.nanargmax(diff))

        # 光流は代表フレームだけ
        base_frame = frames[rep_idx]
        next_frame = frames[rep_idx + 1] if rep_idx + 1 < len(frames) else base_frame
        gray = cv2.cvtColor(base_frame, cv2.COLOR_RGB2GRAY)
        gray_next = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray, gray_next, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        # マスク
        red_mask = extract_red_mask_dynamic(base_frame,
                                            (red_h1, red_s1, red_v1),
                                            (red_h2, red_s2, red_v2),
                                            (red_h1, red_s1, red_v1),
                                            (red_h2, red_s2, red_v2))
        blue_mask = extract_blue_mask_dynamic(base_frame,
                                              (blue_h_lower, blue_s_lower, blue_v_lower),
                                              (blue_h_upper, blue_s_upper, blue_v_upper))

        # 矢印描画（代表のみ）
        arrow_red = draw_flow_arrows(base_frame, flow, red_mask, color=(255, 0, 0),
                                    threshold=arrow_threshold, step=arrow_step)
        arrow_blue = draw_flow_arrows(base_frame, flow, blue_mask, color=(0, 0, 255),
                                     threshold=arrow_threshold, step=arrow_step)
        combined_mask = red_mask | blue_mask
        arrow_combined = draw_flow_arrows(base_frame, flow, combined_mask,
                                         color=(0, 255, 0), threshold=arrow_threshold, step=arrow_step)

        # 描画：時系列
        fig_w, axw = plt.subplots()
        axw.plot(time_axis, mean_wss, label="WSS", color="tab:orange")
        axw.set_title("WSS Trend")
        axw.set_xlabel("Time (s)")
        axw.legend()
        st.pyplot(fig_w, key="wss_trend")

        fig_p, axp = plt.subplots()
        axp.plot(time_axis, pressures[:len(mean_wss)], label="Pressure", color="tab:blue")
        axp.set_title("Pressure Trend")
        axp.set_xlabel("Time (s)")
        axp.legend()
        st.pyplot(fig_p, key="pressure_trend")

        fig_pw, axpw = plt.subplots()
        axpw.plot(time_axis, pressures[:len(mean_wss)], label="Pressure", color="tab:blue")
        axpw2 = axpw.twinx()
        axpw2.plot(time_axis, mean_wss, label="WSS", linestyle="--", color="tab:orange")
        axpw.set_title("WSS & Pressure")
        axpw.set_xlabel("Time (s)")
        axpw.legend(loc="upper left")
        axpw2.legend(loc="upper right")
        st.pyplot(fig_pw, key="combined_trend")

        # 代表フレーム表示
        st.subheader("代表フレームの流れベクトル")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(arrow_red, caption="赤領域の流れ", use_column_width=True)
        with col2:
            st.image(arrow_blue, caption="青領域の流れ", use_column_width=True)
        with col3:
            st.image(arrow_combined, caption="赤＋青合成流れ", use_column_width=True)

        # Bull's Eye 風
        st.subheader("短軸風分布")
        fig_be_w, arr_w = bullseye_map_highlight(mean_wss[:12], "Bull’s Eye (WSS)", cmap="Blues")
        fig_be_p, arr_p = bullseye_map_highlight(np.array(pressures[:12]), "Bull’s Eye (Pressure)", cmap="Reds")
        b1, b2 = st.columns(2)
        with b1:
            st.pyplot(fig_be_w)
            st.markdown(get_high_sectors(arr_w, "WSS"))
        with b2:
            st.pyplot(fig_be_p)
            st.markdown(get_high_sectors(arr_p, "Pressure"))

        # 特徴量表示
        st.subheader("特徴量 / 狭窄示唆")
        st.markdown(f"- Correlation (WSS vs Pressure): {feat['corr_pressure_wss']:.2f}")
        st.markdown(f"- Lag time: {feat['lag_sec_wss_after_pressure']:.2f} 秒")
        st.markdown(f"- Simultaneous peaks: {feat['simultaneous_peak_counts']} 回")

        wss_peaks = detect_local_peaks(mean_wss)
        p_peaks = detect_local_peaks(np.array(pressures[:len(mean_wss)]))
        if wss_peaks and p_peaks:
            dt = time_axis[1] - time_axis[0] if len(time_axis) > 1 else 0
            delta_sec = (wss_peaks[0] - p_peaks[0]) * dt
            st.markdown(f"- WSS/Pressure 最初ピーク時間差: {delta_sec:.2f} 秒 ({'WSS先行' if delta_sec < 0 else 'WSS遅延'})")
        else:
            st.markdown("- 明確な両方のピークは検出されませんでした。")

        # 簡易示唆
        sim = feat['simultaneous_peak_counts']
        lag = feat['lag_sec_wss_after_pressure']
        corr_val = feat['corr_pressure_wss']
        if sim >= 80 and abs(lag) >= 2.0:
            verdict = "高度狭窄疑い"
        elif sim >= 50 or abs(lag) >= 0.8 or abs(corr_val) >= 0.3:
            verdict = "軽度〜中等度狭窄疑い"
        else:
            verdict = "狭窄なし寄り"
        st.markdown(f"### 示唆: {verdict}")
        st.markdown(f"- 同時ピーク数: {sim}, ラグ: {lag:.2f}, 相関絶対値: {abs(corr_val):.2f}")

        # CSV 出力
        df = pd.DataFrame({
            "Frame": np.arange(len(mean_wss)),
            "Time (s)": time_axis,
            "WSS": mean_wss,
            "Pressure": pressures[:len(mean_wss)]
        })
        st.download_button("結果をCSV保存", data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="flow_summary.csv", mime="text/csv")

        st.success(f"完了 (処理時間: {time.time()-t_start:.2f}s)")
