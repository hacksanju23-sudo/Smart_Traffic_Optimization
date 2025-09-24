%%writefile app.py
import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import tempfile

# ======= PAGE STYLE =======
st.set_page_config(page_title="🚦 Smart Traffic Optimization", page_icon="🚦", layout="wide")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: url("OIP (1) (1).jpg") no-repeat center center fixed;
    background-size: cover;
    color: white;
}
h1, h2, h3, h4 {
    font-family: 'Arial Black', sans-serif;
    text-shadow: 2px 2px 8px black;
    color: gold;
}
.stButton>button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    border: none;
    padding: 12px 28px;
    box-shadow: 0px 0px 15px rgba(255,215,0,0.7);
}
.stButton>button:hover {
    background: linear-gradient(90deg, #24c6dc, #514a9d);
}
.block-container {
    padding-top: 1rem;
}
.glass-card {
    background: rgba(255, 255, 255, 0.15);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 0px 20px gold;
    backdrop-filter: blur(10px);
}
.metric-card {
    background: rgba(0,0,0,0.6);
    border-radius: 15px;
    padding: 15px;
    text-align: center;
    box-shadow: 0px 0px 10px #FFD700;
}
.banner {
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 0px 20px rgba(255,215,0,0.7);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ======= HERO BANNER =======
st.markdown(
    """
    <div class="banner" style="text-align:center;">
        <h1 style="font-size:60px;">
            🟥 🟨 🟩  Smart Traffic Optimization Dashboard 🚗🚌🚙
        </h1>
        <h3 style="color:#FFD700;">AI-powered Vehicle Detection & Intelligent Signal Control</h3>
        <p style="font-size:18px; color:white;">
            Real-time dashboard for monitoring vehicles, optimizing traffic signals,
            and reducing congestion using AI 🚦
        </p>
    </div>
    """, unsafe_allow_html=True
)

# ======= LOAD MODEL =======
model = YOLO("best.pt")

# ======= LAYOUT =======
left, right = st.columns([1,2])

with left:
    st.markdown("### 📤 Upload a traffic video to analyze")
    uploaded_file = st.file_uploader("Choose a video", type=["mp4", "mov", "avi"])

with right:
    st.markdown("### 📺 Live Video Feed with Detection")

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    frame_counts = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.markdown("### 🔍 Analyzing video... please wait")
    stframe = st.empty()
    progress = st.progress(0)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, conf=0.4)
        counts = {}
        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            counts[name] = counts.get(name, 0) + 1
        frame_counts.append(counts)

        annotated = results[0].plot()
        stframe.image(annotated, channels="BGR", caption=f"📹 Frame {i+1}/{total_frames}")

        progress.progress((i+1)/total_frames)

    cap.release()

    # ======= SUMMARY =======
    df = pd.DataFrame(frame_counts).fillna(0).astype(int)
    st.markdown("<h2 style='color:#FFD700;'>📊 Dashboard Metrics</h2>", unsafe_allow_html=True)

    totals = df.sum().to_dict()

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"<div class='metric-card'><h3>🚗 Cars</h3><h2>{totals.get('car',0)}</h2></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><h3>🚌 Buses</h3><h2>{totals.get('bus',0)}</h2></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><h3>🏍️ Bikes</h3><h2>{totals.get('motorcycle',0)}</h2></div>", unsafe_allow_html=True)

    total = df.sum().sum()

    # ======= DECISION =======
    if total > 50:
        action = "🟢 Extend Green by 10s"
        light = "https://upload.wikimedia.org/wikipedia/commons/1/1b/Traffic_light_green.png"
    elif "bus" in df.columns and df["bus"].sum() > 5:
        action = "🚌 Priority Green for Bus Lane"
        light = "https://upload.wikimedia.org/wikipedia/commons/8/89/Traffic_light_green.png"
    else:
        action = "🟡 Normal Cycle"
        light = "https://upload.wikimedia.org/wikipedia/commons/5/5f/Traffic_light_yellow.png"

    st.markdown(f"""
    <div class="glass-card">
        <h2 style="color:#FFD700;">🚥 Traffic Light Decision</h2>
        <h3 style="color:#7CFC00;">{action}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.image(light, caption="Dynamic Traffic Signal", width=150)

    # ======= DOWNLOAD CSV =======
    csv_path = "vehicle_counts.csv"
    df.to_csv(csv_path, index=False)
    with open(csv_path, "rb") as f:
        st.download_button("📥 Download Vehicle Counts CSV", f, "vehicle_counts.csv")
