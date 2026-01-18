import streamlit as st
import requests
from PIL import Image
from io import BytesIO

API = "http://localhost:8000"

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Fashion Context Search",
    page_icon="🔍",
    layout="wide",
)

# ---------------- Styles ----------------
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0b1220 0%, #030712 60%);
    color: white;
}

.status-pill {
    background: linear-gradient(90deg, #14532d, #16a34a);
    padding: 10px 18px;
    border-radius: 999px;
    font-weight: 600;
    width: fit-content;
    margin-bottom: 1.2rem;
}

.header {
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}

.subheader {
    color: #9ca3af;
    font-size: 1rem;
    margin-bottom: 1.6rem;
}

.search-box input {
    background: #020617 !important;
    border-radius: 14px !important;
    border: 1px solid #1f2937 !important;
    font-size: 1.05rem !important;
    padding: 14px !important;
}

.result-card {
    background: linear-gradient(180deg, #0b1220, #020617);
    border-radius: 18px;
    padding: 16px 18px;
    margin-bottom: 18px;
    border: 1px solid #1f2937;
    box-shadow: 0 0 25px rgba(0,0,0,0.35);
}

.result-row {
    display: flex;
    gap: 18px;
}

.thumb {
    width: 110px;
    height: 110px;
    border-radius: 14px;
    object-fit: cover;
    border: 1px solid #1f2937;
}

.result-title {
    font-weight: 700;
    font-size: 1.05rem;
}

.result-desc {
    color: #9ca3af;
    font-size: 0.9rem;
    margin-top: 2px;
    margin-bottom: 10px;
}

.tag {
    display: inline-block;
    background: #111827;
    border: 1px solid #1f2937;
    color: #e5e7eb;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 0.75rem;
    margin-right: 6px;
    margin-top: 4px;
}

.match-wrap {
    margin-top: 10px;
}

.match-text {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-bottom: 4px;
}

.match-bar {
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e, #3b82f6, #a855f7);
}

.prompt-chip {
    display: inline-block;
    background: linear-gradient(180deg, #020617, #020617);
    border: 1px solid #1f2937;
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: #d1d5db;
    cursor: pointer;
    margin-right: 8px;
    margin-bottom: 10px;
    transition: all 0.15s ease;
}
.prompt-chip:hover {
    background: #020617;
    border-color: #3b82f6;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Backend Status ----------------
def backend_alive():
    try:
        return requests.get(f"{API}/health", timeout=2).status_code == 200
    except:
        return False


if not backend_alive():
    st.error("🔴 Backend Offline — start FastAPI first")
    st.stop()

st.markdown('<div class="status-pill">🟢 Backend Connected</div>', unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<div class="header">🔍 Fashion Context Search</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subheader">Multimodal fashion & environment retrieval using CLIP + FAISS + SCHP Human Parsing + YOLO + LLM Query Understanding</div>',
    unsafe_allow_html=True
)

# ---------------- Sample Prompts ----------------
st.markdown("### ✨ Try these")
sample_prompts = [
    "Model walking on a runway wearing a black outfit",
    "Professional business attire inside a modern office",
    "Casual weekend outfit for a city walk",
    "Winter coat worn outdoors in snowy environment",
    "Blue shirt sitting on a park bench",
    "Streetwear hoodie in urban night setting",
]

cols = st.columns(3)
for i, prompt in enumerate(sample_prompts):
    if cols[i % 3].button(prompt, key=f"prompt_{i}"):
        st.session_state.query = prompt

# ---------------- Search Input ----------------
query = st.text_input(
    "Describe what you're looking for",
    placeholder="e.g. black outfit in office environment, winter coat, runway model...",
    key="query",
)

top_k = st.slider("Results", 4, 20, 8)

search_clicked = st.button("🔍 Search", use_container_width=True)

# ---------------- Search Logic ----------------
if search_clicked and query.strip():
    with st.spinner("Searching images..."):
        payload = {
            "query": query,
            "top_k": top_k,
            "candidate_k": max(40, top_k * 5),
            "use_llm": True,
        }

        try:
            r = requests.post(f"{API}/api/search", json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    results = data.get("results", [])

    st.markdown(f"## 🖼 Results ({len(results)})")

    for r in results:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1.3, 7])

        # ---------- Image ----------
        with col1:
            if r.get("image_path"):
                try:
                    img = Image.open(BytesIO(requests.get(r["image_path"], timeout=5).content))
                    st.image(img, use_column_width=True)
                except:
                    st.image("https://via.placeholder.com/150?text=No+Image")
            else:
                st.image("https://via.placeholder.com/150?text=No+Image")

        # ---------- Content ----------
        with col2:
            st.markdown(f"<div class='result-title'>{r['image_id']}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='result-desc'>{r.get('explanation','')}</div>",
                unsafe_allow_html=True,
            )

            tags = []
            if r.get("clothing_type"):
                tags.append(r["clothing_type"])
            tags += r.get("colors", [])
            if r.get("environment"):
                tags.append(r["environment"])
            if r.get("vibe"):
                tags.append(r["vibe"])

            for t in tags:
                st.markdown(f"<span class='tag'>{t}</span>", unsafe_allow_html=True)

            score = int(r.get("relevance_score", 0) * 100)
            st.markdown(
                f"""
                <div class="match-wrap">
                    <div class="match-text">Match {score}%</div>
                    <div class="match-bar" style="width:{score}%;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
