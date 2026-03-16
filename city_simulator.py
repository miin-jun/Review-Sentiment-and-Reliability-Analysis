import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import pipeline

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="🏙️ 도시 성장 시뮬레이터",
    page_icon="🏙️",
    layout="wide",
)

# ─────────────────────────────────────────
# CSS 스타일
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
    }

    .main { background-color: #0f1117; }

    .title-block {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .title-block h1 {
        font-size: 2.8rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
    }
    .title-block p {
        color: #888;
        font-size: 1rem;
    }

    .event-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #2e3650;
        border-left: 4px solid #4ade80;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin: 1rem 0;
        animation: fadeIn 0.4s ease;
    }
    .event-card .label {
        font-size: 0.75rem;
        color: #4ade80;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 700;
    }
    .event-card .message {
        font-size: 1.1rem;
        color: #fff;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    .event-card .score-info {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.3rem;
    }

    .stat-card {
        background: #1a1f2e;
        border: 1px solid #2e3650;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .stat-card .stat-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-card .stat-value {
        font-size: 1.6rem;
        font-weight: 900;
        color: #4ade80;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .stTextInput > div > div > input {
        background-color: #1a1f2e !important;
        color: #fff !important;
        border: 1px solid #2e3650 !important;
        border-radius: 8px !important;
        font-family: 'Noto Sans KR', sans-serif !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4ade80, #22c55e) !important;
        color: #0f1117 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-family: 'Noto Sans KR', sans-serif !important;
        transition: transform 0.1s;
    }
    .stButton > button:hover { transform: scale(1.03); }

    div[data-testid="stHorizontalBlock"] > div { padding: 0 0.3rem; }

    .history-item {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.4rem 0;
        border-bottom: 1px solid #1e2435;
        font-size: 0.88rem;
        color: #ccc;
    }
    .history-badge {
        background: #1e2d1f;
        color: #4ade80;
        border-radius: 6px;
        padding: 0.1rem 0.5rem;
        font-size: 0.78rem;
        font-weight: 700;
        white-space: nowrap;
    }
    .reset-btn > button {
        background: transparent !important;
        color: #f87171 !important;
        border: 1px solid #f87171 !important;
        font-size: 0.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 모델 로드 (캐시)
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return pipeline(
        'zero-shot-classification',
        model='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
    )

# ─────────────────────────────────────────
# 초기 상태
# ─────────────────────────────────────────
CATEGORIES = ['경제', '산업', '주거', '환경', '교통', '보건', '복지', '교육', '문화', '안전']
HYPOTHESIS_TEMPLATE = '이 텍스트는 {}에 관한 내용입니다.'

CATEGORY_EMOJI = {
    '경제': '💰', '산업': '🏭', '주거': '🏠', '환경': '🌿', '교통': '🚆',
    '보건': '🏥', '복지': '🤝', '교육': '📚', '문화': '🎭', '안전': '🛡️'
}

if 'city_state' not in st.session_state:
    st.session_state.city_state = {c: 50 for c in CATEGORIES}
if 'history' not in st.session_state:
    st.session_state.history = []   # list of (sentence, label, score)
if 'last_event' not in st.session_state:
    st.session_state.last_event = None

# ─────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🏙️ 도시 성장 시뮬레이터</h1>
    <p>문장을 입력하면 AI가 도시의 어떤 분야가 성장하는지 분석합니다</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 상단 요약 통계
# ─────────────────────────────────────────
city = st.session_state.city_state
total_score = sum(city.values())
top_field = max(city, key=city.get)
total_events = len(st.session_state.history)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-label">총 도시 점수</div>
        <div class="stat-value">{total_score}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-label">최고 분야</div>
        <div class="stat-value">{CATEGORY_EMOJI[top_field]} {top_field}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-label">최고 점수</div>
        <div class="stat-value">{city[top_field]}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-label">누적 이벤트</div>
        <div class="stat-value">{total_events}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 입력 + 버튼
# ─────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    sentence = st.text_input(
        label="입력",
        placeholder="예: 새로운 지하철 노선이 개통되었습니다.",
        label_visibility="collapsed"
    )
with col_btn:
    analyze_btn = st.button("분석 →", use_container_width=True)

# ─────────────────────────────────────────
# 분석 실행
# ─────────────────────────────────────────
if analyze_btn and sentence.strip():
    with st.spinner("AI가 분석 중..."):
        clf = load_model()
        result = clf(
            sentence,
            candidate_labels=CATEGORIES,
            hypothesis_template=HYPOTHESIS_TEMPLATE
        )
    top_label = result['labels'][0]
    top_score = result['scores'][0]

    st.session_state.city_state[top_label] += 10
    st.session_state.last_event = (sentence, top_label, top_score)
    st.session_state.history.insert(0, (sentence, top_label, top_score))
    st.rerun()

elif analyze_btn and not sentence.strip():
    st.warning("문장을 입력해 주세요.")

# ─────────────────────────────────────────
# 최신 이벤트 카드
# ─────────────────────────────────────────
if st.session_state.last_event:
    s, label, score = st.session_state.last_event
    st.markdown(f"""
    <div class="event-card">
        <div class="label">✅ 최신 이벤트</div>
        <div class="message">{CATEGORY_EMOJI[label]} {label} 분야가 발전했습니다! (+10점)</div>
        <div class="score-info">입력: "{s}"&nbsp;&nbsp;|&nbsp;&nbsp;분류 신뢰도: {score:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# 메인 레이아웃: 차트 | 히스토리
# ─────────────────────────────────────────
left_col, right_col = st.columns([3, 2])

with left_col:
    st.markdown("#### 📊 도시 현황")
    df = pd.DataFrame({
        '분야': [f"{CATEGORY_EMOJI[k]} {k}" for k in CATEGORIES],
        '점수': [city[k] for k in CATEGORIES]
    }).sort_values('점수', ascending=True)

    colors = ['#4ade80' if v == max(city.values()) else '#3b82f6'
              for v in df['점수']]

    fig = go.Figure(go.Bar(
        x=df['점수'],
        y=df['분야'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0)', width=0)
        ),
        text=df['점수'],
        textposition='outside',
        textfont=dict(color='#ccc', size=12)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=50, t=10, b=10),
        height=380,
        xaxis=dict(
            showgrid=True, gridcolor='#2e3650',
            color='#666', range=[0, max(city.values()) + 20]
        ),
        yaxis=dict(color='#ccc', tickfont=dict(size=13)),
        font=dict(family='Noto Sans KR'),
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.markdown("#### 🕑 이벤트 히스토리")
    if st.session_state.history:
        for s, label, score in st.session_state.history[:15]:
            short_s = s[:22] + "…" if len(s) > 22 else s
            st.markdown(f"""
            <div class="history-item">
                <span class="history-badge">{CATEGORY_EMOJI[label]} {label}</span>
                <span>{short_s}</span>
                <span style="margin-left:auto;color:#555;font-size:0.78rem">{score:.0%}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#555;font-size:0.9rem'>아직 이벤트가 없습니다.</p>",
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    if st.button("🔄 도시 초기화", use_container_width=True):
        st.session_state.city_state = {c: 50 for c in CATEGORIES}
        st.session_state.history = []
        st.session_state.last_event = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)