import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════
#  페이지 설정 (아이콘 및 타이틀)
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="국내주식 RSI 신호기",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════
#  모바일 최적화 CSS (v3.0 디자인 계승)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 14px; }
.block-container { padding: 0.8rem 0.8rem 2rem; max-width: 100%; }
.sig-card { border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; border-left: 5px solid; }
.card-buy     { background:#0d2b1a; border-color:#22c55e; }
.card-sell    { background:#2b0d0d; border-color:#ef4444; }
.card-watch-l { background:#1a1700; border-color:#eab308; }
.card-watch-h { background:#1a0e00; border-color:#f97316; }
.card-normal  { background:#1a1f2e; border-color:#334155; }
.card-title   { font-size:15px; font-weight:700; margin-bottom:4px; }
.card-row     { font-size:12px; color:#94a3b8; margin-top:3px; }
.card-signal  { font-size:13px; font-weight:600; margin-top:6px; }
.badge { display:inline-block; padding:2px 8px; border-radius:20px; font-size:11px; font-weight:600; margin-right:4px; }
.b-buy    { background:#166534; color:#bbf7d0; }
.b-sell   { background:#7f1d1d; color:#fecaca; }
.b-watch  { background:#713f12; color:#fef08a; }
.b-normal { background:#1e293b; color:#94a3b8; }
button[data-baseweb="tab"] { font-size:13px !important; padding:10px 14px !important; }
[data-testid="stMetricValue"] { font-size:22px !important; }
hr { margin: 0.5rem 0; border-color: #334155; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  국내 주요 50개 종목 리스트 (거래대금/시총 상위)
# ══════════════════════════════════════════════════════════════
# (종목명, 티커, 카테고리)
KR_STOCKS = [
    ("삼성전자", "005930.KS", "반도체"), ("SK하이닉스", "000660.KS", "반도체"),
    ("LG에너지솔루션", "373220.KS", "2차전지"), ("삼성바이오로직스", "207940.KS", "제약바이오"),
    ("현대차", "005380.KS", "자동차"), ("기아", "000270.KS", "자동차"),
    ("셀트리온", "068270.KS", "제약바이오"), ("POSCO홀딩스", "005490.KS", "철강"),
    ("KB금융", "105560.KS", "금융"), ("NAVER", "035420.KS", "플랫폼"),
    ("신한지주", "055550.KS", "금융"), ("삼성물산", "028260.KS", "지배구조"),
    ("삼성SDI", "006400.KS", "2차전지"), ("LG화학", "051910.KS", "화학"),
    ("카카오", "035720.KS", "플랫폼"), ("포스코퓨처엠", "003670.KS", "2차전지"),
    ("하나금융지주", "086790.KS", "금융"), ("메리츠금융지주", "138040.KS", "금융"),
    ("현대모비스", "012330.KS", "자동차"), ("에코프로비엠", "247540.KQ", "2차전지"),
    ("에코프로", "086520.KQ", "2차전지"), ("삼성생명", "032830.KS", "보험"),
    ("LG전자", "066570.KS", "가전"), ("삼성화재", "000810.KS", "보험"),
    ("크래프톤", "259960.KS", "게임"), ("카카오뱅크", "323410.KS", "금융"),
    ("HMM", "011200.KS", "해운"), ("한국전력", "015760.KS", "유틸리티"),
    ("삼성에스디에스", "018260.KS", "IT서비스"), ("SK이노베이션", "096770.KS", "에너지"),
    ("대한항공", "003490.KS", "항공"), ("두산에너빌리티", "034020.KS", "에너지"),
    ("KT&G", "033780.KS", "필수소비재"), ("하이브", "352820.KS", "엔터"),
    ("HD현대중공업", "329180.KS", "조선"), ("SK", "034730.KS", "지주사"),
    ("카카오페이", "377300.KS", "금융"), ("S-Oil", "010950.KS", "정유"),
    ("우리금융지주", "316140.KS", "금융"), ("삼성중공업", "010140.KS", "조선"),
    ("LIG넥스원", "079550.KS", "방산"), ("아모레퍼시픽", "090430.KS", "뷰티"),
    ("한화오션", "042660.KS", "조선"), ("넷마블", "251270.KS", "게임"),
    ("유한양행", "000100.KS", "제약"), ("한미반도체", "042700.KS", "반도체"),
    ("루닛", "328130.KQ", "의료AI"), ("알테오젠", "191170.KQ", "제약바이오"),
    ("HLB", "028300.KQ", "제약바이오"), ("레인보우로보틱스", "277810.KQ", "로봇")
]

# ══════════════════════════════════════════════════════════════
#  파라미터 (v3.0 설정 유지)
# ══════════════════════════════════════════════════════════════
RSI_PERIOD      = 14
RSI_MA_PERIOD   = 5
OVERSOLD        = 30
OVERBOUGHT      = 70
LOOKBACK        = 10
DATA_PERIOD     = "120d"

# ══════════════════════════════════════════════════════════════
#  계산 함수
# ══════════════════════════════════════════════════════════════
def calc_rsi(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).round(2)

def detect_signal(rsi: pd.Series):
    ma = rsi.rolling(RSI_MA_PERIOD).mean()
    if ma.isna().all() or len(rsi) < LOOKBACK + 2:
        return "normal", "-", None

    cur_r, prev_r = float(rsi.iloc[-1]), float(rsi.iloc[-2])
    cur_m, prev_m = float(ma.iloc[-1]),  float(ma.iloc[-2])

    window = rsi.iloc[-(LOOKBACK + 1):-1]
    oversold   = bool((window < OVERSOLD).any())
    overbought = bool((window > OVERBOUGHT).any())

    # 매수: 과매도 구역 진입 후 RSI-MA 상향 돌파
    if oversold and prev_r < prev_m and cur_r >= cur_m:
        return "buy", "🟢 매수 신호 (골든크로스)", "BUY"

    # 매도: 과매수 구역 진입 후 RSI-MA 하향 돌파
    if overbought and prev_r > prev_m and cur_r <= cur_m:
        return "sell", "🔴 매도 시그널 (데드크로스)", "SELL"

    if cur_r < OVERSOLD:
        return "watch_low", f"⚠️ 과매도 진입 ({cur_r:.1f})", None
    if cur_r > OVERBOUGHT:
        return "watch_high", f"⚠️ 과매수 진입 ({cur_r:.1f})", None

    return "normal", "-", None

@st.cache_data(ttl=3600)
def fetch_kr_data():
    results = []
    for name, ticker, cat in KR_STOCKS:
        try:
            raw = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
            if raw.empty: continue
            
            # 종가 추출
            close = raw["Close"].iloc[:, 0] if isinstance(raw["Close"], pd.DataFrame) else raw["Close"]
            rsi = calc_rsi(close)
            tag, signal, _ = detect_signal(rsi)
            
            results.append({
                "name": name, "ticker": ticker, "cat": cat,
                "rsi": round(rsi.iloc[-1], 1),
                "rsi_ma": round(rsi.rolling(RSI_MA_PERIOD).mean().iloc[-1], 1),
                "delta": round(rsi.iloc[-1] - rsi.iloc[-2], 1),
                "weekly": rsi.iloc[-7:].tolist(),
                "tag": tag, "signal": signal
            })
        except: continue
    return results

# ══════════════════════════════════════════════════════════════
#  UI 렌더링 함수
# ══════════════════════════════════════════════════════════════
def sparkline(weekly, tag):
    colors = {"buy":"#22c55e", "sell":"#ef4444", "watch_low":"#eab308", "watch_high":"#f97316", "normal":"#64748b"}
    color = colors.get(tag, "#64748b")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=weekly, mode='lines+markers', line=dict(color=color, width=2), marker=dict(size=4)))
    fig.update_layout(height=70, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False, range=[0, 100]))
    return fig

def render_card(r):
    cc = {"buy":"card-buy","sell":"card-sell","watch_low":"card-watch-l","watch_high":"card-watch-h"}.get(r["tag"],"card-normal")
    bc = {"buy":"b-buy","sell":"b-sell","watch_low":"b-watch","watch_high":"b-watch"}.get(r["tag"],"b-normal")
    
    return f"""
    <div class="sig-card {cc}">
      <div class="card-title"><span class="badge {bc}">{r['cat']}</span>{r['name']} <span style="font-size:11px;color:#64748b;">{r['ticker']}</span></div>
      <div class="card-row">RSI: <b>{r['rsi']}</b> | MA: {r['rsi_ma']} | 변화: {'▲' if r['delta']>0 else '▼'}{abs(r['delta'])}</div>
      <div class="card-signal">{r['signal']}</div>
    </div>
    """

# ══════════════════════════════════════════════════════════════
#  메인 실행부
# ══════════════════════════════════════════════════════════════
def main():
    st.markdown("## 🇰🇷 국내주식 RSI 신호 트래커")
    st.caption(f"RSI {RSI_PERIOD} / MA {RSI_MA_PERIOD} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if st.button("🔄 데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("국내 50개 종목 분석 중..."):
        results = fetch_kr_data()

    if not results:
        st.error("데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
        return

    # 요약 메트릭
    counts = pd.Series([r['tag'] for r in results]).value_counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🟢 매수", counts.get('buy', 0))
    m2.metric("🔴 매도", counts.get('sell', 0))
    m3.metric("⚠️ 과매도", counts.get('watch_low', 0))
    m4.metric("🟠 과매수", counts.get('watch_high', 0))

    # 탭 구성
    tab_labels = ["🟢 매수", "🔴 매도", "⚠️ 대기", "전체"]
    tabs = st.tabs(tab_labels)
    
    # 탭별 데이터 분류
    sections = [
        ([r for r in results if r['tag'] == 'buy'], tabs[0]),
        ([r for r in results if r['tag'] == 'sell'], tabs[1]),
        ([r for r in results if 'watch' in r['tag']], tabs[2]),
        (results, tabs[3])
    ] # <--- 이 부분의 괄호와 구조를 수정했습니다.
    
    # 데이터 렌더링 루프
    for items, tab in sections:
        with tab:
            if not items:
                st.info("조건에 맞는 종목이 없습니다.")
            else:
                # 리스트를 돌며 카드와 그래프 출력
                for r in items:
                    with st.container():
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(render_card(r), unsafe_allow_html=True)
                        with c2:
                            st.plotly_chart(
                                sparkline(r['weekly'], r['tag']), 
                                use_container_width=True, 
                                config={'displayModeBar': False}
                            )
if __name__ == "__main__":
    main()
