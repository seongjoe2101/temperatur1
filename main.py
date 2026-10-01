import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 연평균기온 데이터를 이용해 연도별 기온을 분석하고 예측합니다.")

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 날짜 또는 평균기온이 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# --------------------------------------------------
# 연도별 평균기온 계산
# --------------------------------------------------
# 2025년까지만 사용
df = df[df["연도"] <= 2025].copy()

annual = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 관측일이 300일 미만인 연도 제외
annual = annual[annual["관측일수"] >= 300].copy()

# 정렬
annual = annual.sort_values("연도").reset_index(drop=True)

# --------------------------------------------------
# 회귀 분석
# --------------------------------------------------
# 1908년부터 지난 연수를 독립 변수로 사용
annual["경과연수"] = annual["연도"] - 1908

x = annual["경과연수"].to_numpy()
y = annual["연평균기온"].to_numpy()

# 1차 선형 회귀
slope, intercept = np.polyfit(x, y, 1)

# 예측값
annual["회귀예측기온"] = slope * x + intercept

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 회귀식
if intercept >= 0:
    equation = f"기온 = {slope:.4f} × 경과연수 + {intercept:.4f}"
else:
    equation = f"기온 = {slope:.4f} × 경과연수 - {abs(intercept):.4f}"

# --------------------------------------------------
# 분석 정보
# --------------------------------------------------
start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
data_count = len(annual)

st.subheader("📊 회귀 분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("회귀에 사용한 연도 수", f"{data_count}개")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

with col4:
    st.metric("상관계수", f"{correlation:.4f}")

st.caption(f"회귀식: {equation}")

# --------------------------------------------------
# 산점도 + 회귀선
# --------------------------------------------------
st.subheader("📈 연도별 평균기온과 회귀선")

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=annual["관측일수"],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)

# 회귀선
# 1908~2025의 실제 분석 구간을 표시
line_years = np.arange(start_year, end_year + 1)
line_x = line_years - 1908
line_y = slope * line_x + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="회귀선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예상기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (℃)",
    hovermode="x unified",
    height=550,
    xaxis=dict(
        dtick=10,
        tickmode="linear"
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    f"상관계수는 {correlation:.4f}입니다. "
    "상관계수는 연도와 연평균기온 사이의 선형적인 관계가 어느 정도인지를 나타냅니다."
)

# --------------------------------------------------
# 연도 슬라이더
# --------------------------------------------------
st.subheader("🔮 연도별 예상 기온")

selected_year = st.slider(
    "예상하고 싶은 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 선택한 연도의 경과연수
selected_elapsed = selected_year - 1908

# 회귀식으로 예상 기온 계산
predicted_temp = slope * selected_elapsed + intercept

st.markdown(
    f"""
    <div style="
        background-color:#f5f7fa;
        padding:30px;
        border-radius:15px;
        text-align:center;
        margin-top:20px;
        margin-bottom:20px;
    ">
        <div style="font-size:24px; color:#555;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="font-size:55px; font-weight:bold; margin-top:10px;">
            {predicted_temp:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# 선택 연도의 실제값이 있는 경우 비교
# --------------------------------------------------
selected_actual = annual[
    annual["연도"] == selected_year
]

if not selected_actual.empty:
    actual_temp = selected_actual.iloc[0]["연평균기온"]
    difference = predicted_temp - actual_temp

    st.write(
        f"**실제 연평균기온:** {actual_temp:.2f}℃  "
        f"/ **회귀 예상값:** {predicted_temp:.2f}℃  "
        f"/ **차이:** {difference:+.2f}℃"
    )
else:
    st.write(
        "선택한 연도는 회귀 분석에 사용된 실제 관측자료가 없으므로 "
        "회귀식을 이용한 예상값만 표시합니다."
    )

# --------------------------------------------------
# 데이터 조건 안내
# --------------------------------------------------
st.subheader("ℹ️ 분석 기준")

st.write(
    """
    - 분석 기준 기간: 2025년까지
    - 연간 관측일수가 300일 미만인 연도는 제외
    - 연평균기온은 해당 연도의 유효한 평균기온 자료로 계산
    - 회귀분석의 독립변수: 1908년부터 지난 연수
    - 슬라이더 범위: 1900년 ~ 2100년
    """
)

# --------------------------------------------------
# 원본 연도별 데이터
# --------------------------------------------------
with st.expander("📋 회귀 분석에 사용된 연도별 데이터 보기"):
    display_df = annual[
        ["연도", "관측일수", "연평균기온", "회귀예측기온"]
    ].copy()

    display_df["연평균기온"] = display_df["연평균기온"].round(2)
    display_df["회귀예측기온"] = display_df["회귀예측기온"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
