import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.markdown("서울 기온 데이터를 활용하여 연도별 평균기온을 분석하고, 선형 회귀를 통해 미래의 기온을 예측하는 스트림릿 앱입니다.")

# Load data with caching
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding='utf-8')
    return df

try:
    with st.spinner("데이터를 불러오는 중입니다..."):
        df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# Data preprocessing
# Columns expected: 날짜, 지점, 평균기온, 최저기온, 최고기온
df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
df['연도'] = df['날짜'].dt.year
df['월'] = df['날짜'].dt.month
df['일'] = df['날짜'].dt.day

# Filter data up to 2025 as per instructions
df = df[df['연도'] <= 2025]

# Group by year to calculate annual mean temperature and count valid observation days
annual_stats = df.groupby('연도').agg(
    연평균기온=('평균기온', 'mean'),
    관측일수=('평균기온', 'count')
).reset_index()

# Exclude years with less than 300 observation days
filtered_stats = annual_stats[annual_stats['관측일수'] >= 300].copy()

# Filter for regression starting from 1908 onwards as specified
reg_data = filtered_stats[filtered_stats['연도'] >= 1908].copy()

if reg_data.empty:
    st.warning("회귀 분석을 위한 충분한 데이터가 없습니다 (1908년 이후 관측일수 300일 이상인 해가 없음).")
    st.stop()

# Independent variable: elapsed years since 1908 (e.g., 1908 -> 0, 1909 -> 1, ...)
reg_data['지난연수'] = reg_data['연도'] - 1908

# Calculate linear regression coefficients using numpy (polyfit degree 1)
# y = slope * 지난연수 + intercept
slope, intercept = np.polyfit(reg_data['지난연수'], reg_data['연평균기온'], 1)

# Calculate correlation coefficient (Pearson) between '연도' and '연평균기온'
correlation = reg_data['연도'].corr(reg_data['연평균기온'])

# Summary metrics
num_years = len(reg_data)
start_year = int(reg_data['연도'].min())
end_year = int(reg_data['연도'].max())

# Display dataset information
st.markdown("---")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("분석에 사용된 해의 개수", f"{num_years}개")
col_m2.metric("시작 연도", f"{start_year}년")
col_m3.metric("끝 연도", f"{end_year}년")
col_m4.metric("상관계수 (연도 vs 기온)", f"{correlation:.4f}")

# Generate regression line values for plotting
reg_data['회귀예측기온'] = slope * reg_data['지난연수'] + intercept

# Plotly Scatter Plot with Regression Line
fig = px.scatter(
    reg_data,
    x='연도',
    y='연평균기온',
    title="서울 연도별 평균기온 및 회귀 직선 (1908년~)",
    labels={'연도': '연도', '연평균기온': '연평균기온 (°C)'},
    template='plotly_white'
)

# Add regression line
fig.add_trace(
    go.Scatter(
        x=reg_data['연도'],
        y=reg_data['회귀예측기온'],
        mode='lines',
        name='회귀 직선',
        line=dict(color='red', width=2)
    )
)

fig.update_layout(
    xaxis=dict(tickmode='linear', dtick=10),
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Temperature Predictor Slider Section
st.markdown("---")
st.subheader("🔮 특정 연도 기온 예측하기")
st.markdown("슬라이더를 움직여 원하는 연도를 선택하면, 회귀 모델을 바탕으로 한 해당 연도의 예상 평균기온을 확인할 수 있습니다.")

selected_year = st.slider("조회/예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)

# Calculate prediction for selected year using the formula based on 1908
selected_elapsed = selected_year - 1908
predicted_temp = slope * selected_elapsed + intercept

# Display big result
st.markdown(
    f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center; border: 2px solid #ff4b4b;">
        <h3 style="margin: 0; color: #31333F;">{selected_year}년 예상 평균기온</h3>
        <h1 style="margin: 10px 0 0 0; color: #ff4b4b; font-size: 48px;">{predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
st.caption("데이터 출처: 서울 기온 공공 데이터 | 개발: 기온 예측기 스트림릿 앱")
