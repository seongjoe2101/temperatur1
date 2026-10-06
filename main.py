import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="서울 기온 다항 회귀 예측", layout="wide")
st.title("📈 서울 연평균 기온 다항 회귀(1차, 3차, 9차) 모델 비교")

# 데이터 불러오기 및 전처리
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 데이터 처리 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    df = df.dropna(subset=["평균기온"])
    
    # 연도별 관측일수 및 연평균 기온 집계
    yearly = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 필터링: 2025년 이하 & 관측일수 300일 이상
    filtered = yearly[(yearly["연도"] <= 2025) & (yearly["관측일수"] >= 300)].copy()
    return filtered

yearly_df = load_data()

# ---------------------------------------------------------
# 데이터 분할 (2005년 이전: 훈련용, 2005년 이후: 테스트용)
# ---------------------------------------------------------
train_df = yearly_df[yearly_df["연도"] < 2005].copy()
test_df = yearly_df[yearly_df["연도"] >= 2005].copy()

n_train = len(train_df)
n_test = len(test_df)

st.subheader("📌 데이터 분할 정보")
st.markdown(f"""
- **훈련용 데이터 (Train Set, ~2004년)**: 총 **{n_train}개** 연도 (`{int(train_df['연도'].min())}년 ~ {int(train_df['연도'].max())}년`)
- **테스트용 데이터 (Test Set, 2005년~)**: 총 **{n_test}개** 연도 (`{int(test_df['연도'].min())}년 ~ {int(test_df['연도'].max())}년`)
""")

st.divider()

# ---------------------------------------------------------
# 수치 안정성을 위한 X변수 표준화 (Standardization)
# ---------------------------------------------------------
# 훈련 데이터의 연도 평균과 표준편차 기준으로 변환
x_train_orig = train_df["연도"].values
y_train = train_df["평균기온"].values

x_mean = np.mean(x_train_orig)
x_std = np.std(x_train_orig)

x_train_scaled = (x_train_orig - x_mean) / x_std

x_test_orig = test_df["연도"].values
y_test = test_df["평균기온"].values
x_test_scaled = (x_test_orig - x_mean) / x_std

# 2050년 스케일링 값
x_2050_scaled = (2050 - x_mean) / x_std

# ---------------------------------------------------------
# 1차, 3차, 9차 다항 회귀 적합 및 평가
# ---------------------------------------------------------
degrees = [1, 3, 9]
results = []
models = {}

for deg in degrees:
    # 훈련용 데이터로만 모델 적합 (np.polyfit)
    weights = np.polyfit(x_train_scaled, y_train, deg)
    poly_func = np.poly1d(weights)
    models[deg] = poly_func
    
    # 학습에 사용되지 않은 테스트용 데이터로 예측 및 채점
    y_pred_test = poly_func(x_test_scaled)
    
    # 평균 오차(MAE: 평균 몇 도나 빗나갔는지)
    mae = np.mean(np.abs(y_test - y_pred_test))
    
    # 2050년 기온 예측값
    pred_2050 = poly_func(x_2050_scaled)
    
    results.append({
        "차수": f"{deg}차 모델",
        "테스트 데이터 평균 오차 (MAE, °C)": round(mae, 3),
        "2050년 예상 기온 (°C)": round(pred_2050, 2)
    })

results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# 평가 결과 표 출력
# ---------------------------------------------------------
st.subheader("📊 모델별 테스트 데이터 채점 결과 및 2050년 예측값 비교")
st.table(results_df)

st.info("""
💡 **결과 해석 참고:**
- **1차 직선 모델**: 추세를 단순화하여 무난하게 외삽(Extrapolation)하지만 최근 상승 흐름을 다 담지 못합니다.
- **3차 곡선 모델**: 전체적인 완만한 곡선 변화를 반영하여 테스트 데이터에 대해 가장 낮은 평균 오차를 보입니다.
- **9차 고차 모델**: 훈련 데이터의 미세한 노이즈까지 과적합(Overfitting)하여, 미래 연도(2050년 등) 예측 시 수치가 폭발적으로 발산하는 현상을 확인할 수 있습니다.
""")

st.divider()

# ---------------------------------------------------------
# Plotly 곡선 추세 시각화
# ---------------------------------------------------------
st.subheader("📈 훈련 데이터 vs 테스트 데이터 및 차수별 회귀 곡선")

fig = go.Figure()

# 훈련 데이터 점
fig.add_trace(
    go.Scatter(
        x=train_df["연도"],
        y=train_df["평균기온"],
        mode="markers",
        name="훈련용 데이터 (~2004)",
        marker=dict(color="steelblue", opacity=0.7, size=7)
    )
)

# 테스트 데이터 점
fig.add_trace(
    go.Scatter(
        x=test_df["연도"],
        y=test_df["평균기온"],
        mode="markers",
        name="테스트용 데이터 (2005~)",
        marker=dict(color="crimson", size=9, symbol="diamond")
    )
)

# 회귀 곡선 그래프용 연도 세분화 (1900년 ~ 2050년)
plot_years = np.linspace(1900, 2050, 400)
plot_years_scaled = (plot_years - x_mean) / x_std

colors = {1: "gray", 3: "green", 9: "purple"}
dash_styles = {1: "dash", 3: "solid", 9: "dot"}

for deg in degrees:
    poly_func = models[deg]
    y_curve = poly_func(plot_years_scaled)
    
    fig.add_trace(
        go.Scatter(
            x=plot_years,
            y=y_curve,
            mode="lines",
            name=f"{deg}차 곡선 모델",
            line=dict(color=colors[deg], width=2.5, dash=dash_styles[deg])
        )
    )

fig.update_layout(
    xaxis=dict(title="연도", range=[1900, 2055], dtick=15),
    yaxis=dict(title="평균기온 (°C)", range=[8, 30]),  # 9차 폭발 발산 가시화를 위해 범위 제어
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    template="plotly_white",
    height=600
)

st.plotly_chart(fig, use_container_width=True)
