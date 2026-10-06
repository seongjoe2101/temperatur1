
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from numpy.polynomial import Polynomial

# -----------------------------------------
# 기본 설정
# -----------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("연평균기온을 이용해 1차, 3차, 9차 곡선으로 미래 기온을 예측합니다.")

# -----------------------------------------
# 데이터 불러오기
# -----------------------------------------
URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

try:
    df = pd.read_csv(URL, encoding="utf-8")
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# -----------------------------------------
# 데이터 전처리
# -----------------------------------------
df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

df = df.dropna(subset=["날짜", "평균기온"]).copy()

# 연도 추출
df["연도"] = df["날짜"].dt.year

# 2025년까지만 사용
df = df[df["연도"] <= 2025].copy()

# -----------------------------------------
# 연도별 관측일 수 계산
# 300일 미만인 연도 제외
# -----------------------------------------
year_count = (
    df.groupby("연도")
    .size()
    .reset_index(name="관측일수")
)

valid_years = year_count.loc[
    year_count["관측일수"] >= 300,
    "연도"
]

df = df[df["연도"].isin(valid_years)].copy()

# -----------------------------------------
# 연도별 평균기온
# -----------------------------------------
annual = (
    df.groupby("연도")["평균기온"]
    .mean()
    .reset_index()
    .sort_values("연도")
)

# -----------------------------------------
# 학습 / 테스트 분리
# -----------------------------------------
train = annual[annual["연도"] < 2005].copy()
test = annual[annual["연도"] >= 2005].copy()

st.subheader("📊 학습용과 테스트용 데이터")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "훈련용 연도 수",
        f"{len(train)}개 연도"
    )

with col2:
    st.metric(
        "테스트용 연도 수",
        f"{len(test)}개 연도"
    )

st.info(
    "훈련용 데이터는 2005년 이전, 테스트용 데이터는 2005년부터입니다. "
    "모든 모델은 훈련용 데이터만 사용하여 학습하고, 테스트용 데이터는 마지막 평가에만 사용합니다."
)

# -----------------------------------------
# 데이터가 충분한지 확인
# -----------------------------------------
if len(train) < 10:
    st.error("훈련용 데이터가 너무 적어서 곡선 회귀를 수행하기 어렵습니다.")
    st.stop()

if len(test) == 0:
    st.error("테스트용 데이터가 없습니다.")
    st.stop()

# -----------------------------------------
# 연도를 작은 숫자로 변환
#
# 1908 → 0
# 1909 → 1
# ...
#
# 이렇게 하면 9차 다항식 계산에서
# 지나치게 큰 숫자가 생기는 것을 방지할 수 있음.
# -----------------------------------------
base_year = annual["연도"].min()

train_x = (train["연도"] - base_year).to_numpy(dtype=float)
train_y = train["평균기온"].to_numpy(dtype=float)

test_x = (test["연도"] - base_year).to_numpy(dtype=float)
test_y = test["평균기온"].to_numpy(dtype=float)

# -----------------------------------------
# 모델 학습
# Polynomial.fit은 입력 범위를 자동으로 안정적으로
# 변환하여 고차 다항식 회귀에 사용하기 좋음.
# -----------------------------------------
degrees = [1, 3, 9]

models = {}

for degree in degrees:
    models[degree] = Polynomial.fit(
        train_x,
        train_y,
        degree
    )

# -----------------------------------------
# 테스트 데이터로 평가
#
# MAE = 평균 절대 오차
# 실제 테스트 기온과 예측 기온의 차이의 절댓값 평균
# -----------------------------------------
results = []

for degree in degrees:
    model = models[degree]

    test_pred = model(test_x)

    mae = np.mean(
        np.abs(test_y - test_pred)
    )

    prediction_2050 = float(
        model(2050 - base_year)
    )

    results.append({
        "모델": f"{degree}차",
        "테스트 평균 오차": mae,
        "2050년 예측기온": prediction_2050
    })

result_df = pd.DataFrame(results)

# 화면 표시용 반올림
display_df = result_df.copy()

display_df["테스트 평균 오차"] = (
    display_df["테스트 평균 오차"].round(2)
)

display_df["2050년 예측기온"] = (
    display_df["2050년 예측기온"].round(2)
)

st.subheader("📋 모델 비교")

st.dataframe(
    display_df.rename(columns={
        "모델": "모델",
        "테스트 평균 오차": "테스트 평균 오차(℃)",
        "2050년 예측기온": "2050년 예측기온(℃)"
    }),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "테스트 평균 오차는 학습에 사용하지 않은 2005년 이후 데이터의 평균 절대 오차(MAE)입니다."
)

# -----------------------------------------
# 어떤 모델이 테스트에서 가장 잘 맞았는지
# -----------------------------------------
best_row = result_df.loc[
    result_df["테스트 평균 오차"].idxmin()
]

st.success(
    f"테스트 데이터에서 가장 적게 빗나간 모델은 "
    f"**{best_row['모델']}**입니다. "
    f"평균적으로 약 **{best_row['테스트 평균 오차']:.2f}℃** 차이가 났습니다."
)

# -----------------------------------------
# 그래프
# -----------------------------------------
st.subheader("📈 연평균기온과 3가지 곡선")

fig = go.Figure()

# 전체 실제 데이터
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=6)
    )
)

# 학습용 데이터
fig.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["평균기온"],
        mode="markers",
        name="훈련용 데이터",
        marker=dict(size=7, symbol="circle")
    )
)

# 테스트용 데이터
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="markers",
        name="테스트용 데이터",
        marker=dict(size=7, symbol="diamond")
    )
)

# 곡선을 그릴 연도 범위
plot_years = np.linspace(
    annual["연도"].min(),
    2050,
    600
)

plot_x = plot_years - base_year

line_names = {
    1: "1차 곡선",
    3: "3차 곡선",
    9: "9차 곡선"
}

for degree in degrees:
    model = models[degree]

    predicted = model(plot_x)

    fig.add_trace(
        go.Scatter(
            x=plot_years,
            y=predicted,
            mode="lines",
            name=line_names[degree]
        )
    )

# 2050년 표시
for degree in degrees:
    prediction = float(
        models[degree](2050 - base_year)
    )

    fig.add_trace(
        go.Scatter(
            x=[2050],
            y=[prediction],
            mode="markers",
            name=f"{degree}차 2050년 예측",
            marker=dict(size=11)
        )
    )

fig.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="2005년: 학습 → 테스트",
    annotation_position="top"
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (℃)",
    hovermode="x unified",
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------
# 모델별 설명
# -----------------------------------------
st.subheader("🔎 모델 해석")

st.write(
    """
- **1차 곡선**: 전체적인 상승 또는 하락 추세를 직선으로 표현합니다.
- **3차 곡선**: 직선보다 더 유연하게 기온 변화의 굴곡을 표현합니다.
- **9차 곡선**: 훈련 데이터의 복잡한 변화까지 따라갈 수 있지만, 
  너무 복잡해지면 테스트 데이터에서 오히려 크게 빗나갈 수 있습니다.
"""
)

st.info(
    "중요: 2050년 예측값은 각 모델이 2005년 이전의 훈련용 데이터만 학습한 뒤 계산한 값입니다. "
    "2005년 이후 테스트 데이터는 2050년 예측을 만드는 과정에도 사용하지 않았습니다."
)

# -----------------------------------------
# 원본 데이터 요약통계
# -----------------------------------------
st.subheader("📌 원본 데이터 요약통계")

summary = df["평균기온"].describe()

summary_df = pd.DataFrame({
    "항목": [
        "개수",
        "평균",
        "최소",
        "최대"
    ],
    "값": [
        summary["count"],
        summary["mean"],
        summary["min"],
        summary["max"]
    ]
})

summary_df["값"] = summary_df["값"].round(2)

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)
