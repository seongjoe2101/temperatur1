import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균기온을 이용해 1차, 3차, 9차 곡선 회귀모델을 만들고 "
    "학습에 사용하지 않은 테스트 데이터로 성능을 평가합니다."
)


# =========================================================
# 데이터 주소
# =========================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# =========================================================
# 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# =========================================================
# 연도별 평균기온 계산
# =========================================================

# 2025년까지만 사용
df = df[df["연도"] <= 2025].copy()

# 평균기온이 없는 행 제외
df = df.dropna(subset=["연도", "평균기온"])

# 연도별 평균기온과 관측일 수 계산
yearly = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 관측일수가 300일 미만인 해 제외
yearly = yearly[yearly["관측일수"] >= 300].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# =========================================================
# 훈련 / 테스트 데이터 분리
# =========================================================
# 2005년 이전 = 훈련
# 2005년부터 = 테스트

train = yearly[yearly["연도"] < 2005].copy()
test = yearly[yearly["연도"] >= 2005].copy()


# =========================================================
# 데이터 개수 표시
# =========================================================
st.subheader("📚 훈련 데이터와 테스트 데이터")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "훈련 데이터",
        f"{len(train)}개 연도"
    )

    if len(train) > 0:
        st.write(
            f"기간: **{int(train['연도'].min())}년 ~ "
            f"{int(train['연도'].max())}년**"
        )

with col2:
    st.metric(
        "테스트 데이터",
        f"{len(test)}개 연도"
    )

    if len(test) > 0:
        st.write(
            f"기간: **{int(test['연도'].min())}년 ~ "
            f"{int(test['연도'].max())}년**"
        )


st.info(
    "테스트 데이터는 모델을 만드는 과정에 사용하지 않고, "
    "모델이 처음 보는 데이터로 성능을 평가하는 데만 사용합니다."
)


if len(train) < 10 or len(test) == 0:
    st.error("훈련 또는 테스트 데이터가 충분하지 않습니다.")
    st.stop()


# =========================================================
# 수치 안정성을 위한 x 변환
# =========================================================
# 실제 연도를 그대로 9차식에 넣으면
# 1900^9 같은 매우 큰 숫자가 생겨 수치적으로 불안정할 수 있음.
#
# 따라서 연도를 평균과 표준편차를 이용해 작은 값으로 변환한다.
#
# x_scaled = (연도 - 훈련연도 평균) / 훈련연도 표준편차
#
# 이 변환은 모델의 입력값 크기만 줄이는 것이며,
# 실제 예측 대상 연도는 그대로 사용할 수 있다.

x_train_raw = train["연도"].to_numpy(dtype=float)
y_train = train["연평균기온"].to_numpy(dtype=float)

x_test_raw = test["연도"].to_numpy(dtype=float)
y_test = test["연평균기온"].to_numpy(dtype=float)

x_center = x_train_raw.mean()
x_scale = x_train_raw.std()

if x_scale == 0:
    st.error("훈련 데이터의 연도 값이 충분히 다양하지 않습니다.")
    st.stop()


def transform_year(year):
    return (np.asarray(year, dtype=float) - x_center) / x_scale


# =========================================================
# 다항회귀 함수
# =========================================================
def fit_polynomial(degree):
    """
    훈련 데이터에만 다항회귀를 적합한다.
    """

    x_train = transform_year(x_train_raw)

    coefficients = np.polyfit(
        x_train,
        y_train,
        degree
    )

    return coefficients


def predict(coefficients, years):
    """
    학습된 회귀계수로 특정 연도의 기온을 예측한다.
    """

    x = transform_year(years)

    return np.polyval(
        coefficients,
        x
    )


# =========================================================
# 평가 지표
# =========================================================
def calculate_mae(actual, predicted):
    return np.mean(
        np.abs(actual - predicted)
    )


def calculate_mse(actual, predicted):
    return np.mean(
        (actual - predicted) ** 2
    )


def calculate_r2(actual, predicted):
    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if ss_tot == 0:
        return np.nan

    return 1 - (ss_res / ss_tot)


# =========================================================
# 1차 / 3차 / 9차 모델 학습
# =========================================================
degrees = [1, 3, 9]

models = {}
results = []

for degree in degrees:

    # 오직 훈련 데이터로 모델 생성
    coefficients = fit_polynomial(degree)

    models[degree] = coefficients

    # 테스트 데이터에 대한 예측
    test_prediction = predict(
        coefficients,
        x_test_raw
    )

    # 테스트 데이터 평가
    mae = calculate_mae(
        y_test,
        test_prediction
    )

    mse = calculate_mse(
        y_test,
        test_prediction
    )

    r2 = calculate_r2(
        y_test,
        test_prediction
    )

    # 2050년 예측
    prediction_2050 = predict(
        coefficients,
        [2050]
    )[0]

    results.append({
        "차수": f"{degree}차",
        "테스트 MAE (℃)": mae,
        "테스트 MSE": mse,
        "테스트 R²": r2,
        "2050년 예상기온 (℃)": prediction_2050
    })


results_df = pd.DataFrame(results)


# =========================================================
# 평가 결과 표
# =========================================================
st.subheader("🏆 테스트 데이터 성능 비교")

display_results = results_df.copy()

display_results["테스트 MAE (℃)"] = (
    display_results["테스트 MAE (℃)"]
    .map(lambda x: f"{x:.3f}")
)

display_results["테스트 MSE"] = (
    display_results["테스트 MSE"]
    .map(lambda x: f"{x:.3f}")
)

display_results["테스트 R²"] = (
    display_results["테스트 R²"]
    .map(lambda x: f"{x:.3f}")
)

display_results["2050년 예상기온 (℃)"] = (
    display_results["2050년 예상기온 (℃)"]
    .map(lambda x: f"{x:.2f}")
)

st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)


st.caption(
    "MAE가 작을수록 평균적으로 적게 빗나가며, "
    "MSE가 작을수록 큰 오차가 적습니다. "
    "R²는 테스트 데이터의 변동을 얼마나 잘 설명하는지를 나타냅니다."
)


# =========================================================
# 테스트 성능 한눈에 보기
# =========================================================
best_model = results_df.loc[
    results_df["테스트 MAE (℃)"].idxmin()
]

st.success(
    f"테스트 MAE가 가장 작은 모델은 "
    f"**{best_model['차수']} 모델**입니다. "
    f"평균적으로 약 **{best_model['테스트 MAE (℃)']:.2f}℃** "
    f"정도 빗나갔습니다."
)


# =========================================================
# 회귀선 그래프
# =========================================================
st.subheader("📈 훈련 데이터로 만든 1차·3차·9차 회귀곡선")

# 그래프용 연도
plot_years = np.linspace(
    int(yearly["연도"].min()),
    2050,
    600
)

fig = go.Figure()


# ---------------------------------------------------------
# 실제 연평균기온
# ---------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["연평균기온"],
        mode="markers",
        name="훈련 데이터",
        marker=dict(size=6),
        customdata=train["관측일수"],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f} ℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)


# ---------------------------------------------------------
# 테스트 데이터
# ---------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="markers",
        name="테스트 데이터",
        marker=dict(
            size=8,
            symbol="diamond"
        ),
        customdata=test["관측일수"],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "실제 연평균기온: %{y:.2f} ℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)


# ---------------------------------------------------------
# 회귀곡선 3개
# ---------------------------------------------------------
for degree in degrees:

    coefficients = models[degree]

    predicted = predict(
        coefficients,
        plot_years
    )

    fig.add_trace(
        go.Scatter(
            x=plot_years,
            y=predicted,
            mode="lines",
            name=f"{degree}차 회귀곡선",
            line=dict(width=3 if degree != 9 else 2),
            hovertemplate=(
                f"<b>{degree}차 회귀곡선</b><br>"
                "연도: %{x:.0f}<br>"
                "예상기온: %{y:.2f} ℃"
                "<extra></extra>"
            )
        )
    )


# 2005년 경계선
fig.add_vline(
    x=2005,
    line_dash="dash",
    line_width=2,
    annotation_text="2005년: 테스트 시작",
    annotation_position="top"
)


# 2050년 선
fig.add_vline(
    x=2050,
    line_dash="dot",
    line_width=2,
    annotation_text="2050년",
    annotation_position="top"
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    hovermode="x unified",
    height=600,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 2050년 예측 비교
# =========================================================
st.subheader("🔮 2050년 예측 비교")

prediction_2050_df = results_df[
    ["차수", "2050년 예상기온 (℃)"]
].copy()

prediction_2050_df["2050년 예상기온 (℃)"] = (
    prediction_2050_df["2050년 예상기온 (℃)"]
    .round(2)
)

st.dataframe(
    prediction_2050_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 각 모델의 자세한 설명
# =========================================================
st.subheader("💡 모델 비교")

for degree in degrees:

    row = results_df[
        results_df["차수"] == f"{degree}차"
    ].iloc[0]

    st.write(
        f"**{degree}차 모델:** "
        f"테스트 MAE {row['테스트 MAE (℃)']:.3f}℃, "
        f"테스트 MSE {row['테스트 MSE']:.3f}, "
        f"테스트 R² {row['테스트 R²']:.3f}, "
        f"2050년 예측 {row['2050년 예상기온 (℃)']:.2f}℃"
    )


# =========================================================
# 데이터 확인
# =========================================================
with st.expander("📋 사용된 연도별 데이터 확인"):

    show_df = yearly[
        ["연도", "관측일수", "연평균기온"]
    ].copy()

    show_df["구분"] = np.where(
        show_df["연도"] < 2005,
        "훈련",
        "테스트"
    )

    show_df["연평균기온"] = (
        show_df["연평균기온"]
        .round(2)
    )

    show_df = show_df[
        ["연도", "구분", "관측일수", "연평균기온"]
    ]

    st.dataframe(
        show_df,
        use_container_width=True,
        hide_index=True
    )
