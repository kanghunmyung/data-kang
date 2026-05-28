import math
import textwrap
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud

st.set_page_config(
    page_title="데이터 시각화 놀이터",
    page_icon="🎨",
    layout="wide",
)

PALETTE = [
    "#FF6B6B",
    "#4ECDC4",
    "#FFD166",
    "#6A4C93",
    "#1A759F",
    "#F3722C",
    "#90BE6D",
    "#577590",
]

sns.set_theme(style="whitegrid")


def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    if uploaded_file.name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("CSV 또는 엑셀 파일만 올려주세요.")


def classify_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = []
    text_cols = []

    for col in df.columns:
        if col in numeric_cols:
            continue

        converted = pd.to_datetime(df[col], errors="coerce")
        success_ratio = converted.notna().mean()
        if success_ratio > 0.7:
            datetime_cols.append(col)
        else:
            text_cols.append(col)

    category_cols = text_cols.copy()
    for col in df.columns:
        if col not in category_cols and df[col].nunique(dropna=True) <= 15:
            category_cols.append(col)

    category_cols = list(dict.fromkeys(category_cols))
    return numeric_cols, category_cols, datetime_cols, text_cols


def friendly_summary(df):
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    return f"이 데이터는 {rows}줄, {cols}칸이고, 비어 있는 칸은 모두 {missing}개예요."


def simple_fact(df, category_col, value_col):
    view = df[[category_col, value_col]].dropna().copy()
    if view.empty:
        return "설명할 수 있는 데이터가 없어요."
    top = view.sort_values(value_col, ascending=False).iloc[0]
    bottom = view.sort_values(value_col, ascending=True).iloc[0]
    total = view[value_col].sum()
    avg = view[value_col].mean()
    return (
        f"가장 큰 것은 '{top[category_col]}'이고 값은 {top[value_col]:,.2f}예요. "
        f"가장 작은 것은 '{bottom[category_col]}'이고 값은 {bottom[value_col]:,.2f}예요. "
        f"전체를 더하면 {total:,.2f}, 평균은 {avg:,.2f}예요."
    )


def is_long_text_series(series, threshold=12):
    cleaned = series.dropna().astype(str)
    if cleaned.empty:
        return False
    avg_len = cleaned.map(len).mean()
    return avg_len >= threshold


def build_recommendations(df, numeric_cols, category_cols, datetime_cols, text_cols):
    recommendations = []

    if datetime_cols and numeric_cols:
        recommendations.append(
            {
                "title": "시간의 흐름 보기",
                "reason": "날짜와 숫자가 함께 있어서 시간이 지나며 어떻게 바뀌는지 보기 좋아요.",
                "charts": ["선그래프", "영역그래프", "버블 산점도"],
                "kind": "time",
                "columns": {"x": datetime_cols[0], "y": numeric_cols[0]},
            }
        )

    if category_cols and numeric_cols:
        recommendations.append(
            {
                "title": "무엇이 가장 큰지 비교하기",
                "reason": "이름이나 종류와 숫자가 함께 있어서 서로 크기를 비교하기 좋아요.",
                "charts": ["막대그래프", "원그래프", "트리맵", "깔때기 그래프"],
                "kind": "category_value",
                "columns": {"category": category_cols[0], "value": numeric_cols[0]},
            }
        )

    if numeric_cols:
        recommendations.append(
            {
                "title": "숫자가 어떻게 퍼져 있는지 보기",
                "reason": "숫자 열이 있어서 값이 어디에 많이 모여 있는지 살펴볼 수 있어요.",
                "charts": ["히스토그램", "박스플롯", "바이올린 플롯", "밀도 그래프"],
                "kind": "distribution",
                "columns": {"value": numeric_cols[0]},
            }
        )

    if len(numeric_cols) >= 2:
        recommendations.append(
            {
                "title": "숫자끼리 어떤 관계인지 보기",
                "reason": "숫자 열이 2개 이상 있어서 함께 커지는지 비교할 수 있어요.",
                "charts": ["산점도", "밀도 윤곽 그래프", "상관관계 히트맵"],
                "kind": "relationship",
                "columns": {"x": numeric_cols[0], "y": numeric_cols[1]},
            }
        )

    long_text_candidates = [col for col in text_cols if is_long_text_series(df[col])]
    if long_text_candidates:
        recommendations.append(
            {
                "title": "글자에서 자주 나오는 말 보기",
                "reason": "긴 글자 데이터가 있어서 어떤 말이 많이 나오는지 볼 수 있어요.",
                "charts": ["워드클라우드", "빈도 막대그래프"],
                "kind": "text",
                "columns": {"text": long_text_candidates[0]},
            }
        )
    elif text_cols:
        recommendations.append(
            {
                "title": "글자 종류 세어 보기",
                "reason": "글자 열이 있어서 어떤 항목이 많이 나오는지 셀 수 있어요.",
                "charts": ["빈도 막대그래프"],
                "kind": "text",
                "columns": {"text": text_cols[0]},
            }
        )

    return recommendations


def render_plotly(fig, title):
    st.subheader(title)
    fig.update_layout(font=dict(size=16), colorway=PALETTE)
    st.plotly_chart(fig, use_container_width=True)


def render_matplotlib(fig, title):
    st.subheader(title)
    st.pyplot(fig)
    plt.close(fig)


def render_recommendation_preview(df, recommendation):
    kind = recommendation["kind"]

    if kind == "time":
        x = recommendation["columns"]["x"]
        y = recommendation["columns"]["y"]
        time_df = df[[x, y]].dropna().copy()
        time_df[x] = pd.to_datetime(time_df[x], errors="coerce")
        time_df = time_df.dropna().sort_values(x)
        if not time_df.empty:
            render_plotly(px.line(time_df, x=x, y=y, markers=True), "추천 그래프 미리보기")

    elif kind == "category_value":
        category = recommendation["columns"]["category"]
        value = recommendation["columns"]["value"]
        chart_df = df[[category, value]].dropna().copy()
        chart_df = chart_df.groupby(category, as_index=False)[value].sum().sort_values(value, ascending=False)
        if not chart_df.empty:
            st.success(simple_fact(chart_df, category, value))
            render_plotly(px.bar(chart_df, x=category, y=value, color=category, text=value), "추천 그래프 미리보기")

    elif kind == "distribution":
        value = recommendation["columns"]["value"]
        dist_df = df[[value]].dropna().copy()
        if not dist_df.empty:
            render_plotly(px.histogram(dist_df, x=value, nbins=20), "추천 그래프 미리보기")

    elif kind == "relationship":
        x = recommendation["columns"]["x"]
        y = recommendation["columns"]["y"]
        rel_df = df[[x, y]].dropna().copy()
        if not rel_df.empty:
            corr = rel_df[x].corr(rel_df[y])
            st.success(f"'{x}'와 '{y}'의 상관 정도는 {corr:.2f}예요.")
            render_plotly(px.scatter(rel_df, x=x, y=y, trendline="ols"), "추천 그래프 미리보기")

    elif kind == "text":
        text_col = recommendation["columns"]["text"]
        series = df[text_col].dropna().astype(str)
        if not series.empty:
            counts = series.value_counts().head(15).reset_index()
            counts.columns = [text_col, "count"]
            render_plotly(px.bar(counts, x=text_col, y="count", color=text_col, text="count"), "추천 그래프 미리보기")
            if is_long_text_series(series):
                long_text = " ".join(series.tolist())
                wc = WordCloud(width=900, height=400, background_color="white", colormap="Set2").generate(long_text)
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                ax.set_title("추천 워드클라우드")
                render_matplotlib(fig, "추천 워드클라우드")


def render_auto_recommendations(df, numeric_cols, category_cols, datetime_cols, text_cols):
    st.header("⭐ 자동 추천 그래프")
    st.write("데이터를 읽고, 어떤 그래프가 잘 어울리는지 먼저 골라봤어요.")

    recommendations = build_recommendations(df, numeric_cols, category_cols, datetime_cols, text_cols)
    if not recommendations:
        st.info("아직 추천할 수 있는 그래프를 찾지 못했어요.")
        return

    for idx, recommendation in enumerate(recommendations, start=1):
        st.subheader(f"{idx}. {recommendation['title']}")
        st.info(recommendation["reason"])
        st.write("추천 그래프:", ", ".join(recommendation["charts"]))
        render_recommendation_preview(df, recommendation)
        st.markdown("---")


def add_download_button(df, filename):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="현재 데이터 CSV로 내려받기",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def visualization_tab_basic(df, category_cols, numeric_cols):
    st.header("1. 기본 비교 그래프")
    if not category_cols or not numeric_cols:
        st.info("기본 그래프를 만들려면 종류 열 1개와 숫자 열 1개가 필요해요.")
        return

    cat = st.selectbox("종류 열", category_cols, key="basic_cat")
    num = st.selectbox("숫자 열", numeric_cols, key="basic_num")

    chart_df = df[[cat, num]].dropna().copy()
    chart_df = chart_df.groupby(cat, as_index=False)[num].sum()
    chart_df = chart_df.sort_values(num, ascending=False)

    st.info(simple_fact(chart_df, cat, num))

    render_plotly(px.bar(chart_df, x=cat, y=num, color=cat, text=num), "막대그래프")
    render_plotly(px.pie(chart_df, names=cat, values=num, hole=0.3), "원그래프")
    render_plotly(px.treemap(chart_df, path=[cat], values=num), "트리맵")
    render_plotly(px.funnel(chart_df, x=num, y=cat), "깔때기 그래프")


def visualization_tab_time(df, datetime_cols, numeric_cols):
    st.header("2. 시간에 따라 보기")
    if not datetime_cols or not numeric_cols:
        st.info("시간 그래프를 만들려면 날짜 열과 숫자 열이 필요해요.")
        return

    date_col = st.selectbox("날짜 열", datetime_cols, key="time_date")
    value_col = st.selectbox("숫자 열 선택", numeric_cols, key="time_num")

    time_df = df[[date_col, value_col]].dropna().copy()
    time_df[date_col] = pd.to_datetime(time_df[date_col], errors="coerce")
    time_df = time_df.dropna().sort_values(date_col)
    if time_df.empty:
        st.warning("날짜를 읽을 수 없어요.")
        return

    render_plotly(px.line(time_df, x=date_col, y=value_col, markers=True), "선그래프")
    render_plotly(px.area(time_df, x=date_col, y=value_col), "영역그래프")
    render_plotly(px.scatter(time_df, x=date_col, y=value_col, size=value_col, color=value_col), "버블 산점도")


def visualization_tab_distribution(df, numeric_cols):
    st.header("3. 숫자 퍼짐 보기")
    if not numeric_cols:
        st.info("숫자 열이 있어야 해요.")
        return

    value_col = st.selectbox("보고 싶은 숫자 열", numeric_cols, key="dist_num")
    series = df[value_col].dropna()
    if series.empty:
        st.warning("값이 없어요.")
        return

    render_plotly(px.histogram(df, x=value_col, nbins=20), "히스토그램")
    render_plotly(px.box(df, y=value_col, points="all"), "박스플롯")
    render_plotly(px.violin(df, y=value_col, box=True, points="all"), "바이올린 플롯")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(series, fill=True, ax=ax, color=PALETTE[0])
    ax.set_title("밀도 그래프")
    render_matplotlib(fig, "밀도 그래프")


def visualization_tab_relation(df, numeric_cols):
    st.header("4. 숫자끼리 관계 보기")
    if len(numeric_cols) < 2:
        st.info("숫자 열이 2개 이상 있어야 해요.")
        return

    x_col = st.selectbox("가로축 숫자", numeric_cols, key="rel_x")
    y_candidates = [c for c in numeric_cols if c != x_col]
    y_col = st.selectbox("세로축 숫자", y_candidates, key="rel_y")

    plot_df = df[[x_col, y_col]].dropna()
    if plot_df.empty:
        st.warning("두 열을 함께 볼 데이터가 없어요.")
        return

    corr = plot_df[x_col].corr(plot_df[y_col])
    st.info(f"이 두 숫자의 상관 정도는 {corr:.2f}예요. 1에 가까우면 함께 커지고, -1에 가까우면 반대로 움직여요.")

    render_plotly(px.scatter(plot_df, x=x_col, y=y_col, trendline="ols"), "산점도")
    render_plotly(px.density_contour(plot_df, x=x_col, y=y_col), "밀도 윤곽 그래프")

    corr_df = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap="YlGnBu", ax=ax)
    ax.set_title("상관관계 히트맵")
    render_matplotlib(fig, "상관관계 히트맵")


def visualization_tab_text(df, text_cols):
    st.header("5. 글자 데이터 보기")
    if not text_cols:
        st.info("글자 열이 없어요.")
        return

    text_col = st.selectbox("글자 열 선택", text_cols, key="text_col")
    series = df[text_col].dropna().astype(str)
    if series.empty:
        st.warning("글자가 없어요.")
        return

    counts = series.value_counts().head(15).reset_index()
    counts.columns = [text_col, "count"]

    render_plotly(px.bar(counts, x=text_col, y="count", color=text_col, text="count"), "가장 많이 나온 글자")

    long_text = " ".join(series.tolist())
    if long_text.strip():
        wc = WordCloud(width=900, height=400, background_color="white", colormap="Set2").generate(long_text)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("워드클라우드")
        render_matplotlib(fig, "워드클라우드")


def visualization_tab_advanced(df, category_cols, numeric_cols):
    st.header("6. 여러 가지 특별한 보기")
    if not numeric_cols:
        st.info("숫자 열이 있어야 특별 그래프를 볼 수 있어요.")
        return

    value_col = st.selectbox("숫자 열 선택", numeric_cols, key="adv_num")
    render_plotly(px.ecdf(df.dropna(subset=[value_col]), x=value_col), "누적분포 그래프")

    if category_cols:
        cat_col = st.selectbox("색으로 구분할 종류 열", category_cols, key="adv_cat")
        sample_df = df[[cat_col, value_col]].dropna().copy()
        sample_df = sample_df.groupby(cat_col, as_index=False)[value_col].mean()
        render_plotly(px.sunburst(sample_df, path=[cat_col], values=value_col), "선버스트")
        render_plotly(px.strip(df.dropna(subset=[cat_col, value_col]), x=cat_col, y=value_col, color=cat_col), "스트립 플롯")


def visualization_tab_table(df, numeric_cols):
    st.header("7. 숫자 요약 표")
    st.dataframe(df.describe(include="all").fillna(""), use_container_width=True)

    if numeric_cols:
        corr_df = df[numeric_cols].corr(numeric_only=True).round(2)
        st.subheader("숫자끼리의 관계 표")
        st.dataframe(corr_df, use_container_width=True)


def visualization_tab_quiz(df, category_cols, numeric_cols):
    st.header("8. 퀴즈")
    if not category_cols or not numeric_cols:
        st.info("퀴즈를 만들려면 종류 열과 숫자 열이 필요해요.")
        return

    cat = st.selectbox("퀴즈용 종��� 열", category_cols, key="quiz_cat")
    num = st.selectbox("퀴즈용 숫자 열", numeric_cols, key="quiz_num")
    qdf = df[[cat, num]].dropna().groupby(cat, as_index=False)[num].sum().sort_values(num, ascending=False)
    if qdf.empty:
        st.warning("퀴즈를 만들 데이터가 없어요.")
        return

    top = qdf.iloc[0]
    bottom = qdf.iloc[-1]
    st.success(f"퀴즈 1! 가장 큰 값은 무엇일까요? 정답: {top[cat]}")
    st.success(f"퀴즈 2! 가장 작은 값은 무엇일까요? 정답: {bottom[cat]}")
    st.success(f"퀴즈 3! 모두 더하면 얼마일까요? 정답: {qdf[num].sum():,.2f}")


def main():
    st.title("🎨 데이터를 넣으면 여러 방법으로 보여주는 시각화 놀이터")
    st.write("CSV나 엑셀 파일을 올리면, 가능한 여러 가지 그래프로 자동 탐험할 수 있어요.")

    uploaded_file = st.file_uploader("CSV, XLSX, XLS 파일을 올려주세요", type=["csv", "xlsx", "xls"])

    with st.expander("샘플 데이터 형식 보기"):
        sample = pd.DataFrame(
            {
                "과일": ["사과", "바나나", "포도", "딸기"],
                "개수": [10, 6, 8, 12],
                "날짜": ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04"],
            }
        )
        st.dataframe(sample, use_container_width=True)

    if uploaded_file is None:
        st.info("파일을 올리면 막대그래프, 원그래프, 선그래프, 히트맵, 워드클라우드 등 여러 방식으로 보여드려요.")
        return

    try:
        df = load_data(uploaded_file)
    except Exception as exc:
        st.error(f"파일을 읽는 중 문제가 생겼어요: {exc}")
        return

    if df.empty:
        st.warning("비어 있는 파일이에요.")
        return

    numeric_cols, category_cols, datetime_cols, text_cols = classify_columns(df)

    st.success("파일을 잘 읽었어요!")
    st.info(friendly_summary(df))

    with st.expander("원본 데이터 보기", expanded=True):
        st.dataframe(df, use_container_width=True)
        add_download_button(df, "clean_data.csv")

    st.subheader("열 종류 살펴보기")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("숫자 열", len(numeric_cols))
    c2.metric("종류 열", len(category_cols))
    c3.metric("날짜 열", len(datetime_cols))
    c4.metric("글자 열", len(text_cols))

    render_auto_recommendations(df, numeric_cols, category_cols, datetime_cols, text_cols)

    tabs = st.tabs(
        [
            "기본 비교",
            "시간 변화",
            "숫자 퍼짐",
            "숫자 관계",
            "글자 보기",
            "특별 그래프",
            "요약 표",
            "퀴즈",
        ]
    )

    with tabs[0]:
        visualization_tab_basic(df, category_cols, numeric_cols)
    with tabs[1]:
        visualization_tab_time(df, datetime_cols, numeric_cols)
    with tabs[2]:
        visualization_tab_distribution(df, numeric_cols)
    with tabs[3]:
        visualization_tab_relation(df, numeric_cols)
    with tabs[4]:
        visualization_tab_text(df, text_cols)
    with tabs[5]:
        visualization_tab_advanced(df, category_cols, numeric_cols)
    with tabs[6]:
        visualization_tab_table(df, numeric_cols)
    with tabs[7]:
        visualization_tab_quiz(df, category_cols, numeric_cols)


if __name__ == "__main__":
    main()
