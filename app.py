from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


# =========================================================
# Streamlit基本設定
# =========================================================
st.set_page_config(
    page_title="株価指標・チャート確認",
    page_icon="📈",
    layout="wide",
)

st.title("📈 第10段階：指標計算・チャート表示")

st.caption(
    "移動平均線、日次騰落率、RSI、ヒストリカル・ボラティリティ、"
    "出来高移動平均を計算して表示します。"
)


# =========================================================
# 定数
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
WATCHLIST_PATH = BASE_DIR / "assets" / "watchlist.csv"

REQUIRED_MASTER_COLUMNS = [
    "display_order",
    "enabled",
    "name",
    "symbol",
    "provider_symbol",
    "market",
    "category",
    "currency",
    "note",
]

PRICE_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


# =========================================================
# 補助関数
# =========================================================
def normalize_text(value) -> str:
    """文字列の前後にある半角・全角空白を除去します。"""
    if pd.isna(value):
        return ""

    return str(value).replace("\u3000", " ").strip()


def parse_enabled(value) -> bool:
    """CSVのenabled列をTrue/Falseへ変換します。"""
    normalized = normalize_text(value).lower()

    true_values = {
        "true",
        "1",
        "yes",
        "y",
        "on",
        "有効",
    }

    return normalized in true_values


@st.cache_data(ttl=300)
def load_watchlist(csv_path: str) -> pd.DataFrame:
    """銘柄マスターCSVを読み込み、文字列を正規化します。"""
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSVファイルが見つかりません：{path}"
        )

    master_df = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
    )

    master_df.columns = [
        normalize_text(column) for column in master_df.columns
    ]

    missing_columns = [
        column
        for column in REQUIRED_MASTER_COLUMNS
        if column not in master_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSVに必要な列がありません："
            + ", ".join(missing_columns)
        )

    for column in master_df.columns:
        master_df[column] = master_df[column].map(normalize_text)

    master_df["enabled"] = master_df["enabled"].map(parse_enabled)

    master_df["display_order"] = pd.to_numeric(
        master_df["display_order"],
        errors="coerce",
    )

    master_df = master_df.sort_values(
        by="display_order",
        na_position="last",
    ).reset_index(drop=True)

    return master_df


@st.cache_data(ttl=1800, show_spinner=False)
def download_price_data(
    provider_symbol: str,
    period: str,
) -> pd.DataFrame:
    """公開データ提供元から日足価格を取得します。"""
    raw_df = yf.download(
        tickers=provider_symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    # yfinanceのバージョンによってはMultiIndexになるため平坦化
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in raw_df.columns
        ]

    raw_df = raw_df.reset_index()

    return raw_df


def normalize_price_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """価格データの日付・数値・重複・OHLCを正規化します。"""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    # 日付列名をDateに統一
    date_candidates = [
        column
        for column in df.columns
        if str(column).lower() in {"date", "datetime"}
    ]

    if not date_candidates:
        return pd.DataFrame()

    date_column = date_candidates[0]

    if date_column != "Date":
        df = df.rename(columns={date_column: "Date"})

    # 必要列の存在確認
    missing_price_columns = [
        column
        for column in PRICE_COLUMNS
        if column not in df.columns
    ]

    if missing_price_columns:
        return pd.DataFrame()

    use_columns = PRICE_COLUMNS.copy()

    if "Adj Close" in df.columns:
        use_columns.append("Adj Close")

    df = df[use_columns].copy()

    # 日付を統一
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        utc=True,
    )

    df["Date"] = df["Date"].dt.tz_convert(None)

    # 価格・出来高を数値へ統一
    numeric_columns = [
        column
        for column in [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]
        if column in df.columns
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    # 日付とOHLCが欠損した行を除外
    df = df.dropna(
        subset=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    # 重複日付を除去
    df = (
        df.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    # OHLC整合性
    max_price = df[
        ["Open", "High", "Low", "Close"]
    ].max(axis=1)

    min_price = df[
        ["Open", "High", "Low", "Close"]
    ].min(axis=1)

    df["OHLC整合"] = (
        (df["High"] >= max_price)
        & (df["Low"] <= min_price)
    )

    # 異常な出来高
    df["出来高整合"] = (
        df["Volume"].isna()
        | (df["Volume"] >= 0)
    )

    # 整合性を満たす行だけを利用
    df = df[
        df["OHLC整合"] & df["出来高整合"]
    ].copy()

    return df.reset_index(drop=True)


def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Wilder方式に近い指数平滑平均を用いてRSIを計算します。"""
    price_change = close.diff()

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    relative_strength = average_gain / average_loss

    rsi = 100 - (
        100 / (1 + relative_strength)
    )

    # 下落がなく平均損失が0の場合
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100,
    )

    # 値動きがない場合
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain == 0),
        50,
    )

    return rsi


def calculate_indicators(
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """第10段階の各種指標を計算します。"""
    if price_df.empty:
        return pd.DataFrame()

    df = price_df.copy()
    close = df["Close"]

    # 日次騰落率
    df["日次騰落率_pct"] = close.pct_change() * 100

    # 単純移動平均
    df["SMA_20"] = close.rolling(
        window=20,
        min_periods=20,
    ).mean()

    df["SMA_50"] = close.rolling(
        window=50,
        min_periods=50,
    ).mean()

    # RSI
    df["RSI_14"] = calculate_rsi(
        close=close,
        period=14,
    )

    # 20営業日ヒストリカル・ボラティリティ
    # 日次リターンの標準偏差を営業日252日で年率換算
    daily_return = close.pct_change()

    df["ボラティリティ_20日_pct"] = (
        daily_return
        .rolling(window=20, min_periods=20)
        .std()
        * np.sqrt(252)
        * 100
    )

    # 出来高20日移動平均
    df["出来高_SMA_20"] = (
        df["Volume"]
        .rolling(window=20, min_periods=20)
        .mean()
    )

    return df


def create_chart(
    indicator_df: pd.DataFrame,
    display_symbol: str,
    name: str,
) -> go.Figure:
    """ローソク足・移動平均・出来高・RSIを作成します。"""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.20, 0.22],
        subplot_titles=(
            "価格・移動平均線",
            "出来高",
            "RSI（14日）",
        ),
    )

    # ローソク足
    fig.add_trace(
        go.Candlestick(
            x=indicator_df["Date"],
            open=indicator_df["Open"],
            high=indicator_df["High"],
            low=indicator_df["Low"],
            close=indicator_df["Close"],
            name="ローソク足",
            increasing_line_color="#e53935",
            decreasing_line_color="#1e88e5",
        ),
        row=1,
        col=1,
    )

    # 20日移動平均
    fig.add_trace(
        go.Scatter(
            x=indicator_df["Date"],
            y=indicator_df["SMA_20"],
            mode="lines",
            name="SMA 20日",
            line=dict(
                color="#ff9800",
                width=1.8,
            ),
        ),
        row=1,
        col=1,
    )

    # 50日移動平均
    fig.add_trace(
        go.Scatter(
            x=indicator_df["Date"],
            y=indicator_df["SMA_50"],
            mode="lines",
            name="SMA 50日",
            line=dict(
                color="#7b1fa2",
                width=1.8,
            ),
        ),
        row=1,
        col=1,
    )

    # 出来高
    volume_colors = np.where(
        indicator_df["Close"]
        >= indicator_df["Open"],
        "#ef5350",
        "#42a5f5",
    )

    fig.add_trace(
        go.Bar(
            x=indicator_df["Date"],
            y=indicator_df["Volume"],
            name="出来高",
            marker_color=volume_colors,
            opacity=0.65,
        ),
        row=2,
        col=1,
    )

    # 出来高移動平均
    fig.add_trace(
        go.Scatter(
            x=indicator_df["Date"],
            y=indicator_df["出来高_SMA_20"],
            mode="lines",
            name="出来高SMA 20日",
            line=dict(
                color="#455a64",
                width=1.5,
            ),
        ),
        row=2,
        col=1,
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=indicator_df["Date"],
            y=indicator_df["RSI_14"],
            mode="lines",
            name="RSI 14日",
            line=dict(
                color="#00897b",
                width=1.8,
            ),
        ),
        row=3,
        col=1,
    )

    # RSI参考線
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="#e53935",
        line_width=1,
        row=3,
        col=1,
    )

    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="#1e88e5",
        line_width=1,
        row=3,
        col=1,
    )

    fig.update_yaxes(
        title_text="価格",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="出来高",
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        row=3,
        col=1,
    )

    fig.update_layout(
        title=f"{name}（{display_symbol}）",
        height=850,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(
            l=20,
            r=20,
            t=100,
            b=20,
        ),
    )

    return fig


def format_number(value, decimal_places=2) -> str:
    """欠損値を考慮した数値表示です。"""
    if pd.isna(value):
        return "計算期間不足"

    return f"{value:,.{decimal_places}f}"


# =========================================================
# 1. 銘柄マスター読み込み
# =========================================================
try:
    master_df = load_watchlist(
        str(WATCHLIST_PATH)
    )

except Exception as error:
    st.error("銘柄マスターを読み込めませんでした。")
    st.exception(error)
    st.stop()


target_master = master_df[
    master_df["enabled"] == True
].copy()

excluded_master = master_df[
    master_df["enabled"] == False
].copy()


if target_master.empty:
    st.error(
        "enabled=trueの処理対象銘柄がありません。"
    )
    st.stop()


st.success(
    f"確認完了：処理対象{len(target_master)}件、"
    f"処理対象外{len(excluded_master)}件です。"
)


with st.expander("処理対象・処理対象外を確認する"):
    st.subheader("処理対象")

    st.dataframe(
        target_master[
            [
                "display_order",
                "name",
                "symbol",
                "provider_symbol",
                "market",
                "currency",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("処理対象外")

    if excluded_master.empty:
        st.info("処理対象外の銘柄はありません。")
    else:
        st.dataframe(
            excluded_master[
                [
                    "display_order",
                    "name",
                    "symbol",
                    "provider_symbol",
                    "market",
                    "currency",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# 2. 表示条件
# =========================================================
st.subheader("表示条件")

period_options = {
    "6か月": "6mo",
    "1年": "1y",
    "2年": "2y",
    "5年": "5y",
}

period_label = st.selectbox(
    "価格データの取得期間",
    options=list(period_options.keys()),
    index=1,
)

selected_period = period_options[period_label]

selection_labels = {}

for _, row in target_master.iterrows():
    label = f"{row['name']}（{row['symbol']}）"
    selection_labels[label] = row["symbol"]

selected_label = st.selectbox(
    "チャートを表示する銘柄",
    options=list(selection_labels.keys()),
)

selected_symbol = selection_labels[selected_label]

selected_row = target_master[
    target_master["symbol"] == selected_symbol
].iloc[0]

selected_name = selected_row["name"]
provider_symbol = selected_row["provider_symbol"]
currency = selected_row["currency"]


# =========================================================
# 3. 価格取得・正規化・指標計算
# =========================================================
with st.spinner(
    f"{selected_name}（{selected_symbol}）の"
    "価格データを処理しています..."
):
    raw_price_df = download_price_data(
        provider_symbol=provider_symbol,
        period=selected_period,
    )

    normalized_df = normalize_price_data(
        raw_price_df
    )

    indicator_df = calculate_indicators(
        normalized_df
    )


if indicator_df.empty:
    st.error(
        f"{selected_name}（{selected_symbol}）の"
        "有効な価格データを取得できませんでした。"
    )

    st.info(
        "provider_symbol、通信状況、データ提供元の"
        "取扱状況を確認してください。"
    )

    st.stop()


# =========================================================
# 4. 最新計算値
# =========================================================
latest = indicator_df.iloc[-1]
previous = (
    indicator_df.iloc[-2]
    if len(indicator_df) >= 2
    else None
)

latest_close = latest["Close"]

if previous is not None:
    close_change = (
        latest_close - previous["Close"]
    )

    close_change_pct = (
        close_change / previous["Close"] * 100
        if previous["Close"] != 0
        else np.nan
    )
else:
    close_change = np.nan
    close_change_pct = np.nan


st.subheader(
    f"最新計算値：{selected_name}（{selected_symbol}）"
)

st.caption(
    f"データ上の最終日："
    f"{latest['Date'].strftime('%Y-%m-%d')} ／ "
    f"通貨：{currency}"
)

metric_columns = st.columns(4)

metric_columns[0].metric(
    label="終値",
    value=format_number(latest_close),
    delta=(
        f"{format_number(close_change)} "
        f"({format_number(close_change_pct)}%)"
        if not pd.isna(close_change)
        else None
    ),
)

metric_columns[1].metric(
    label="20日移動平均",
    value=format_number(
        latest["SMA_20"]
    ),
)

metric_columns[2].metric(
    label="RSI（14日）",
    value=format_number(
        latest["RSI_14"]
    ),
)

metric_columns[3].metric(
    label="20日ボラティリティ（年率換算）",
    value=(
        f"{format_number(latest['ボラティリティ_20日_pct'])}%"
        if not pd.isna(
            latest["ボラティリティ_20日_pct"]
        )
        else "計算期間不足"
    ),
)


# =========================================================
# 5. チャート
# =========================================================
st.subheader("テクニカルチャート")

chart = create_chart(
    indicator_df=indicator_df,
    display_symbol=selected_symbol,
    name=selected_name,
)

st.plotly_chart(
    chart,
    use_container_width=True,
)


# =========================================================
# 6. 計算結果表
# =========================================================
st.subheader("指標計算結果")

display_df = indicator_df.copy()

display_df["Date"] = display_df[
    "Date"
].dt.strftime("%Y-%m-%d")

display_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "日次騰落率_pct",
    "SMA_20",
    "SMA_50",
    "RSI_14",
    "ボラティリティ_20日_pct",
    "出来高_SMA_20",
]

display_df = display_df[
    display_columns
].copy()

st.dataframe(
    display_df.sort_values(
        by="Date",
        ascending=False,
    ),
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# 7. CSVダウンロード
# =========================================================
download_df = display_df.copy()

download_df.insert(
    0,
    "name",
    selected_name,
)

download_df.insert(
    1,
    "symbol",
    selected_symbol,
)

download_df.insert(
    2,
    "provider_symbol",
    provider_symbol,
)

csv_data = download_df.to_csv(
    index=False,
).encode("utf-8-sig")

safe_symbol = selected_symbol.replace(
    ".",
    "_",
)

st.download_button(
    label="指標計算済みCSVをダウンロード",
    data=csv_data,
    file_name=(
        f"{safe_symbol}_indicators.csv"
    ),
    mime="text/csv",
)


# =========================================================
# 8. 指標説明
# =========================================================
with st.expander("表示している指標の説明"):
    st.markdown(
        """
- **SMA 20日・50日**：終値の単純移動平均です。
- **日次騰落率**：前営業日の終値に対する変化率です。
- **RSI 14日**：直近の上昇幅と下落幅を基に、0～100で算出します。
- **20日ボラティリティ**：日次リターンの標準偏差を252営業日で年率換算した参考値です。
- **出来高SMA 20日**：出来高の20日単純移動平均です。
        """
    )


st.warning(
    "本画面の価格および指標は、データ提供元の更新時刻、"
    "市場休場日、為替、株式分割などの影響を受ける場合があります。"
    "リアルタイム価格を保証するものではありません。"
)

st.caption(
    "本内容は一般的な情報提供を目的としており、"
    "特定の取引判断を推奨するものではありません。"
)
