from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


# =========================================================
# Streamlit 基本設定
# =========================================================
st.set_page_config(
    page_title="株式学習チャート",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📈 株式学習チャート")
st.caption(
    "第11段階：処理対象銘柄の比較サマリー、指標確認、個別チャート表示"
)


# =========================================================
# iPhone・スマートフォン向け表示調整
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-left: 0.7rem;
        padding-right: 0.7rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border-radius: 10px;
        padding: 10px;
    }

    div[data-testid="stDataFrame"] {
        width: 100%;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.35rem;
            padding-right: 0.35rem;
        }

        h1 {
            font-size: 1.65rem !important;
        }

        h2, h3 {
            font-size: 1.2rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 設定
# =========================================================
CSV_PATH = Path("assets/watchlist.csv")

REQUIRED_COLUMNS = [
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


# =========================================================
# CSVマスター読み込み
# =========================================================
@st.cache_data
def load_watchlist(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSVファイルが見つかりません：{path.as_posix()}"
        )

    master_df = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
    )

    master_df.columns = (
        master_df.columns
        .astype(str)
        .str.strip()
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in master_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSVに必要な列がありません："
            + ", ".join(missing_columns)
        )

    # 文字列の前後にある半角・全角空白を除去
    for column in master_df.columns:
        if master_df[column].dtype == "object":
            master_df[column] = (
                master_df[column]
                .fillna("")
                .astype(str)
                .str.replace("\u3000", " ", regex=False)
                .str.strip()
            )

    master_df["display_order"] = pd.to_numeric(
        master_df["display_order"],
        errors="coerce",
    )

    enabled_text = (
        master_df["enabled"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    master_df["enabled_flag"] = enabled_text.isin(
        ["true", "1", "yes", "y", "on"]
    )

    master_df = (
        master_df
        .sort_values(
            by=["display_order", "name"],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    return master_df


# =========================================================
# 株価データ取得
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def download_stock_data(
    provider_symbol: str,
    period: str,
) -> pd.DataFrame:
    provider_symbol = str(provider_symbol).strip()

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
        raw_df.columns = raw_df.columns.get_level_values(0)

    raw_df = raw_df.reset_index()

    date_column = None

    for candidate in ["Date", "Datetime", "date", "datetime"]:
        if candidate in raw_df.columns:
            date_column = candidate
            break

    if date_column is None:
        return pd.DataFrame()

    raw_df = raw_df.rename(
        columns={
            date_column: "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "Adj_Close",
            "Volume": "Volume",
        }
    )

    required_price_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
    ]

    if any(
        column not in raw_df.columns
        for column in required_price_columns
    ):
        return pd.DataFrame()

    if "Volume" not in raw_df.columns:
        raw_df["Volume"] = 0

    raw_df["Date"] = pd.to_datetime(
        raw_df["Date"],
        errors="coerce",
    )

    # タイムゾーン情報がある場合は除去
    try:
        raw_df["Date"] = raw_df["Date"].dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in numeric_columns:
        raw_df[column] = pd.to_numeric(
            raw_df[column],
            errors="coerce",
        )

    raw_df = raw_df.dropna(
        subset=["Date", "Open", "High", "Low", "Close"]
    )

    raw_df = (
        raw_df
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # OHLCの整合性確認
    valid_ohlc = (
        (raw_df["High"] >= raw_df["Open"])
        & (raw_df["High"] >= raw_df["Close"])
        & (raw_df["High"] >= raw_df["Low"])
        & (raw_df["Low"] <= raw_df["Open"])
        & (raw_df["Low"] <= raw_df["Close"])
    )

    raw_df = raw_df.loc[valid_ohlc].copy()

    raw_df["Volume"] = (
        raw_df["Volume"]
        .fillna(0)
        .clip(lower=0)
    )

    return raw_df.reset_index(drop=True)


# =========================================================
# テクニカル指標計算
# =========================================================
def calculate_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()

    df["MA20"] = (
        df["Close"]
        .rolling(window=20, min_periods=20)
        .mean()
    )

    df["MA50"] = (
        df["Close"]
        .rolling(window=50, min_periods=50)
        .mean()
    )

    # RSI（14日）
    price_change = df["Close"].diff()

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    df["RSI14"] = 100 - (
        100 / (1 + relative_strength)
    )

    # 下落がない場合
    df.loc[
        (average_loss == 0) & (average_gain > 0),
        "RSI14",
    ] = 100

    # 値動きがない場合
    df.loc[
        (average_loss == 0) & (average_gain == 0),
        "RSI14",
    ] = 50

    # 前日比
    df["Change"] = df["Close"].diff()
    df["Change_Pct"] = df["Close"].pct_change() * 100

    return df


# =========================================================
# 状態表示用関数
# =========================================================
def get_ma_status(
    ma20: float,
    ma50: float,
) -> str:
    if pd.isna(ma20) or pd.isna(ma50):
        return "計算期間不足"

    if ma20 > ma50:
        return "MA20 ＞ MA50"

    if ma20 < ma50:
        return "MA20 ＜ MA50"

    return "MA20 ＝ MA50"


def get_rsi_status(rsi: float) -> str:
    if pd.isna(rsi):
        return "計算期間不足"

    if rsi >= 70:
        return "70以上"

    if rsi <= 30:
        return "30以下"

    return "30～70"


def format_number(
    value,
    digits: int = 2,
) -> str:
    if pd.isna(value):
        return "-"

    return f"{value:,.{digits}f}"


# =========================================================
# チャート作成
# =========================================================
def create_chart(
    price_df: pd.DataFrame,
    display_name: str,
    symbol: str,
    currency: str,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.58, 0.20, 0.22],
        subplot_titles=(
            f"{display_name}（{symbol}）",
            "出来高",
            "RSI（14日）",
        ),
    )

    # ローソク足
    fig.add_trace(
        go.Candlestick(
            x=price_df["Date"],
            open=price_df["Open"],
            high=price_df["High"],
            low=price_df["Low"],
            close=price_df["Close"],
            name="ローソク足",
            increasing_line_color="#e74c3c",
            decreasing_line_color="#3498db",
        ),
        row=1,
        col=1,
    )

    # 20日移動平均線
    fig.add_trace(
        go.Scatter(
            x=price_df["Date"],
            y=price_df["MA20"],
            mode="lines",
            name="MA20",
            line=dict(
                color="#f39c12",
                width=1.8,
            ),
        ),
        row=1,
        col=1,
    )

    # 50日移動平均線
    fig.add_trace(
        go.Scatter(
            x=price_df["Date"],
            y=price_df["MA50"],
            mode="lines",
            name="MA50",
            line=dict(
                color="#8e44ad",
                width=1.8,
            ),
        ),
        row=1,
        col=1,
    )

    # 出来高の色
    volume_colors = np.where(
        price_df["Close"] >= price_df["Open"],
        "rgba(231, 76, 60, 0.65)",
        "rgba(52, 152, 219, 0.65)",
    )

    fig.add_trace(
        go.Bar(
            x=price_df["Date"],
            y=price_df["Volume"],
            name="出来高",
            marker_color=volume_colors,
        ),
        row=2,
        col=1,
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=price_df["Date"],
            y=price_df["RSI14"],
            mode="lines",
            name="RSI14",
            line=dict(
                color="#16a085",
                width=1.8,
            ),
        ),
        row=3,
        col=1,
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="rgba(231, 76, 60, 0.7)",
        row=3,
        col=1,
    )

    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="rgba(52, 152, 219, 0.7)",
        row=3,
        col=1,
    )

    fig.update_yaxes(
        title_text=f"価格（{currency}）",
        fixedrange=False,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="出来高",
        fixedrange=False,
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        fixedrange=False,
        row=3,
        col=1,
    )

    fig.update_xaxes(
        rangeslider_visible=False,
        fixedrange=False,
    )

    fig.update_layout(
        height=800,
        margin=dict(
            l=15,
            r=15,
            t=70,
            b=20,
        ),
        hovermode="x unified",
        dragmode="pan",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
        modebar=dict(
            orientation="h",
        ),
        uirevision=symbol,
    )

    return fig


# =========================================================
# メイン処理
# =========================================================
try:
    master_df = load_watchlist(CSV_PATH.as_posix())

except Exception as error:
    st.error("銘柄マスターCSVを読み込めませんでした。")
    st.code(str(error))
    st.stop()


target_master = master_df.loc[
    master_df["enabled_flag"]
].copy()

excluded_master = master_df.loc[
    ~master_df["enabled_flag"]
].copy()


if target_master.empty:
    st.error("enabled=true の処理対象銘柄がありません。")
    st.stop()


# =========================================================
# 期間選択
# =========================================================
period_options = {
    "6か月": "6mo",
    "1年": "1y",
    "2年": "2y",
    "5年": "5y",
}

selected_period_label = st.selectbox(
    "データ取得期間",
    options=list(period_options.keys()),
    index=1,
)

selected_period = period_options[selected_period_label]


# =========================================================
# 全対象銘柄のデータ取得
# =========================================================
stock_data = {}
summary_records = []
error_records = []

with st.spinner("処理対象銘柄のデータを取得しています…"):
    for _, stock in target_master.iterrows():
        name = stock["name"]
        symbol = stock["symbol"]
        provider_symbol = stock["provider_symbol"]
        market = stock["market"]
        category = stock["category"]
        currency = stock["currency"]

        try:
            price_df = download_stock_data(
                provider_symbol=provider_symbol,
                period=selected_period,
            )

            if price_df.empty:
                error_records.append(
                    {
                        "銘柄": name,
                        "銘柄コード": symbol,
                        "内容": "価格データを取得できませんでした",
                    }
                )
                continue

            price_df = calculate_indicators(price_df)

            if price_df.empty:
                error_records.append(
                    {
                        "銘柄": name,
                        "銘柄コード": symbol,
                        "内容": "整形後のデータがありません",
                    }
                )
                continue

            stock_data[symbol] = price_df

            latest = price_df.iloc[-1]

            summary_records.append(
                {
                    "表示順": stock["display_order"],
                    "銘柄": name,
                    "銘柄コード": symbol,
                    "市場": market,
                    "業種": category,
                    "通貨": currency,
                    "データ日": latest["Date"].strftime("%Y-%m-%d"),
                    "終値": latest["Close"],
                    "前日比": latest["Change"],
                    "前日比（%）": latest["Change_Pct"],
                    "MA20": latest["MA20"],
                    "MA50": latest["MA50"],
                    "移動平均線の状態": get_ma_status(
                        latest["MA20"],
                        latest["MA50"],
                    ),
                    "RSI14": latest["RSI14"],
                    "RSIの範囲": get_rsi_status(
                        latest["RSI14"]
                    ),
                    "取得件数": len(price_df),
                }
            )

        except Exception as error:
            error_records.append(
                {
                    "銘柄": name,
                    "銘柄コード": symbol,
                    "内容": str(error),
                }
            )


summary_df = pd.DataFrame(summary_records)


if summary_df.empty:
    st.error(
        "処理対象銘柄の価格データを取得できませんでした。"
    )

    if error_records:
        st.dataframe(
            pd.DataFrame(error_records),
            use_container_width=True,
            hide_index=True,
        )

    st.stop()


summary_df = (
    summary_df
    .sort_values("表示順")
    .reset_index(drop=True)
)


# =========================================================
# 第11段階：比較サマリー
# =========================================================
st.subheader("第11段階：処理対象銘柄の比較サマリー")

summary_display_df = summary_df[
    [
        "銘柄",
        "銘柄コード",
        "データ日",
        "通貨",
        "終値",
        "前日比（%）",
        "MA20",
        "MA50",
        "移動平均線の状態",
        "RSI14",
        "RSIの範囲",
    ]
].copy()

numeric_display_columns = [
    "終値",
    "前日比（%）",
    "MA20",
    "MA50",
    "RSI14",
]

for column in numeric_display_columns:
    summary_display_df[column] = (
        summary_display_df[column]
        .round(2)
    )

st.dataframe(
    summary_display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "終値": st.column_config.NumberColumn(
            "終値",
            format="%.2f",
        ),
        "前日比（%）": st.column_config.NumberColumn(
            "前日比（%）",
            format="%.2f%%",
        ),
        "MA20": st.column_config.NumberColumn(
            "MA20",
            format="%.2f",
        ),
        "MA50": st.column_config.NumberColumn(
            "MA50",
            format="%.2f",
        ),
        "RSI14": st.column_config.NumberColumn(
            "RSI14",
            format="%.2f",
        ),
    },
)


# =========================================================
# サマリーCSV保存
# =========================================================
summary_csv = summary_display_df.to_csv(
    index=False,
    encoding="utf-8-sig",
)

st.download_button(
    label="比較サマリーをCSVで保存",
    data=summary_csv,
    file_name="stock_summary.csv",
    mime="text/csv",
    use_container_width=True,
)


# =========================================================
# 個別銘柄選択
# =========================================================
st.divider()
st.subheader("個別銘柄の確認")

available_symbols = summary_df["銘柄コード"].tolist()

label_map = {
    row["銘柄コード"]:
        f'{row["銘柄"]}（{row["銘柄コード"]}）'
    for _, row in summary_df.iterrows()
}

selected_symbol = st.selectbox(
    "表示する銘柄",
    options=available_symbols,
    format_func=lambda value: label_map.get(value, value),
)

selected_summary = summary_df.loc[
    summary_df["銘柄コード"] == selected_symbol
].iloc[0]

selected_price_df = stock_data[selected_symbol]

selected_name = selected_summary["銘柄"]
selected_currency = selected_summary["通貨"]


# =========================================================
# 最新値カード
# =========================================================
column1, column2 = st.columns(2)

with column1:
    st.metric(
        label=f"終値（{selected_currency}）",
        value=format_number(
            selected_summary["終値"],
            2,
        ),
        delta=(
            f'{format_number(selected_summary["前日比（%）"], 2)}%'
            if not pd.isna(selected_summary["前日比（%）"])
            else None
        ),
    )

with column2:
    st.metric(
        label="RSI（14日）",
        value=format_number(
            selected_summary["RSI14"],
            2,
        ),
        delta=None,
    )

column3, column4 = st.columns(2)

with column3:
    st.metric(
        label="20日移動平均",
        value=format_number(
            selected_summary["MA20"],
            2,
        ),
    )

with column4:
    st.metric(
        label="50日移動平均",
        value=format_number(
            selected_summary["MA50"],
            2,
        ),
    )

st.info(
    f'データ日：{selected_summary["データ日"]}　｜　'
    f'移動平均線：{selected_summary["移動平均線の状態"]}　｜　'
    f'RSI範囲：{selected_summary["RSIの範囲"]}'
)


# =========================================================
# 個別チャート
# =========================================================
chart = create_chart(
    price_df=selected_price_df,
    display_name=selected_name,
    symbol=selected_symbol,
    currency=selected_currency,
)

st.plotly_chart(
    chart,
    use_container_width=True,
    config={
        "responsive": True,
        "scrollZoom": True,
        "displaylogo": False,
        "displayModeBar": True,
        "doubleClick": "reset",
        "showTips": True,
    },
)


# =========================================================
# 個別データCSV保存
# =========================================================
download_df = selected_price_df.copy()

download_df["Date"] = (
    download_df["Date"]
    .dt.strftime("%Y-%m-%d")
)

individual_csv = download_df.to_csv(
    index=False,
    encoding="utf-8-sig",
)

st.download_button(
    label=f"{selected_symbol} のデータをCSVで保存",
    data=individual_csv,
    file_name=f"{selected_symbol}_price_data.csv",
    mime="text/csv",
    use_container_width=True,
)


# =========================================================
# 処理結果
# =========================================================
with st.expander("処理対象・処理対象外を確認"):
    st.write(
        f"処理対象：{len(target_master)}件"
    )

    st.dataframe(
        target_master[
            [
                "name",
                "symbol",
                "provider_symbol",
                "enabled",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.write(
        f"処理対象外：{len(excluded_master)}件"
    )

    st.dataframe(
        excluded_master[
            [
                "name",
                "symbol",
                "provider_symbol",
                "enabled",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


if error_records:
    with st.expander("取得できなかった銘柄を確認"):
        st.dataframe(
            pd.DataFrame(error_records),
            use_container_width=True,
            hide_index=True,
        )


st.success(
    f"第11段階の処理が完了しました。"
    f"正常取得：{len(summary_df)}件、"
    f"処理対象外：{len(excluded_master)}件"
)

st.caption(
    "RSIや移動平均線の表示は、過去データを機械的に計算した学習用情報です。"
    "売買判断を示すものではありません。"
    "取得データはリアルタイムとは限らないため、最新の相場はmoomooでご確認ください。"
)
