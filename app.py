from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from plotly.subplots import make_subplots


# =========================================================
# Streamlit基本設定
# =========================================================
st.set_page_config(
    page_title="株式学習チャート",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📈 株式学習チャート")
st.caption(
    "第12段階：処理対象銘柄の相対推移比較と、"
    "スマートフォン向けピンチ操作対応チャート"
)


# =========================================================
# スマートフォン向け画面調整
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
# 基本設定
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
# 銘柄マスターCSV読み込み
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
        .str.replace("\u3000", " ", regex=False)
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

    # 各項目の前後にある半角・全角空白を除去
    for column in master_df.columns:
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

    # yfinanceの列がMultiIndexの場合に平坦化
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)

    raw_df = raw_df.reset_index()

    date_column = None

    for candidate in [
        "Date",
        "Datetime",
        "date",
        "datetime",
    ]:
        if candidate in raw_df.columns:
            date_column = candidate
            break

    if date_column is None:
        return pd.DataFrame()

    raw_df = raw_df.rename(
        columns={
            date_column: "Date",
            "Adj Close": "Adj_Close",
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
        subset=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    raw_df = (
        raw_df
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # OHLCデータの整合性確認
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
def calculate_indicators(
    price_df: pd.DataFrame,
) -> pd.DataFrame:

    df = price_df.copy()

    # 20日・50日移動平均
    df["MA20"] = (
        df["Close"]
        .rolling(
            window=20,
            min_periods=20,
        )
        .mean()
    )

    df["MA50"] = (
        df["Close"]
        .rolling(
            window=50,
            min_periods=50,
        )
        .mean()
    )

    # RSI 14日
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

    relative_strength = (
        average_gain
        / average_loss.replace(0, np.nan)
    )

    df["RSI14"] = (
        100
        - (100 / (1 + relative_strength))
    )

    # 下落がない場合
    df.loc[
        (average_loss == 0)
        & (average_gain > 0),
        "RSI14",
    ] = 100

    # 値動きがない場合
    df.loc[
        (average_loss == 0)
        & (average_gain == 0),
        "RSI14",
    ] = 50

    df["Change"] = df["Close"].diff()
    df["Change_Pct"] = (
        df["Close"].pct_change() * 100
    )

    return df


# =========================================================
# 状態表示用関数
# =========================================================
def get_ma_status(ma20, ma50) -> str:
    if pd.isna(ma20) or pd.isna(ma50):
        return "計算期間不足"

    if ma20 > ma50:
        return "MA20 ＞ MA50"

    if ma20 < ma50:
        return "MA20 ＜ MA50"

    return "MA20 ＝ MA50"


def get_rsi_status(rsi) -> str:
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
# 最大ドローダウン計算
# =========================================================
def calculate_max_drawdown(
    close_series: pd.Series,
) -> float:

    clean_series = (
        pd.to_numeric(
            close_series,
            errors="coerce",
        )
        .dropna()
    )

    if clean_series.empty:
        return np.nan

    running_high = clean_series.cummax()

    drawdown = (
        clean_series / running_high - 1
    ) * 100

    return float(drawdown.min())


# =========================================================
# 個別銘柄チャート作成
# =========================================================
def create_individual_chart(
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
        row_heights=[
            0.58,
            0.20,
            0.22,
        ],
        subplot_titles=(
            f"{display_name}（{symbol}）",
            "出来高",
            "RSI（14日）",
        ),
    )

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
        line_color=(
            "rgba(231, 76, 60, 0.7)"
        ),
        row=3,
        col=1,
    )

    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color=(
            "rgba(52, 152, 219, 0.7)"
        ),
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
            l=12,
            r=12,
            t=75,
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
        uirevision=(
            f"individual_{symbol}"
        ),
    )

    return fig


# =========================================================
# 第12段階：相対推移比較チャート
# =========================================================
def create_relative_chart(
    stock_data: dict,
    summary_df: pd.DataFrame,
) -> go.Figure:

    fig = go.Figure()

    for _, summary_row in summary_df.iterrows():
        symbol = summary_row["銘柄コード"]
        name = summary_row["銘柄"]

        if symbol not in stock_data:
            continue

        price_df = stock_data[symbol].copy()

        price_df = price_df.loc[
            price_df["Close"].notna()
            & (price_df["Close"] > 0)
        ].copy()

        if price_df.empty:
            continue

        first_close = price_df["Close"].iloc[0]

        price_df["Relative_Value"] = (
            price_df["Close"]
            / first_close
            * 100
        )

        fig.add_trace(
            go.Scatter(
                x=price_df["Date"],
                y=price_df["Relative_Value"],
                mode="lines",
                name=f"{name}（{symbol}）",
                line=dict(width=2.2),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    "相対値：%{y:.2f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="rgba(80, 80, 80, 0.6)",
        annotation_text="開始時点＝100",
        annotation_position="bottom right",
    )

    fig.update_xaxes(
        title_text="日付",
        fixedrange=False,
        rangeslider_visible=False,
        showgrid=True,
        gridcolor="rgba(128, 128, 128, 0.15)",
    )

    fig.update_yaxes(
        title_text="相対値（開始時点＝100）",
        fixedrange=False,
        showgrid=True,
        gridcolor="rgba(128, 128, 128, 0.15)",
    )

    fig.update_layout(
        title=(
            "処理対象銘柄の相対推移"
            "（各銘柄の開始時点＝100）"
        ),
        height=520,
        hovermode="x unified",
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

# =========================================================
# スマートフォン向けピンチ操作対応表示
# =========================================================
def render_pinch_chart(
    fig: go.Figure,
    chart_key: str,
    height: int,
    xaxis_names: list,
) -> None:

    safe_key = re.sub(
        pattern=r"[^a-zA-Z0-9_]",
        repl="_",
        string=chart_key,
    )

    chart_id = f"pinch_chart_{safe_key}"

    config = {
        "responsive": True,
        "scrollZoom": True,
        "displaylogo": False,
        "displayModeBar": True,
        "doubleClick": "reset",
        "showTips": True,
        "modeBarButtonsToAdd": [
            "zoomIn2d",
            "zoomOut2d",
            "resetScale2d",
        ],
    }

    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config=config,
        div_id=chart_id,
    )

    axis_json = json.dumps(xaxis_names)

    touch_script = """
    <script>
    (function() {
        const graph = document.getElementById(
            "__CHART_ID__"
        );

        const axisNames = __AXIS_NAMES__;

        if (!graph) {
            return;
        }

        graph.style.width = "100%";
        graph.style.touchAction = "pan-y";

        let pinchActive = false;
        let startDistance = 0;
        let startCenterX = 0;
        let startMinimum = 0;
        let startMaximum = 0;
        let suppressTapUntil = 0;
        let lastTapTime = 0;

        function touchDistance(touches) {
            const dx =
                touches[0].clientX
                - touches[1].clientX;

            const dy =
                touches[0].clientY
                - touches[1].clientY;

            return Math.sqrt(
                (dx * dx) + (dy * dy)
            );
        }

        function touchCenterX(touches) {
            return (
                touches[0].clientX
                + touches[1].clientX
            ) / 2;
        }

        function dateToNumber(value) {
            if (typeof value === "number") {
                return value;
            }

            return new Date(value).getTime();
        }

        function numberToDate(value) {
            return new Date(value).toISOString();
        }

        function getXAxis() {
            if (!graph._fullLayout) {
                return null;
            }

            for (const axisName of axisNames) {
                const axis =
                    graph._fullLayout[axisName];

                if (
                    axis
                    && axis.range
                    && axis.range.length === 2
                ) {
                    return axis;
                }
            }

            return null;
        }

        graph.addEventListener(
            "touchstart",
            function(event) {
                if (event.touches.length !== 2) {
                    return;
                }

                const axis = getXAxis();

                if (!axis || !axis.range) {
                    return;
                }

                event.preventDefault();
                event.stopPropagation();

                startDistance =
                    touchDistance(event.touches);

                startCenterX =
                    touchCenterX(event.touches);

                startMinimum =
                    dateToNumber(axis.range[0]);

                startMaximum =
                    dateToNumber(axis.range[1]);

                pinchActive = true;
            },
            {
                passive: false,
                capture: true
            }
        );

        graph.addEventListener(
            "touchmove",
            function(event) {
                if (
                    !pinchActive
                    || event.touches.length !== 2
                ) {
                    return;
                }

                event.preventDefault();
                event.stopPropagation();

                const currentDistance =
                    touchDistance(event.touches);

                if (
                    startDistance <= 0
                    || currentDistance <= 0
                ) {
                    return;
                }

                const axis = getXAxis();

                if (!axis) {
                    return;
                }

                const graphRectangle =
                    graph.getBoundingClientRect();

                const axisLeft =
                    graphRectangle.left
                    + axis._offset;

                const axisWidth = axis._length;

                if (!axisWidth || axisWidth <= 0) {
                    return;
                }

                const startSpan =
                    startMaximum - startMinimum;

                const scale =
                    startDistance
                    / currentDistance;

                let newSpan = startSpan * scale;

                const minimumSpan =
                    5 * 24 * 60 * 60 * 1000;

                const maximumSpan =
                    20 * 365 * 24 * 60 * 60 * 1000;

                newSpan = Math.max(
                    minimumSpan,
                    Math.min(
                        maximumSpan,
                        newSpan
                    )
                );

                let centerRatio =
                    (
                        startCenterX
                        - axisLeft
                    ) / axisWidth;

                centerRatio = Math.max(
                    0,
                    Math.min(1, centerRatio)
                );

                const anchor =
                    startMinimum
                    + (
                        startSpan
                        * centerRatio
                    );

                const currentCenter =
                    touchCenterX(
                        event.touches
                    );

                let currentRatio =
                    (
                        currentCenter
                        - axisLeft
                    ) / axisWidth;

                currentRatio = Math.max(
                    0,
                    Math.min(1, currentRatio)
                );

                const newMinimum =
                    anchor
                    - (
                        newSpan
                        * currentRatio
                    );

                const newMaximum =
                    newMinimum + newSpan;

                const update = {};

                for (const axisName of axisNames) {
                    update[
                        axisName + ".range[0]"
                    ] = numberToDate(newMinimum);

                    update[
                        axisName + ".range[1]"
                    ] = numberToDate(newMaximum);

                    update[
                        axisName + ".autorange"
                    ] = false;
                }

                Plotly.relayout(graph, update);
            },
            {
                passive: false,
                capture: true
            }
        );

        function finishPinch(event) {
            if (
                pinchActive
                && event.touches.length < 2
            ) {
                pinchActive = false;
                suppressTapUntil =
                    Date.now() + 450;
            }
        }

        graph.addEventListener(
            "touchend",
            finishPinch,
            {
                passive: true,
                capture: true
            }
        );

        graph.addEventListener(
            "touchcancel",
            finishPinch,
            {
                passive: true,
                capture: true
            }
        );

        // ダブルタップで表示範囲をリセット
        graph.addEventListener(
            "touchend",
            function(event) {
                const currentTime = Date.now();

                if (
                    event.touches.length !== 0
                    || currentTime
                        < suppressTapUntil
                ) {
                    return;
                }

                if (
                    currentTime
                    - lastTapTime
                    < 350
                ) {
                    const update = {};

                    for (
                        const axisName
                        of axisNames
                    ) {
                        update[
                            axisName
                            + ".autorange"
                        ] = true;
                    }

                    Plotly.relayout(
                        graph,
                        update
                    );

                    lastTapTime = 0;
                } else {
                    lastTapTime = currentTime;
                }
            },
            {
                passive: true
            }
        );

        window.addEventListener(
            "resize",
            function() {
                if (
                    graph
                    && window.Plotly
                ) {
                    Plotly.Plots.resize(graph);
                }
            }
        );
    })();
    </script>
    """

    touch_script = (
        touch_script
        .replace("__CHART_ID__", chart_id)
        .replace("__AXIS_NAMES__", axis_json)
    )

    complete_html = (
        """
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta
                name="viewport"
                content="
                    width=device-width,
                    initial-scale=1.0,
                    maximum-scale=5.0,
                    user-scalable=yes
                "
            >
            <style>
                html, body {
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    overflow: hidden;
                    background-color: white;
                }

                .plotly-graph-div {
                    width: 100% !important;
                    touch-action: pan-y;
                    -webkit-user-select: none;
                    user-select: none;
                    -webkit-tap-highlight-color:
                        transparent;
                }
            </style>
        </head>
        <body>
        """
        + plot_html
        + touch_script
        + """
        </body>
        </html>
        """
    )

    components.html(
        complete_html,
        height=height,
        scrolling=False,
    )


# =========================================================
# 銘柄マスター読み込み
# =========================================================
try:
    master_df = load_watchlist(
        CSV_PATH.as_posix()
    )

except Exception as error:
    st.error(
        "銘柄マスターCSVを読み込めませんでした。"
    )
    st.code(str(error))
    st.stop()


target_master = master_df.loc[
    master_df["enabled_flag"]
].copy()

excluded_master = master_df.loc[
    ~master_df["enabled_flag"]
].copy()


if target_master.empty:
    st.error(
        "enabled=true の処理対象銘柄がありません。"
    )
    st.stop()


# =========================================================
# データ取得期間
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

selected_period = period_options[
    selected_period_label
]


# =========================================================
# 全処理対象銘柄のデータ取得
# =========================================================
stock_data = {}
summary_records = []
relative_records = []
error_records = []

with st.spinner(
    "処理対象銘柄のデータを取得しています…"
):
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
                        "内容": (
                            "価格データを"
                            "取得できませんでした"
                        ),
                    }
                )
                continue

            price_df = calculate_indicators(
                price_df
            )

            if price_df.empty:
                error_records.append(
                    {
                        "銘柄": name,
                        "銘柄コード": symbol,
                        "内容": (
                            "整形後のデータが"
                            "ありません"
                        ),
                    }
                )
                continue

            stock_data[symbol] = price_df

            latest = price_df.iloc[-1]
            first = price_df.iloc[0]

            period_change_pct = (
                latest["Close"]
                / first["Close"]
                - 1
            ) * 100

            max_drawdown = (
                calculate_max_drawdown(
                    price_df["Close"]
                )
            )

            summary_records.append(
                {
                    "表示順":
                        stock["display_order"],
                    "銘柄":
                        name,
                    "銘柄コード":
                        symbol,
                    "市場":
                        market,
                    "業種":
                        category,
                    "通貨":
                        currency,
                    "データ日":
                        latest["Date"].strftime(
                            "%Y-%m-%d"
                        ),
                    "終値":
                        latest["Close"],
                    "前日比":
                        latest["Change"],
                    "前日比（%）":
                        latest["Change_Pct"],
                    "MA20":
                        latest["MA20"],
                    "MA50":
                        latest["MA50"],
                    "移動平均線の状態":
                        get_ma_status(
                            latest["MA20"],
                            latest["MA50"],
                        ),
                    "RSI14":
                        latest["RSI14"],
                    "RSIの範囲":
                        get_rsi_status(
                            latest["RSI14"]
                        ),
                    "取得件数":
                        len(price_df),
                }
            )

            relative_records.append(
                {
                    "表示順":
                        stock["display_order"],
                    "銘柄":
                        name,
                    "銘柄コード":
                        symbol,
                    "通貨":
                        currency,
                    "開始日":
                        first["Date"].strftime(
                            "%Y-%m-%d"
                        ),
                    "終了日":
                        latest["Date"].strftime(
                            "%Y-%m-%d"
                        ),
                    "開始終値":
                        first["Close"],
                    "終了終値":
                        latest["Close"],
                    "期間変化率（%）":
                        period_change_pct,
                    "最大ドローダウン（%）":
                        max_drawdown,
                    "取得件数":
                        len(price_df),
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


summary_df = pd.DataFrame(
    summary_records
)

relative_summary_df = pd.DataFrame(
    relative_records
)


if summary_df.empty:
    st.error(
        "処理対象銘柄の価格データを"
        "取得できませんでした。"
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

relative_summary_df = (
    relative_summary_df
    .sort_values("表示順")
    .reset_index(drop=True)
)


# =========================================================
# 第12段階：相対推移比較
# =========================================================
st.subheader(
    "第12段階：処理対象銘柄の相対推移比較"
)

st.info(
    "各銘柄の取得期間の最初の終値を100として、"
    "その後の推移を表示しています。"
    "JPYとUSDなど通貨が異なる銘柄でも、"
    "価格水準ではなく変化の推移を確認できます。"
)

relative_chart = create_relative_chart(
    stock_data=stock_data,
    summary_df=summary_df,
)

st.caption(
    "比較グラフ操作：2本指を広げると拡大、"
    "閉じると縮小します。"
    "1本指で横移動、ダブルタップでリセットできます。"
)

render_pinch_chart(
    fig=relative_chart,
    chart_key=(
        f"relative_{selected_period}"
    ),
    height=540,
    xaxis_names=["xaxis"],
)


# =========================================================
# 相対推移の集計表
# =========================================================
st.subheader("取得期間内の変化確認")

relative_display_df = relative_summary_df[
    [
        "銘柄",
        "銘柄コード",
        "通貨",
        "開始日",
        "終了日",
        "開始終値",
        "終了終値",
        "期間変化率（%）",
        "最大ドローダウン（%）",
        "取得件数",
    ]
].copy()

relative_numeric_columns = [
    "開始終値",
    "終了終値",
    "期間変化率（%）",
    "最大ドローダウン（%）",
]

for column in relative_numeric_columns:
    relative_display_df[column] = (
        relative_display_df[column].round(2)
    )

st.dataframe(
    relative_display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "開始終値":
            st.column_config.NumberColumn(
                "開始終値",
                format="%.2f",
            ),
        "終了終値":
            st.column_config.NumberColumn(
                "終了終値",
                format="%.2f",
            ),
        "期間変化率（%）":
            st.column_config.NumberColumn(
                "期間変化率（%）",
                format="%.2f%%",
            ),
        "最大ドローダウン（%）":
            st.column_config.NumberColumn(
                "最大ドローダウン（%）",
                format="%.2f%%",
            ),
    },
)

relative_csv = relative_display_df.to_csv(
    index=False,
    encoding="utf-8-sig",
)

st.download_button(
    label="相対推移の集計表をCSVで保存",
    data=relative_csv,
    file_name="relative_performance_summary.csv",
    mime="text/csv",
    use_container_width=True,
)


# =========================================================
# 第11段階：最新値比較サマリー
# =========================================================
st.divider()
st.subheader(
    "処理対象銘柄の最新値サマリー"
)

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

summary_numeric_columns = [
    "終値",
    "前日比（%）",
    "MA20",
    "MA50",
    "RSI14",
]

for column in summary_numeric_columns:
    summary_display_df[column] = (
        summary_display_df[column].round(2)
    )

st.dataframe(
    summary_display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "終値":
            st.column_config.NumberColumn(
                "終値",
                format="%.2f",
            ),
        "前日比（%）":
            st.column_config.NumberColumn(
                "前日比（%）",
                format="%.2f%%",
            ),
        "MA20":
            st.column_config.NumberColumn(
                "MA20",
                format="%.2f",
            ),
        "MA50":
            st.column_config.NumberColumn(
                "MA50",
                format="%.2f",
            ),
        "RSI14":
            st.column_config.NumberColumn(
                "RSI14",
                format="%.2f",
            ),
    },
)

summary_csv = summary_display_df.to_csv(
    index=False,
    encoding="utf-8-sig",
)

st.download_button(
    label="最新値サマリーをCSVで保存",
    data=summary_csv,
    file_name="stock_summary.csv",
    mime="text/csv",
    use_container_width=True,
)


# =========================================================
# 個別銘柄の選択
# =========================================================
st.divider()
st.subheader("個別銘柄の確認")

available_symbols = (
    summary_df["銘柄コード"].tolist()
)

label_map = {
    row["銘柄コード"]:
        f'{row["銘柄"]}（{row["銘柄コード"]}）'
    for _, row in summary_df.iterrows()
}

selected_symbol = st.selectbox(
    "表示する銘柄",
    options=available_symbols,
    format_func=lambda value:
        label_map.get(value, value),
)

selected_summary = summary_df.loc[
    summary_df["銘柄コード"]
    == selected_symbol
].iloc[0]

selected_price_df = stock_data[
    selected_symbol
]

selected_name = selected_summary["銘柄"]
selected_currency = selected_summary["通貨"]


# =========================================================
# 個別銘柄の最新値
# =========================================================
column1, column2 = st.columns(2)

with column1:
    st.metric(
        label=f"終値（{selected_currency}）",
        value=format_number(
            selected_summary["終値"]
        ),
        delta=(
            f"{format_number(selected_summary['前日比（%）'])}%"
            if not pd.isna(selected_summary["前日比（%）"])
            else None
        ),
    )

with column2:
    st.metric(
        label="RSI（14日）",
        value=format_number(
            selected_summary["RSI14"]
        ),
    )

column3, column4 = st.columns(2)

with column3:
    st.metric(
        label="20日移動平均",
        value=format_number(
            selected_summary["MA20"]
        ),
    )

with column4:
    st.metric(
        label="50日移動平均",
        value=format_number(
            selected_summary["MA50"]
        ),
    )

st.info(
    f'データ日：'
    f'{selected_summary["データ日"]}　｜　'
    f'移動平均線：'
    f'{selected_summary["移動平均線の状態"]}　｜　'
    f'RSI範囲：'
    f'{selected_summary["RSIの範囲"]}'
)


# =========================================================
# 個別銘柄チャート
# =========================================================
st.caption(
    "個別グラフ操作：2本指を広げると拡大、"
    "閉じると縮小します。"
    "1本指で横移動、ダブルタップでリセットできます。"
)

individual_chart = create_individual_chart(
    price_df=selected_price_df,
    display_name=selected_name,
    symbol=selected_symbol,
    currency=selected_currency,
)

render_pinch_chart(
    fig=individual_chart,
    chart_key=(
        f"individual_"
        f"{selected_symbol}_"
        f"{selected_period}"
    ),
    height=820,
    xaxis_names=[
        "xaxis",
        "xaxis2",
        "xaxis3",
    ],
)


# =========================================================
# 個別データCSV
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
    label=(
        f"{selected_symbol} のデータをCSVで保存"
    ),
    data=individual_csv,
    file_name=(
        f"{selected_symbol}_price_data.csv"
    ),
    mime="text/csv",
    use_container_width=True,
)


# =========================================================
# 処理対象・対象外確認
# =========================================================
with st.expander(
    "処理対象・処理対象外を確認"
):
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


# =========================================================
# 取得エラー確認
# =========================================================
if error_records:
    with st.expander(
        "取得できなかった銘柄を確認"
    ):
        st.dataframe(
            pd.DataFrame(error_records),
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# 完了表示
# =========================================================
st.success(
    "第12段階の処理が完了しました。"
    f"正常取得：{len(summary_df)}件、"
    f"処理対象外：{len(excluded_master)}件"
)

st.caption(
    "相対値は各銘柄の取得開始時点を100として"
    "計算した参考値です。"
    "通貨換算、配当、税金、手数料、株式分割等の影響を"
    "完全に反映するものではありません。"
)

st.caption(
    "表示データはリアルタイムとは限りません。"
    "最新の相場や取引可能価格はmoomooでご確認ください。"
    "表示内容は学習・参考用であり、"
    "投資判断を構成するものではありません。"
)
