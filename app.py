from pathlib import Path

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
    page_title="株価チャート学習",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 画面全体のCSS
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

    h1 {
        font-size: 1.65rem !important;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border-radius: 10px;
        padding: 8px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 基本設定
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
WATCHLIST_PATH = BASE_DIR / "assets" / "watchlist.csv"

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

PERIOD_OPTIONS = {
    "3か月": "3mo",
    "6か月": "6mo",
    "1年": "1y",
    "2年": "2y",
    "5年": "5y",
}


# =========================================================
# 文字列の空白を正規化
# =========================================================
def normalize_text(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .replace("\u3000", " ")
        .replace("\u00a0", " ")
        .strip()
    )


# =========================================================
# enabled列をTrue/Falseへ変換
# =========================================================
def normalize_enabled(value):
    text = normalize_text(value).lower()

    true_values = {
        "true",
        "1",
        "yes",
        "y",
        "on",
        "有効",
    }

    return text in true_values


# =========================================================
# 銘柄マスターCSV読込
# =========================================================
@st.cache_data
def load_watchlist(csv_path_string):
    csv_path = Path(csv_path_string)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"銘柄マスターが見つかりません：{csv_path}"
        )

    master_df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        dtype=str,
    )

    master_df.columns = [
        normalize_text(column)
        for column in master_df.columns
    ]

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

    for column in master_df.columns:
        master_df[column] = master_df[column].map(normalize_text)

    master_df["enabled"] = master_df["enabled"].map(normalize_enabled)

    master_df["display_order"] = pd.to_numeric(
        master_df["display_order"],
        errors="coerce",
    )

    master_df = master_df.sort_values(
        by="display_order",
        na_position="last",
    ).reset_index(drop=True)

    return master_df


# =========================================================
# yfinanceの複数階層列を通常列へ変換
# =========================================================
def flatten_yfinance_columns(data, ticker):
    if not isinstance(data.columns, pd.MultiIndex):
        return data

    level_zero = data.columns.get_level_values(0)
    level_one = data.columns.get_level_values(1)

    if ticker in level_one:
        data = data.xs(
            ticker,
            axis=1,
            level=1,
            drop_level=True,
        )
    elif ticker in level_zero:
        data = data.xs(
            ticker,
            axis=1,
            level=0,
            drop_level=True,
        )
    else:
        data.columns = data.columns.get_level_values(0)

    return data


# =========================================================
# 株価データ取得・検証・正規化
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def download_and_prepare_data(provider_symbol, period):
    raw_data = yf.download(
        tickers=provider_symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if raw_data is None or raw_data.empty:
        return pd.DataFrame(), {
            "original_count": 0,
            "normalized_count": 0,
            "removed_count": 0,
        }

    data = raw_data.copy()
    data = flatten_yfinance_columns(data, provider_symbol)

    rename_map = {}

    for column in data.columns:
        normalized_column = normalize_text(column).lower()

        if normalized_column == "open":
            rename_map[column] = "Open"
        elif normalized_column == "high":
            rename_map[column] = "High"
        elif normalized_column == "low":
            rename_map[column] = "Low"
        elif normalized_column == "close":
            rename_map[column] = "Close"
        elif normalized_column == "adj close":
            rename_map[column] = "Adj Close"
        elif normalized_column == "volume":
            rename_map[column] = "Volume"

    data = data.rename(columns=rename_map)

    required_price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_price_columns = [
        column
        for column in required_price_columns
        if column not in data.columns
    ]

    if missing_price_columns:
        raise ValueError(
            "取得データに必要な列がありません："
            + ", ".join(missing_price_columns)
        )

    original_count = len(data)

    data = data[required_price_columns].copy()

    for column in required_price_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()

    try:
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
    except AttributeError:
        pass

    # OHLCの整合性確認
    valid_ohlc = (
        (data["Open"] > 0)
        & (data["High"] > 0)
        & (data["Low"] > 0)
        & (data["Close"] > 0)
        & (data["High"] >= data["Open"])
        & (data["High"] >= data["Close"])
        & (data["High"] >= data["Low"])
        & (data["Low"] <= data["Open"])
        & (data["Low"] <= data["Close"])
    )

    data = data.loc[valid_ohlc].copy()

    data["Volume"] = data["Volume"].fillna(0)
    data.loc[data["Volume"] < 0, "Volume"] = 0

    # 移動平均線
    data["SMA20"] = data["Close"].rolling(
        window=20,
        min_periods=20,
    ).mean()

    data["SMA50"] = data["Close"].rolling(
        window=50,
        min_periods=50,
    ).mean()

    # RSI（14日）
    price_change = data["Close"].diff()

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    average_gain = gain.rolling(
        window=14,
        min_periods=14,
    ).mean()

    average_loss = loss.rolling(
        window=14,
        min_periods=14,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    data["RSI14"] = (
        100
        - (100 / (1 + relative_strength))
    )

    data.loc[
        (average_loss == 0) & (average_gain > 0),
        "RSI14",
    ] = 100

    data.loc[
        (average_loss == 0) & (average_gain == 0),
        "RSI14",
    ] = 50

    normalized_count = len(data)

    validation_result = {
        "original_count": original_count,
        "normalized_count": normalized_count,
        "removed_count": original_count - normalized_count,
    }

    return data, validation_result


# =========================================================
# 出来高の色
# =========================================================
def create_volume_colors(data):
    return np.where(
        data["Close"] >= data["Open"],
        "rgba(239, 83, 80, 0.65)",
        "rgba(38, 166, 154, 0.65)",
    )


# =========================================================
# Plotlyチャート作成
# =========================================================
def create_stock_chart(data, display_name, symbol, currency):
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.58, 0.20, 0.22],
        subplot_titles=(
            "ローソク足・移動平均線",
            "出来高",
            "RSI（14日）",
        ),
    )

    # ローソク足
    figure.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="株価",
            increasing_line_color="#ef5350",
            increasing_fillcolor="#ef5350",
            decreasing_line_color="#26a69a",
            decreasing_fillcolor="#26a69a",
        ),
        row=1,
        col=1,
    )

    # 20日移動平均線
    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA20"],
            mode="lines",
            name="20日移動平均",
            line=dict(
                color="#ff9800",
                width=1.7,
            ),
        ),
        row=1,
        col=1,
    )

    # 50日移動平均線
    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA50"],
            mode="lines",
            name="50日移動平均",
            line=dict(
                color="#2962ff",
                width=1.7,
            ),
        ),
        row=1,
        col=1,
    )

    # 出来高
    figure.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            name="出来高",
            marker_color=create_volume_colors(data),
        ),
        row=2,
        col=1,
    )

    # RSI
    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI14"],
            mode="lines",
            name="RSI（14日）",
            line=dict(
                color="#7e57c2",
                width=1.8,
            ),
        ),
        row=3,
        col=1,
    )

    # RSI基準線
    figure.add_hline(
        y=70,
        line_dash="dash",
        line_color="#ef5350",
        line_width=1,
        row=3,
        col=1,
    )

    figure.add_hline(
        y=30,
        line_dash="dash",
        line_color="#26a69a",
        line_width=1,
        row=3,
        col=1,
    )

    figure.update_layout(
        title=dict(
            text=f"{display_name}（{symbol}）",
            x=0.01,
            xanchor="left",
            font=dict(size=18),
        ),
        height=760,
        margin=dict(
            l=48,
            r=18,
            t=75,
            b=40,
        ),
        template="plotly_white",
        dragmode="pan",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        modebar=dict(
            orientation="h",
            bgcolor="rgba(255,255,255,0.85)",
        ),
        uirevision=f"{symbol}-mobile-chart",
    )

    figure.update_xaxes(
        rangeslider_visible=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        showline=True,
        fixedrange=False,
    )

    figure.update_yaxes(
        title_text=f"価格（{currency}）",
        fixedrange=False,
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="出来高",
        fixedrange=False,
        row=2,
        col=1,
    )

    figure.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        fixedrange=True,
        row=3,
        col=1,
    )

    return figure


# =========================================================
# 2本指ピンチ対応JavaScript
# =========================================================
PINCH_ZOOM_SCRIPT = r"""
(function () {
    const graph = document.getElementById('{plot_id}');

    if (!graph) {
        return;
    }

    graph.style.touchAction = 'none';
    graph.style.webkitUserSelect = 'none';
    graph.style.userSelect = 'none';
    graph.style.webkitTouchCallout = 'none';

    const style = document.createElement('style');

    style.textContent = `
        html, body {
            margin: 0;
            padding: 0;
            overscroll-behavior: none;
        }

        .plotly-graph-div,
        .plot-container,
        .svg-container,
        .main-svg {
            touch-action: none !important;
        }
    `;

    document.head.appendChild(style);

    let pinchState = null;
    let animationFrame = null;
    let pendingRange = null;

    function touchDistance(touch1, touch2) {
        const dx = touch2.clientX - touch1.clientX;
        const dy = touch2.clientY - touch1.clientY;

        return Math.sqrt((dx * dx) + (dy * dy));
    }

    function touchCenterX(touch1, touch2) {
        return (touch1.clientX + touch2.clientX) / 2;
    }

    function dateToMilliseconds(value) {
        if (value instanceof Date) {
            return value.getTime();
        }

        if (typeof value === 'number') {
            return value;
        }

        return Date.parse(value);
    }

    function clamp(value, minimum, maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    function getXAxisInformation() {
        if (
            !graph._fullLayout ||
            !graph._fullLayout.xaxis
        ) {
            return null;
        }

        const xAxis = graph._fullLayout.xaxis;
        const graphRectangle = graph.getBoundingClientRect();

        const axisLeft =
            graphRectangle.left
            + (xAxis._offset || 0);

        const axisWidth =
            xAxis._length
            || graphRectangle.width;

        const currentRange = xAxis.range;

        if (
            !currentRange ||
            currentRange.length < 2
        ) {
            return null;
        }

        const rangeStart =
            dateToMilliseconds(currentRange[0]);

        const rangeEnd =
            dateToMilliseconds(currentRange[1]);

        if (
            !Number.isFinite(rangeStart) ||
            !Number.isFinite(rangeEnd) ||
            rangeEnd <= rangeStart
        ) {
            return null;
        }

        return {
            axisLeft: axisLeft,
            axisWidth: axisWidth,
            rangeStart: rangeStart,
            rangeEnd: rangeEnd
        };
    }

    function applyRange(rangeStart, rangeEnd) {
        const update = {};

        Object.keys(graph._fullLayout).forEach(function (key) {
            if (!/^xaxis\d*$/.test(key)) {
                return;
            }

            const axisObject = graph._fullLayout[key];

            if (!axisObject || axisObject.type !== 'date') {
                return;
            }

            update[key + '.range[0]'] =
                new Date(rangeStart).toISOString();

            update[key + '.range[1]'] =
                new Date(rangeEnd).toISOString();
        });

        Plotly.relayout(graph, update);
    }

    graph.addEventListener(
        'touchstart',
        function (event) {
            if (event.touches.length !== 2) {
                pinchState = null;
                return;
            }

            const axisInformation =
                getXAxisInformation();

            if (!axisInformation) {
                return;
            }

            event.preventDefault();
            event.stopPropagation();

            const firstTouch = event.touches[0];
            const secondTouch = event.touches[1];

            const initialDistance =
                touchDistance(firstTouch, secondTouch);

            const initialCenterX =
                touchCenterX(firstTouch, secondTouch);

            const initialFraction = clamp(
                (
                    initialCenterX
                    - axisInformation.axisLeft
                ) / axisInformation.axisWidth,
                0,
                1
            );

            const initialSpan =
                axisInformation.rangeEnd
                - axisInformation.rangeStart;

            const anchorTime =
                axisInformation.rangeStart
                + (initialFraction * initialSpan);

            pinchState = {
                initialDistance: initialDistance,
                initialSpan: initialSpan,
                anchorTime: anchorTime,
                axisLeft: axisInformation.axisLeft,
                axisWidth: axisInformation.axisWidth
            };
        },
        {
            passive: false,
            capture: true
        }
    );

    graph.addEventListener(
        'touchmove',
        function (event) {
            if (
                event.touches.length !== 2 ||
                !pinchState
            ) {
                return;
            }

            event.preventDefault();
            event.stopPropagation();

            const firstTouch = event.touches[0];
            const secondTouch = event.touches[1];

            const currentDistance =
                touchDistance(firstTouch, secondTouch);

            if (
                !Number.isFinite(currentDistance) ||
                currentDistance <= 0
            ) {
                return;
            }

            const zoomRatio =
                currentDistance
                / pinchState.initialDistance;

            let newSpan =
                pinchState.initialSpan
                / zoomRatio;

            // 1日より小さくなりすぎないように制限
            const minimumSpan =
                24 * 60 * 60 * 1000;

            // 初期表示幅の20倍を上限にする
            const maximumSpan =
                pinchState.initialSpan * 20;

            newSpan = clamp(
                newSpan,
                minimumSpan,
                maximumSpan
            );

            const currentCenterX =
                touchCenterX(firstTouch, secondTouch);

            const currentFraction = clamp(
                (
                    currentCenterX
                    - pinchState.axisLeft
                ) / pinchState.axisWidth,
                0,
                1
            );

            const newRangeStart =
                pinchState.anchorTime
                - (currentFraction * newSpan);

            const newRangeEnd =
                newRangeStart + newSpan;

            pendingRange = [
                newRangeStart,
                newRangeEnd
            ];

            if (animationFrame !== null) {
                return;
            }

            animationFrame =
                window.requestAnimationFrame(
                    function () {
                        if (pendingRange) {
                            applyRange(
                                pendingRange[0],
                                pendingRange[1]
                            );
                        }

                        pendingRange = null;
                        animationFrame = null;
                    }
                );
        },
        {
            passive: false,
            capture: true
        }
    );

    graph.addEventListener(
        'touchend',
        function (event) {
            if (event.touches.length < 2) {
                pinchState = null;
            }
        },
        {
            passive: true,
            capture: true
        }
    );

    graph.addEventListener(
        'touchcancel',
        function () {
            pinchState = null;
        },
        {
            passive: true,
            capture: true
        }
    );

    // iPhone Safari独自のページ拡大操作を抑止
    document.addEventListener(
        'gesturestart',
        function (event) {
            event.preventDefault();
        },
        {
            passive: false
        }
    );

    document.addEventListener(
        'gesturechange',
        function (event) {
            event.preventDefault();
        },
        {
            passive: false
        }
    );

    document.addEventListener(
        'gestureend',
        function (event) {
            event.preventDefault();
        },
        {
            passive: false
        }
    );
})();
"""


# =========================================================
# 2本指対応チャート表示
# =========================================================
def display_touch_chart(figure):
    plotly_config = {
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": True,
        "doubleClick": "reset",
        "showTips": True,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
        ],
        "modeBarButtonsToAdd": [
            "resetScale2d",
        ],
    }

    chart_html = figure.to_html(
        full_html=True,
        include_plotlyjs=True,
        config=plotly_config,
        post_script=PINCH_ZOOM_SCRIPT,
    )

    components.html(
        chart_html,
        height=780,
        scrolling=False,
    )


# =========================================================
# メイン処理
# =========================================================
def main():
    st.title("📈 株価チャート学習")

    st.caption(
        "ローソク足・移動平均線・出来高・RSIを表示します。"
        "チャート上では1本指移動と2本指拡大・縮小に対応しています。"
    )

    # -----------------------------------------------------
    # CSV読込
    # -----------------------------------------------------
    try:
        master_df = load_watchlist(str(WATCHLIST_PATH))
    except FileNotFoundError as error:
        st.error(str(error))
        st.info(
            "app.pyと同じ場所にassetsフォルダを作成し、"
            "その中へwatchlist.csvを配置してください。"
        )
        st.stop()
    except Exception as error:
        st.error(f"銘柄マスターの読込に失敗しました：{error}")
        st.stop()

    target_master = master_df.loc[
        master_df["enabled"] == True
    ].copy()

    excluded_master = master_df.loc[
        master_df["enabled"] == False
    ].copy()

    if target_master.empty:
        st.warning(
            "enabled=trueの処理対象銘柄がありません。"
        )
        st.stop()

    st.success(
        f"確認完了：処理対象{len(target_master)}件、"
        f"処理対象外{len(excluded_master)}件です。"
    )

    # -----------------------------------------------------
    # 選択欄
    # -----------------------------------------------------
    symbol_labels = {}

    for row_index, row in target_master.iterrows():
        label = (
            f"{row['display_order']}. "
            f"{row['name']}（{row['symbol']}）"
        )

        symbol_labels[label] = row_index

    selection_column, period_column = st.columns([2, 1])

    with selection_column:
        selected_label = st.selectbox(
            "表示する銘柄",
            options=list(symbol_labels.keys()),
        )

    with period_column:
        selected_period_label = st.selectbox(
            "表示期間",
            options=list(PERIOD_OPTIONS.keys()),
            index=2,
        )

    selected_row_index = symbol_labels[selected_label]
    selected_row = target_master.loc[selected_row_index]

    display_name = selected_row["name"]
    display_symbol = selected_row["symbol"]
    provider_symbol = selected_row["provider_symbol"]
    currency = selected_row["currency"]
    period = PERIOD_OPTIONS[selected_period_label]

    # -----------------------------------------------------
    # 株価取得
    # -----------------------------------------------------
    with st.spinner(
        f"{display_name}の株価データを取得しています..."
    ):
        try:
            stock_data, validation_result = (
                download_and_prepare_data(
                    provider_symbol=provider_symbol,
                    period=period,
                )
            )
        except Exception as error:
            st.error(
                f"株価データの取得・整形に失敗しました：{error}"
            )
            st.stop()

    if stock_data.empty:
        st.warning(
            f"{display_name}（{display_symbol}）の"
            "株価データを取得できませんでした。"
        )
        st.stop()

    # -----------------------------------------------------
    # 最新値表示
    # -----------------------------------------------------
    latest_row = stock_data.iloc[-1]
    latest_close = float(latest_row["Close"])

    if len(stock_data) >= 2:
        previous_close = float(
            stock_data["Close"].iloc[-2]
        )

        price_change = latest_close - previous_close

        if previous_close != 0:
            price_change_percent = (
                price_change / previous_close
            ) * 100
        else:
            price_change_percent = 0
    else:
        price_change = 0
        price_change_percent = 0

    latest_date = stock_data.index[-1].strftime(
        "%Y-%m-%d"
    )

    metric_column1, metric_column2, metric_column3 = (
        st.columns(3)
    )

    with metric_column1:
        st.metric(
            label="終値",
            value=f"{latest_close:,.2f} {currency}",
            delta=(
                f"{price_change:+,.2f} "
                f"({price_change_percent:+.2f}%)"
            ),
        )

    with metric_column2:
        st.metric(
            label="データ最終日",
            value=latest_date,
        )

    with metric_column3:
        st.metric(
            label="有効データ件数",
            value=f"{len(stock_data):,}件",
        )

    # -----------------------------------------------------
    # 操作説明
    # -----------------------------------------------------
    with st.expander(
        "📱 iPhoneでのチャート操作方法",
        expanded=True,
    ):
        st.markdown(
            """
            - **1本指で左右へ動かす**：表示期間の移動
            - **2本指を広げる**：ピンチアウトして拡大
            - **2本指を狭める**：ピンチインして縮小
            - **2本指を同時に左右へ動かす**：拡大率を保ちながら移動
            - **ダブルタップ**：初期表示へ戻す
            - **虫眼鏡ボタン**：1本指で範囲を指定して拡大
            - **家またはリセットボタン**：表示範囲をリセット

            チャート上では、ページのスクロールよりチャート操作が
            優先されます。ページを上下へ動かす場合は、チャートの外側を
            1本指で操作してください。
            """
        )

    # -----------------------------------------------------
    # チャート表示
    # -----------------------------------------------------
    figure = create_stock_chart(
        data=stock_data,
        display_name=display_name,
        symbol=display_symbol,
        currency=currency,
    )

    display_touch_chart(figure)

    # -----------------------------------------------------
    # 検証結果
    # -----------------------------------------------------
    with st.expander(
        "取得データの検証結果",
        expanded=False,
    ):
        st.write(
            f"取得時の件数："
            f"{validation_result['original_count']}件"
        )

        st.write(
            f"正規化後の件数："
            f"{validation_result['normalized_count']}件"
        )

        st.write(
            f"検証で除外した件数："
            f"{validation_result['removed_count']}件"
        )

        st.dataframe(
            stock_data.tail(10),
            use_container_width=True,
        )

    st.caption(
        "表示データは情報提供元の更新状況により遅延する場合があり、"
        "リアルタイム価格を保証するものではありません。"
        "最新の相場・取引可能価格はmoomooでご確認ください。"
        "本画面は学習・参考用であり、投資判断を目的とするものではありません。"
    )


# =========================================================
# アプリ実行
# =========================================================
if __name__ == "__main__":
    main()
