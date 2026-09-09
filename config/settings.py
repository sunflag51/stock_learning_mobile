"""
アプリ全体で共通して使用する基本設定。

このファイルには、ファイルパス、初期表示、データ取得条件など、
アプリ全体に関係する設定だけをまとめます。
"""

from pathlib import Path


# =========================================================
# ファイル・フォルダ
# =========================================================

APP_ROOT = Path(__file__).resolve().parents[1]

ASSETS_DIR = APP_ROOT / "assets"
DEFAULT_WATCHLIST_PATH = ASSETS_DIR / "watchlist.csv"


# =========================================================
# アプリ表示
# =========================================================

APP_TITLE = "株式分析・投資前確認 学習ダッシュボード"

APP_ICON = "📊"

APP_SUBTITLE = (
    "チャート分析・判断基準・確認漏れ防止を一つにまとめた"
    "スマートフォン向け学習アプリ"
)

APP_VERSION = "1.0.0"


# =========================================================
# データ取得
# =========================================================

DEFAULT_DATA_PERIOD = "2y"

DEFAULT_DATA_INTERVAL = "1d"

SUPPORTED_INTERVALS = {
    "日足": "1d",
    "週足": "1wk",
    "月足": "1mo",
}

SUPPORTED_PERIODS = {
    "6か月": "6mo",
    "1年": "1y",
    "2年": "2y",
    "5年": "5y",
    "10年": "10y",
    "全期間": "max",
}

DEFAULT_INTERVAL_LABEL = "日足"

DEFAULT_PERIOD_LABEL = "2年"

DATA_CACHE_TTL_SECONDS = 900

DATA_REQUEST_TIMEOUT_SECONDS = 20


# =========================================================
# 監視銘柄CSV
# =========================================================

WATCHLIST_REQUIRED_COLUMNS = [
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

WATCHLIST_ENCODING = "utf-8-sig"

MAX_WATCHLIST_ROWS = 500


# =========================================================
# 画面表示
# =========================================================

DEFAULT_CHART_HEIGHT = 650

MINIMUM_REQUIRED_PRICE_ROWS = 60

PRICE_DECIMAL_PLACES = 2

PERCENT_DECIMAL_PLACES = 2


# =========================================================
# 免責表示
# =========================================================

DISCLAIMER_TEXT = (
    "このアプリは、テクニカル指標の学習と確認漏れ防止を目的としています。"
    "表示内容は将来の値動きを保証するものではなく、個別の投資判断を"
    "構成するものではありません。データ最終日と最新相場を必ず確認してください。"
)
