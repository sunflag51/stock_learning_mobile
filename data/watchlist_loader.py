"""
監視銘柄CSVの読み込み機能。

対応する入力元:
1. GitHubリポジトリ内のCSV
2. スマートフォンからアップロードしたCSV
3. Googleスプレッドシートの公開CSV
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import pandas as pd

from config.settings import (
    DEFAULT_WATCHLIST_PATH,
    WATCHLIST_ENCODING,
)
from data.validator import (
    DataValidationError,
    validate_watchlist,
)


MAX_CSV_DOWNLOAD_BYTES = 10 * 1024 * 1024
CSV_DOWNLOAD_TIMEOUT_SECONDS = 20


class WatchlistLoadError(RuntimeError):
    """監視銘柄CSVの読み込みに失敗した場合の例外。"""


# =========================================================
# CSV共通読込
# =========================================================

def _read_csv_bytes(csv_bytes: bytes) -> pd.DataFrame:
    """
    CSVのバイトデータをDataFrameへ変換する。
    """
    if not csv_bytes:
        raise WatchlistLoadError(
            "CSVファイルの内容が空です。"
        )

    if len(csv_bytes) > MAX_CSV_DOWNLOAD_BYTES:
        raise WatchlistLoadError(
            "CSVファイルが10MBを超えています。"
        )

    try:
        dataframe = pd.read_csv(
            BytesIO(csv_bytes),
            encoding=WATCHLIST_ENCODING,
            dtype=str,
            keep_default_na=False,
        )
    except UnicodeDecodeError as error:
        raise WatchlistLoadError(
            "CSVの文字コードを読み取れませんでした。"
            "UTF-8形式で保存してください。"
        ) from error
    except pd.errors.EmptyDataError as error:
        raise WatchlistLoadError(
            "CSVファイルに列名やデータがありません。"
        ) from error
    except pd.errors.ParserError as error:
        raise WatchlistLoadError(
            "CSVの形式を解析できませんでした。"
            "カンマ、引用符、改行を確認してください。"
        ) from error
    except Exception as error:
        raise WatchlistLoadError(
            f"CSVの読み込み中にエラーが発生しました: {error}"
        ) from error

    try:
        return validate_watchlist(dataframe)
    except DataValidationError as error:
        raise WatchlistLoadError(str(error)) from error


# =========================================================
# GitHubリポジトリ内CSV
# =========================================================

def load_watchlist_from_repository(
    csv_path: str | Path = DEFAULT_WATCHLIST_PATH,
) -> pd.DataFrame:
    """
    GitHubリポジトリ内に配置した監視銘柄CSVを読み込む。

    Streamlit Community Cloud上でも、リポジトリ内の
    assets/watchlist.csvを読み込めます。
    """
    path = Path(csv_path).expanduser().resolve()

    if not path.exists():
        raise WatchlistLoadError(
            "監視銘柄CSVが見つかりません。\n"
            f"確認先: {path}"
        )

    if not path.is_file():
        raise WatchlistLoadError(
            f"指定されたパスはファイルではありません: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise WatchlistLoadError(
            "監視銘柄ファイルには.csv形式を使用してください。"
        )

    try:
        csv_bytes = path.read_bytes()
    except OSError as error:
        raise WatchlistLoadError(
            f"監視銘柄CSVを開けませんでした: {error}"
        ) from error

    return _read_csv_bytes(csv_bytes)


# =========================================================
# スマートフォンアップロードCSV
# =========================================================

def load_watchlist_from_upload(
    uploaded_file: BinaryIO,
) -> pd.DataFrame:
    """
    Streamlitのst.file_uploaderで受け取ったCSVを読み込む。

    使用例:
        uploaded_file = st.file_uploader(
            "監視銘柄CSV",
            type=["csv"],
        )

        if uploaded_file is not None:
            watchlist = load_watchlist_from_upload(uploaded_file)
    """
    if uploaded_file is None:
        raise WatchlistLoadError(
            "アップロードされたCSVがありません。"
        )

    uploaded_name = getattr(
        uploaded_file,
        "name",
        "uploaded.csv",
    )

    if not str(uploaded_name).lower().endswith(".csv"):
        raise WatchlistLoadError(
            "アップロードできるファイルはCSVだけです。"
        )

    try:
        if hasattr(uploaded_file, "getvalue"):
            csv_bytes = uploaded_file.getvalue()
        else:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)

            csv_bytes = uploaded_file.read()
    except Exception as error:
        raise WatchlistLoadError(
            f"アップロードCSVを読み取れませんでした: {error}"
        ) from error

    if isinstance(csv_bytes, str):
        csv_bytes = csv_bytes.encode("utf-8")

    if not isinstance(csv_bytes, bytes):
        raise WatchlistLoadError(
            "アップロードCSVのデータ形式を読み取れませんでした。"
        )

    return _read_csv_bytes(csv_bytes)


# =========================================================
# Googleスプレッドシート
# =========================================================

def convert_google_sheets_url_to_csv(url: str) -> str:
    """
    GoogleスプレッドシートURLをCSV取得用URLへ変換する。

    通常の編集URLと公開URLの両方に対応します。

    例:
    https://docs.google.com/spreadsheets/d/FILE_ID/edit#gid=0

    変換後:
    https://docs.google.com/spreadsheets/d/FILE_ID/export?format=csv&gid=0
    """
    normalized_url = str(url).strip()

    if not normalized_url:
        raise WatchlistLoadError(
            "GoogleスプレッドシートURLが入力されていません。"
        )

    parsed = urlparse(normalized_url)

    if parsed.scheme != "https":
        raise WatchlistLoadError(
            "GoogleスプレッドシートURLにはhttpsを使用してください。"
        )

    if parsed.hostname != "docs.google.com":
        raise WatchlistLoadError(
            "docs.google.comのスプレッドシートURLを入力してください。"
        )

    # 通常のスプレッドシートIDを取得します。
    standard_match = re.search(
        r"/spreadsheets/d/([a-zA-Z0-9_-]+)",
        parsed.path,
    )

    if standard_match:
        spreadsheet_id = standard_match.group(1)

        query_values = parse_qs(parsed.query)
        fragment_values = parse_qs(parsed.fragment)

        gid = (
            query_values.get("gid", [None])[0]
            or fragment_values.get("gid", [None])[0]
            or "0"
        )

        return (
            "https://docs.google.com/spreadsheets/d/"
            f"{spreadsheet_id}/export?format=csv&gid={gid}"
        )

    # 「ウェブに公開」URLに対応します。
    published_match = re.search(
        r"/spreadsheets/d/e/([a-zA-Z0-9_-]+)",
        parsed.path,
    )

    if published_match:
        published_id = published_match.group(1)
        query_values = parse_qs(parsed.query)
        gid = query_values.get("gid", [None])[0]

        output_url = (
            "https://docs.google.com/spreadsheets/d/e/"
            f"{published_id}/pub?output=csv"
        )

        if gid:
            output_url += f"&gid={gid}"

        return output_url

    raise WatchlistLoadError(
        "GoogleスプレッドシートIDをURLから確認できませんでした。"
    )


def _download_csv(csv_url: str) -> bytes:
    """
    公開CSVをサイズ制限付きでダウンロードする。
    """
    request = Request(
        csv_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 stock-learning-mobile/1.0"
            ),
            "Accept": "text/csv,text/plain,*/*",
        },
    )

    try:
        with urlopen(
            request,
            timeout=CSV_DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            content_type = response.headers.get(
                "Content-Type",
                "",
            ).lower()

            csv_bytes = response.read(
                MAX_CSV_DOWNLOAD_BYTES + 1
            )

    except HTTPError as error:
        if error.code in {401, 403}:
            message = (
                "Googleスプレッドシートを読み込めませんでした。"
                "閲覧権限または公開設定を確認してください。"
            )
        elif error.code == 404:
            message = (
                "Googleスプレッドシートが見つかりません。"
                "URLを確認してください。"
            )
        else:
            message = (
                "Googleスプレッドシートの取得に失敗しました。"
                f"HTTP状態コード: {error.code}"
            )

        raise WatchlistLoadError(message) from error

    except URLError as error:
        raise WatchlistLoadError(
            "Googleスプレッドシートへ接続できませんでした。"
            "通信状態とURLを確認してください。"
        ) from error

    except TimeoutError as error:
        raise WatchlistLoadError(
            "Googleスプレッドシートの読み込みが"
            "タイムアウトしました。"
        ) from error

    except Exception as error:
        raise WatchlistLoadError(
            "Googleスプレッドシートの取得中に"
            f"エラーが発生しました: {error}"
        ) from error

    if len(csv_bytes) > MAX_CSV_DOWNLOAD_BYTES:
        raise WatchlistLoadError(
            "GoogleスプレッドシートのCSVが10MBを超えています。"
        )

    # 権限エラーなどでHTML画面が返った可能性を検出します。
    beginning = csv_bytes[:500].lower()

    if (
        "text/html" in content_type
        or b"<!doctype html" in beginning
        or b"<html" in beginning
    ):
        raise WatchlistLoadError(
            "CSVではなくHTML画面が返されました。"
            "Googleスプレッドシートをリンク閲覧可能または"
            "ウェブ公開に設定してください。"
        )

    return csv_bytes


def load_watchlist_from_google_sheets(
    google_sheets_url: str,
) -> pd.DataFrame:
    """
    Googleスプレッドシートの公開CSVを読み込む。
    """
    csv_url = convert_google_sheets_url_to_csv(
        google_sheets_url
    )

    csv_bytes = _download_csv(csv_url)

    return _read_csv_bytes(csv_bytes)


# =========================================================
# 有効銘柄抽出
# =========================================================

def get_enabled_watchlist(
    watchlist: pd.DataFrame,
) -> pd.DataFrame:
    """
    enabled=trueの銘柄だけを表示順に取り出す。
    """
    validated = validate_watchlist(watchlist)

    return (
        validated.loc[validated["enabled"]]
        .sort_values(
            by=["display_order", "name"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
