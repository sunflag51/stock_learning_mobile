from pathlib import Path
import csv

output_path = Path("assets/watchlist.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

header = [
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

rows = [
    [1, "true", "任天堂", "7974" + "." + "JP", "7974" + "." + "T",
     "日本株", "ゲーム", "JPY", "操作確認用"],

    [2, "true", "トヨタ自動車", "7203" + "." + "JP", "7203" + "." + "T",
     "日本株", "自動車", "JPY", "操作確認用"],

    [3, "true", "Alphabet Class C", "GOOG" + "." + "US", "GOOG",
     "米国株", "情報技術", "USD", "操作確認用"],

    [4, "true", "Tesla", "TSLA" + "." + "US", "TSLA",
     "米国株", "自動車", "USD", "操作確認用"],

    [5, "false", "Apple", "AAPL" + "." + "US", "AAPL",
     "米国株", "情報技術", "USD", "無効化の確認用"],
]

with output_path.open(
    mode="w",
    encoding="utf-8-sig",
    newline=""
) as file:
    writer = csv.writer(file, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)

print(f"作成完了: {output_path}")

with output_path.open(
    mode="r",
    encoding="utf-8-sig"
) as file:
    for line_number, line in enumerate(file, start=1):
        print(line_number, repr(line.rstrip("\n")))
