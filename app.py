target_master = master_df.loc[
    master_df["enabled"].astype(str).str.lower().eq("true")
].copy()

expected = set(target_master["provider_symbol"])
actual = set(fetched_df["provider_symbol"])

missing = expected - actual
unexpected = actual - expected

if missing:
    raise ValueError(f"取得できなかった銘柄: {sorted(missing)}")

if unexpected:
    raise ValueError(f"処理対象外のデータが含まれています: {sorted(unexpected)}")
