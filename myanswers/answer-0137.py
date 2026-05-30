def detectar_top_percentile(df, percentile):
    thresholds = df.groupby("category")["value"].transform(
        lambda values: values.quantile(percentile / 100)
    )
    return int((df["value"] >= thresholds).sum())
