import numpy as np
import pandas as pd


def flatten_multiindex_columns(columns):
    flat_cols = []
    seen = {}
    for col in columns:
        parts = []
        if isinstance(col, tuple):
            for x in col:
                if pd.notna(x):
                    x = str(x).strip()
                    if x and (not x.startswith("Unnamed")):
                        parts.append(x)
        elif pd.notna(col):
            x = str(col).strip()
            if x and (not x.startswith("Unnamed")):
                parts.append(x)
        name = " | ".join(parts)
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        flat_cols.append(name)
    return flat_cols


def standardize_country_names(df, country_col):
    df = df.copy()
    replacements = {
        "United States of America": "United States",
        "Russian Federation": "Russia",
        "Türkiye": "Turkey",
        "Czechia": "Czech Republic",
        "Republic of Korea": "South Korea",
        "Iran (Islamic Republic of)": "Iran",
        "Egypt, Arab Rep.": "Egypt",
        "Slovak Republic": "Slovakia",
        "Viet Nam": "Vietnam",
        "Lao PDR": "Laos",
        "Brunei Darussalam": "Brunei",
        "Republic of Moldova": "Moldova",
        "United Republic of Tanzania": "Tanzania",
        "Bolivia (Plurinational State of)": "Bolivia",
        "Côte d’Ivoire": "Cote d'Ivoire",
        "Côte d'Ivoire": "Cote d'Ivoire",
    }
    df[country_col] = df[country_col].astype(str).str.strip().replace(replacements)
    return df


def coerce_numeric_series(s):
    """
    Convert messy finance strings to numeric.
    Handles commas and simple text noise.
    """
    s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"nan": np.nan, "": np.nan, "-": np.nan, "NA": np.nan, "N/A": np.nan})
    return pd.to_numeric(s, errors="coerce")


def analyze(emissions, finance):
    e = standardize_country_names(emissions, "country")
    e = e[e.iso_code.str.len() == 3].dropna(subset=["co2", "coal_co2"])
    year = int(e.year.max())
    e = e[(e.year == year) & (e.co2 > 10)].copy()
    e["coal_share"] = e.coal_co2 / e.co2
    f = standardize_country_names(finance, "country")
    f["mitigation_need_billion_usd"] = coerce_numeric_series(f.mitigation_need_billion_usd)
    f = f.dropna(subset=["mitigation_need_billion_usd"])
    f = f[f.mitigation_need_billion_usd >= 0]
    if e.country.duplicated().any() or f.country.duplicated().any():
        raise ValueError("One record per country required")
    result = e.merge(f, on="country", validate="one_to_one")
    if result.empty:
        raise ValueError("No countries matched")
    return year, result.sort_values("country").reset_index(drop=True)


def synthetic_data():
    names = ["Example A", "Example B", "Example C", "Example D", "Example E", "Example F"]
    e = pd.DataFrame(
        {
            "country": names,
            "iso_code": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            "year": [2025] * 6,
            "co2": [100] * 6,
            "coal_co2": [10, 20, 40, 60, 80, 90],
        }
    )
    f = pd.DataFrame(
        {"country": names, "mitigation_need_billion_usd": ["10", "20", "30", "50", "70", "100"]}
    )
    return e, f


def main():
    import argparse, json
    from pathlib import Path
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(
        description="Synthetic coal share and climate finance visualization"
    )
    p.add_argument("--outdir", type=Path, default=Path("output"))
    a = p.parse_args()
    year, df = analyze(*synthetic_data())
    a.outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.outdir / "merged.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df.coal_share, df.mitigation_need_billion_usd)
    for row in df.itertuples():
        ax.annotate(row.country, (row.coal_share, row.mitigation_need_billion_usd))
    ax.set(
        xlabel="Coal share of CO2",
        ylabel="Mitigation need (billion USD)",
        title="Synthetic examples — not country findings",
        yscale="log",
    )
    fig.tight_layout()
    fig.savefig(a.outdir / "synthetic-scatter.png", dpi=140)
    plt.close(fig)
    print(
        json.dumps(
            {
                "synthetic": True,
                "rows": len(df),
                "year": year,
                "coal_share_range": [float(df.coal_share.min()), float(df.coal_share.max())],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
