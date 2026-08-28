"""
Seasonality analysis for the stocks the user uploaded TradingView screenshots
for (in a local screenshot folder, images 1-117). Uses our own historical
data (data/historical/*.parquet) and produces the same year-x-month heatmap
that TradingView shows, plus an opportunity summary for the current/next month.

Run:
    python analyze_seasonality.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from analysis.seasonality import SeasonalAnalyzer, MONTH_ABBR
from core.data_loader import NGXDataLoader

# Stocks identified from TradingView screenshots in a local screenshot folder (1-117).
TICKERS = [
    "DANGCEM",     # 1   — Dangote Cement
    "BUAFOODS",    # 2   — BUA Foods
    "MTNN",        # 3   — MTN Nigeria
    "BUACEMENT",   # 4   — BUA Cement
    "ARADEL",      # 5   — Aradel Holdings
    "WAPCO",       # 6   — Lafarge Africa
    "GTCO",        # 7   — Guaranty Trust Holdings
    "ETI",         # 8   — Ecobank Transnational
    "ZENITHBANK",  # 9   — Zenith Bank
    "FIRSTHOLDCO", # 10  — First HoldCo
    "GEREGU",      # 11  — Geregu Power
    "NB",          # 12  — Nigerian Breweries
    "PRESCO",      # 13  — Presco
    "STANBIC",     # 14  — Stanbic IBTC
    "NESTLE",      # 15  — Nestle Nigeria
    "TRANSCOHOT",  # 16  — Transcorp Hotels
    "INTBREW",     # 17  — International Breweries
    "TRANSPOWER",  # 18  — Transcorp Power
    "UBA",         # 19  — UBA
    "OKOMUOIL",    # 20  — Okomu Oil Palm
    "FIDELITYBK",  # 21  — Fidelity Bank
    "ACCESSCORP",  # 22  — Access Holdings
    "WEMABANK",    # 23  — Wema Bank
    "UNILEVER",    # 24  — Unilever Nigeria
    "DANGSUGAR",   # 25  — Dangote Sugar
    "GUINNESS",    # 26  — Guinness Nigeria
    "FCMB",        # 27  — FCMB Group
    "OANDO",       # 28  — Oando
    "NASCON",      # 29  — NASCON Allied Industries
    "UACN",        # 30  — UAC of Nigeria
    "JBERGER",     # 31  — Julius Berger
    "CUSTODIAN",   # 32  — Custodian Investment
    "TRANSCORP",   # 33  — Transnational Corp of Nigeria
    "NAHCO",       # 34  — Nigerian Aviation Handling Co
    "NGXGROUP",    # 35  — Nigerian Exchange Group
    "JAIZBANK",    # 36  — Jaiz Bank
    "PZ",          # 37  — PZ Cussons Nigeria
    "STERLINGNG",  # 38  — Sterling Financial Holdings
    "FIDSON",      # 39  — Fidson Healthcare
    "MECURE",      # 40  — Mecure Industries
    "BETAGLAS",    # 41  — Beta Glass
    "UCAP",        # 42  — United Capital
    "VITAFOAM",    # 43  — Vitafoam Nigeria
    "TOTAL",       # 44  — TotalEnergies Marketing Nigeria
    "SKYAVN",      # 45  — Skyway Aviation Handling
    "CAP",         # 46  — Chemical & Allied Products
    "AIICO",       # 47  — AIICO Insurance
    "ETRANZACT",   # 48  — eTranzact International
    "CHAMPION",    # 49  — Champion Breweries
    "CADBURY",     # 50  — Cadbury Nigeria
    "NEM",         # 51  — NEM Insurance
    "HONYFLOUR",   # 52  — Honeywell Flour Mills
    "VFDGROUP",    # 53  — VFD Group
    "CONOIL",      # 54  — Conoil
    "MANSARD",     # 55  — AXA Mansard Insurance
    "CORNERST",    # 56  — Cornerstone Insurance
    "ABBEYBDS",    # 57  — Abbey Mortgage Bank
    "MBENEFIT",    # 58  — Mutual Benefits Assurance
    "IKEJAHOTEL",  # 59  — Ikeja Hotel
    "MAYBAKER",    # 60  — May & Baker Nigeria
    "UPDC",        # 61  — UPDC
    "ETERNA",      # 62  — Eterna
    "INFINITY",    # 63  — Infinity Trust Mortgage Bank
    "WAPIC",       # 64  — Coronation Insurance
    "CONHALLPLC",  # 65  — Consolidated Hallmark Holdings
    "CWG",         # 66  — CWG
    "AFRIPRUD",    # 67  — Africa Prudential
    "LINKASSURE",  # 68  — Linkage Assurance
    "BERGER",      # 69  — Berger Paints Nigeria
    "JAPAULGOLD",  # 70  — Japaul Gold & Ventures
    "EUNISELL",    # 71  — Eunisell Interlinked
    "NEIMETH",     # 72  — Neimeth International Pharmaceuticals
    "ELLAHLAKES",  # 73  — Ellah Lakes
    "SOVRENINS",   # 74  — Sovereign Trust Insurance
    "LASACO",      # 75  — Lasaco Assurance
    "CHAMS",       # 76  — Chams Holding Company
    "FTNCOCOA",    # 77  — FTN Cocoa Processors
    "NPFMCRFBK",   # 78  — NPF Microfinance Bank
    "TIP",         # 79  — Initiates
    "REDSTAREX",   # 80  — Red Star Express
    "LIVESTOCK",   # 81  — Livestock Feeds
    "IMG",         # 82  — Industrial & Medical Gases Nigeria
    "UPDCREIT",    # 83  — UPDC Real Estate Investment Trust
    "SUNUASSUR",   # 84  — SUNU Assurances Nigeria
    "VERITASKAP",  # 85  — Veritas Kapital Assurance
    "CUTIX",       # 86  — Cutix
    "TANTALIZER",  # 87  — Tantalizers
    "CAVERTON",    # 88  — Caverton Offshore Support
    "SCOA",        # 89  — SCOA Nigeria
    "UHOMREIT",    # 90  — UH Real Estate Investment Trust
    "CILEASING",   # 91  — C&I Leasing
    "PRESTIGE",    # 92  — Prestige Assurance
    "NCR",         # 93  — NCR (Nigeria)
    "LIVINGTRUST", # 94  — LivingTrust Mortgage Bank
    "RTBRISCOE",   # 95  — RT Briscoe
    "UNIVINSURE",  # 96  — Universal Insurance
    "REGALINS",    # 97  — Regency Alliance Insurance
    "UNITYBNK",    # 98  — Unity Bank
    "ABCTRANS",    # 99  — ABC Transport
    "DAARCOMM",    # 100 — DAAR Communications
    "GUINEAINS",   # 101 — Guinea Insurance
    "NNFM",        # 102 — Northern Nigeria Flour Mills
    "FTGINSURE",   # 103 — Fortis Global Insurance
    "MORISON",     # 104 — Morison Industries
    "ROYALEX",     # 105 — Royal Exchange
    "MULTIVERSE",  # 106 — Multiverse Mining & Exploration
    "MEYER",       # 107 — Meyer
    "CHELLARAM",   # 108 — Chellarams
    "MCNICHOLS",   # 109 — McNichols
    "SFSREIT",     # 110 — SFS Real Estate Investment Trust
    "LEARNAFRCA",  # 111 — Learn Africa
    "DEAPCAP",     # 112 — DEAP Capital Management & Trust
    "JOHNHOLT",    # 113 — John Holt
    "GOLDBREW",    # 114 — Golden Guinea Breweries
    "ACADEMY",     # 115 — Academy Press
    "OMATEK",      # 116 — Omatek Ventures
    "TRANSEXPR",   # 117 — Trans Nationwide Express
]

CURRENT_MONTH = 5    # May 2026
NEXT_MONTH = 6
PORTFOLIO_HELD = {
    "BETAGLAS","BUACEMENT","CUSTODIAN","GEREGU","HONYFLOUR","IKEJAHOTEL",
    "INFINITY","JBERGER","LEARNAFRCA","MBENEFIT","NAHCO","NASCON","OKOMUOIL",
    "PRESCO","SKYAVN","TIP","VFDGROUP","VITAFOAM","WAPCO","TRANSPOWER",
    "NGXGROUP","WEMABANK","MECURE","MAYBAKER","UPDC","ABBEYBDS","ABCTRANS","AIICO",
}


def fmt(v, spec, na="—"):
    return na if pd.isna(v) else format(v, spec)


def show_year_month_heatmap(an: SeasonalAnalyzer, ticker: str) -> None:
    table = an.generate_seasonal_table(ticker)
    if table is None or table.empty:
        print(f"  {ticker}: no data")
        return
    # Show last 7 years
    rows_to_show = ["Average"] + [r for r in table.index if r != "Average"][:7]
    sub = table.loc[rows_to_show]
    print(f"\n{ticker}")
    print("-" * 100)
    header = f"{'Year':>8} " + " ".join(f"{c:>6}" for c in sub.columns)
    print(header)
    for idx, row in sub.iterrows():
        cells = " ".join(fmt(v, "+6.1f") for v in row.values)
        print(f"{str(idx):>8} {cells}")


def main():
    loader = NGXDataLoader(use_cache=False)
    an = SeasonalAnalyzer(loader)

    print("=" * 100)
    print(f"SEASONALITY ANALYSIS — {len(TICKERS)} stocks from user-uploaded screenshots")
    print(f"Using real OHLC from data/historical/  |  current month = {CURRENT_MONTH} ({MONTH_ABBR[CURRENT_MONTH]})")
    print("=" * 100)

    # Per-ticker summary table
    rows = []
    for t in TICKERS:
        m = an.analyze(t)
        if m is None:
            rows.append({"Sym": t, "Status": "no data"})
            continue
        cur_n = m.monthly_sample_sizes.get(CURRENT_MONTH, 0)
        nxt_n = m.monthly_sample_sizes.get(NEXT_MONTH, 0)
        rows.append({
            "Sym": t,
            "Years": m.years_of_data,
            f"{MONTH_ABBR[CURRENT_MONTH]} avg%": m.monthly_avg_returns.get(CURRENT_MONTH, np.nan) * 100,
            f"{MONTH_ABBR[CURRENT_MONTH]} win%": m.monthly_win_rates.get(CURRENT_MONTH, np.nan) * 100,
            f"{MONTH_ABBR[CURRENT_MONTH]} n": cur_n,
            f"{MONTH_ABBR[CURRENT_MONTH]} score": m.current_month_score,
            f"{MONTH_ABBR[NEXT_MONTH]} avg%": m.monthly_avg_returns.get(NEXT_MONTH, np.nan) * 100,
            f"{MONTH_ABBR[NEXT_MONTH]} win%": m.monthly_win_rates.get(NEXT_MONTH, np.nan) * 100,
            f"{MONTH_ABBR[NEXT_MONTH]} n": nxt_n,
            f"{MONTH_ABBR[NEXT_MONTH]} score": m.next_month_score,
            "Best months": ", ".join(f"{MONTH_ABBR[mm]}({v*100:+.1f}%)" for mm, v in m.best_months[:3]),
            "Held": "✓" if t in PORTFOLIO_HELD else "",
        })

    df = pd.DataFrame(rows).set_index("Sym")
    df = df.sort_values(f"{MONTH_ABBR[CURRENT_MONTH]} score", ascending=False)

    print(f"\nCURRENT MONTH ({MONTH_ABBR[CURRENT_MONTH]}) AND NEXT MONTH ({MONTH_ABBR[NEXT_MONTH]}) RANKINGS:")
    print(f"{'Sym':12} {'Yrs':>3} | {'May avg%':>8} {'May win%':>9} {'n':>2} {'score':>6} | "
          f"{'Jun avg%':>8} {'Jun win%':>9} {'n':>2} {'score':>6} | Held | Best months")
    print("-" * 130)
    for sym, r in df.iterrows():
        if r.get("Status") == "no data":
            print(f"{sym:12} (no data)")
            continue
        print(
            f"{sym:12} {int(r['Years']):>3} | "
            f"{fmt(r[f'{MONTH_ABBR[CURRENT_MONTH]} avg%'], '+7.1f'):>8} "
            f"{fmt(r[f'{MONTH_ABBR[CURRENT_MONTH]} win%'], '6.0f'):>9} "
            f"{int(r[f'{MONTH_ABBR[CURRENT_MONTH]} n']):>2} "
            f"{fmt(r[f'{MONTH_ABBR[CURRENT_MONTH]} score'], '+5.2f'):>6} | "
            f"{fmt(r[f'{MONTH_ABBR[NEXT_MONTH]} avg%'], '+7.1f'):>8} "
            f"{fmt(r[f'{MONTH_ABBR[NEXT_MONTH]} win%'], '6.0f'):>9} "
            f"{int(r[f'{MONTH_ABBR[NEXT_MONTH]} n']):>2} "
            f"{fmt(r[f'{MONTH_ABBR[NEXT_MONTH]} score'], '+5.2f'):>6} | "
            f"{r['Held']:^4} | {r['Best months']}"
        )

    # Detailed year-x-month for the highest scoring May names + portfolio holdings
    print("\n\n" + "=" * 100)
    print("YEAR × MONTH HEATMAPS — top 6 by May score + your held names from this list")
    print("=" * 100)
    if "Status" in df.columns:
        df_clean = df[df["Status"].isna()]
    else:
        df_clean = df
    top_may = df_clean.head(6).index.tolist()
    held = [t for t in TICKERS if t in PORTFOLIO_HELD]
    show_set = list(dict.fromkeys(top_may + held))
    for t in show_set:
        show_year_month_heatmap(an, t)

    # June pre-positioning hint
    print("\n\n" + "=" * 100)
    print("JUNE PRE-POSITIONING SHORTLIST — stocks with positive June score AND held in portfolio")
    print("=" * 100)
    candidates = []
    for sym, r in df.iterrows():
        if r.get("Status") == "no data":
            continue
        score = r[f"{MONTH_ABBR[NEXT_MONTH]} score"]
        if pd.isna(score) or score <= 0:
            continue
        candidates.append((sym, score, r[f"{MONTH_ABBR[NEXT_MONTH]} avg%"], r[f"{MONTH_ABBR[NEXT_MONTH]} win%"], r["Held"]))
    candidates.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'Sym':12} {'Jun score':>10} {'Jun avg%':>9} {'Jun win%':>9} {'Held':>4}")
    print("-" * 60)
    for sym, score, avg, win, held in candidates:
        print(f"{sym:12} {score:>+10.2f} {avg:>+9.1f} {win:>9.0f}  {held:>3}")


if __name__ == "__main__":
    main()
