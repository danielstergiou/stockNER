import pandas as pd
import re

df = pd.read_csv('./data/stockslist.csv')
pd.set_option('display.max_rows', None)

df["Industry"] = (
    df["Industry"]
      .str.replace(r"[—–]", "-", regex=True)
      .str.replace(r"\s*-\s*", " - ", regex=True)
      .str.strip()
)

split = df["Industry"].str.split(" - ", n=1, expand=True)
df["Industry"]       = split[0]
df["Sub-Industry"]    = split[1].fillna("")

df["Name"] = (
    df["Name"]
      .str.replace(r"\s+(Inc|Ltd|Corporation|Corp|LLC|PLC)\.?\s*$", "", flags=re.IGNORECASE, regex=True)
      .str.replace(",", "", regex=False)
      .str.strip()
)

for sector in sorted(df['Sector'].dropna().unique()):
    sect_df = df[df['Sector'] == sector]
    # print(f"{sector} ({len(sect_df)})")
    for industry in sorted(sect_df['Industry'].dropna().unique()):
        ind_df = sect_df[sect_df['Industry'] == industry]
        # print(f"  - {industry} ({len(ind_df)})")
        subs = ind_df['Sub-Industry'].dropna().unique()
        for sub in sorted(subs):
            if sub == industry:
                continue
            sub_df = ind_df[ind_df['Sub-Industry'] == sub]
            # print(f"    - {sub} ({len(sub_df)})")
    # print()
    
# df = df.map(lambda x: x.replace('"', '') if isinstance(x, str) else x)

# unique_hierarchy = (df[['Sector','Industry','Sub-Industry']].drop_duplicates())
# unique_hierarchy = unique_hierarchy.sort_values(by=['Sector','Industry','Sub-Industry'],ignore_index=True)
# unique_hierarchy.to_csv('./data/sectorIndustries.csv',index=False)

cleaned = df.sort_values(by=["Ticker"],ignore_index=True)
cleaned.to_csv('./data/stocklist.csv',index=False)

#pd.reset_option('display.max_rows')
