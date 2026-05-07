"""
Usage:
    python data_chunking.py                          # chunk all months (one file per month)
    python data_chunking.py --month 2024-04          # single month -> suno_chat_2024_04.txt
    python data_chunking.py --start 2024-01 --end 2024-06  # range -> suno_chat_2024_01_to_2024_06.txt
    python data_chunking.py --start 2024-01          # range -> suno_chat_2024_01_to_<last>.txt
    python data_chunking.py --end 2024-06            # range -> suno_chat_<first>_to_2024_06.txt
"""

import argparse
import os
import pandas as pd

CSV_PATH = "hf://datasets/hafizhrafizal/suno-discord-chat-history/Suno - SUNO HUB - 💬┃general-chat [1069381916492562585].csv"
OUTPUT_DIR = "chunked_chats"


def parse_args():
    parser = argparse.ArgumentParser(description="Chunk Suno Discord CSV into monthly .txt files.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--month", metavar="YYYY-MM",
                       help="Export a single month (e.g. 2024-04)")
    group.add_argument("--start", metavar="YYYY-MM",
                       help="Start month (inclusive); combine with --end for a range")
    parser.add_argument("--end", metavar="YYYY-MM",
                        help="End month (inclusive); requires --start")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(CSV_PATH, low_memory=False)

    date_col = next((c for c in df.columns if "date" in c.lower() or "timestamp" in c.lower()), None)
    if date_col is None:
        raise ValueError(f"No date column found. Columns: {list(df.columns)}")

    author_col = next((c for c in df.columns if "author" in c.lower()), None)
    content_col = next((c for c in df.columns if "content" in c.lower()), None)
    if author_col is None or content_col is None:
        raise ValueError(f"Could not find author/content columns. Columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], format="mixed", utc=True)
    df = df.sort_values(date_col)
    df["_year_month"] = df[date_col].dt.to_period("M")

    # Filter periods based on CLI args
    if args.month:
        target = pd.Period(args.month, freq="M")
        df = df[df["_year_month"] == target]
        if df.empty:
            print(f"No data found for {args.month}.")
            return
    elif args.start or args.end:
        if args.start:
            df = df[df["_year_month"] >= pd.Period(args.start, freq="M")]
        if args.end:
            df = df[df["_year_month"] <= pd.Period(args.end, freq="M")]
        if df.empty:
            print("No data found for the specified range.")
            return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    is_range = args.start or args.end

    if is_range:
        # Single combined file for the range
        actual_start = df["_year_month"].min()
        actual_end = df["_year_month"].max()
        start_str = f"{actual_start.year}_{actual_start.month:02d}"
        end_str = f"{actual_end.year}_{actual_end.month:02d}"
        filename = f"suno_chat_{start_str}_to_{end_str}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)

        lines = []
        for _, row in df.iterrows():
            ts = row[date_col].strftime("%Y-%m-%d %H:%M:%S")
            author = str(row[author_col]).strip()
            content = str(row[content_col]).strip()
            if content and content.lower() != "nan":
                lines.append(f"[{ts}] {author}: {content}")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  {filename}: {len(lines):,} messages")
    else:
        # Per-month files (all months or single --month)
        for period, group in df.groupby("_year_month"):
            lines = []
            for _, row in group.iterrows():
                ts = row[date_col].strftime("%Y-%m-%d %H:%M:%S")
                author = str(row[author_col]).strip()
                content = str(row[content_col]).strip()
                if content and content.lower() != "nan":
                    lines.append(f"[{ts}] {author}: {content}")

            if not lines:
                continue

            filename = f"suno_chat_{period.year}_{period.month:02d}.txt"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"  {filename}: {len(lines):,} messages")

    print(f"\nDone. Files saved to '{OUTPUT_DIR}/'")
    print(f"Run analysis: python suno_llm_summary.py {OUTPUT_DIR}/{filename}")


if __name__ == "__main__":
    main()
