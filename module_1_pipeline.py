from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/"
FIXED_GBP_TO_INR = 105.50
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
RAW_COLUMNS = ["title", "price", "availability", "star_rating", "category"]

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DB_PATH = DATA_DIR / "books_catalog.sqlite"
SNAPSHOT_PATH = DATA_DIR / "raw_snapshot.csv"
CLEANED_PATH = OUTPUT_DIR / "cleaned_books.csv"
SQL_OUTPUT_PATH = OUTPUT_DIR / "sql_outputs.txt"
PANDAS_OUTPUT_PATH = OUTPUT_DIR / "pandas_comparison.txt"


def fetch_html(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=20)
    response.raise_for_status()
    return response.text


def scrape_pages(pages: int = 5) -> pd.DataFrame:
    """Scrape the first N all-products pages with requests + BeautifulSoup."""
    rows: list[dict[str, str]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "catalog-benchmark/1.0 (educational scraping)"})

    for page_number in range(1, pages + 1):
        listing_url = f"{BASE_URL}page-{page_number}.html"
        soup = BeautifulSoup(fetch_html(session, listing_url), "html.parser")
        products = soup.select("article.product_pod")
        if not products:
            raise RuntimeError(f"No products found on {listing_url}")

        for product in products:
            link = product.select_one("h3 a")
            if link is None or not link.get("href"):
                continue
            detail_url = requests.compat.urljoin(listing_url, link["href"])
            detail_soup = BeautifulSoup(fetch_html(session, detail_url), "html.parser")

            rating_node = detail_soup.select_one("p.star-rating")
            rating_text = ""
            if rating_node:
                classes = rating_node.get("class", [])
                rating_text = next((c for c in classes if c in RATING_MAP), "")

            availability_node = detail_soup.select_one("p.availability")
            availability = availability_node.get_text(" ", strip=True) if availability_node else ""

            breadcrumb = detail_soup.select("ul.breadcrumb li")
            category = breadcrumb[-2].get_text(" ", strip=True) if len(breadcrumb) >= 2 else ""

            price_node = product.select_one("p.price_color")
            rows.append(
                {
                    "title": link.get("title") or link.get_text(" ", strip=True),
                    "price": price_node.get_text(" ", strip=True) if price_node else "",
                    "availability": availability,
                    "star_rating": rating_text,
                    "category": category,
                }
            )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def read_snapshot(path: Path = SNAPSHOT_PATH) -> pd.DataFrame:
    """Read the captured first-five-page source snapshot when live DNS is unavailable."""
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        next(handle)  # header
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            # The title can contain commas. The final four comma-separated fields are stable.
            parts = line.rsplit(",", 4)
            if len(parts) != 5:
                continue
            title, price, availability, star_rating, category = parts
            rows.append(
                {
                    "title": title,
                    "price": price,
                    "availability": availability,
                    "star_rating": star_rating,
                    "category": category,
                }
            )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def clean_books(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = raw.copy()
    before = len(df)

    df["title"] = df["title"].astype("string").str.strip()
    df["category"] = df["category"].astype("string").str.strip()
    df["price_gbp"] = pd.to_numeric(
        df["price"].astype("string").str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )
    df["rating"] = df["star_rating"].map(RATING_MAP).astype("Float64")
    availability_lower = df["availability"].astype("string").str.strip().str.lower()
    df["in_stock"] = availability_lower.map(
        lambda value: True if value.startswith("in stock") else False if value.startswith("out of stock") else pd.NA
    ).astype("boolean")

    # Non-numeric parsing failures are dropped; numeric price/rating failures use median imputation.
    price_missing = int(df["price_gbp"].isna().sum())
    rating_missing = int(df["rating"].isna().sum())
    df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
    df["rating"] = df["rating"].fillna(df["rating"].median()).round().astype("Int64")
    bad_rows = df[["title", "category", "in_stock"]].isna().any(axis=1)
    dropped = int(bad_rows.sum())
    df = df.loc[~bad_rows].copy()

    df["price_inr"] = (df["price_gbp"] * FIXED_GBP_TO_INR).round(2)
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df = df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]

    stats = {
        "raw_rows": before,
        "clean_rows": len(df),
        "dropped_rows": dropped,
        "price_median_imputations": price_missing,
        "rating_median_imputations": rating_missing,
        "categories": int(df["category"].nunique()),
    }
    return df, stats


def create_database(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    if db_path.exists():
        db_path.unlink()
    categories = pd.DataFrame({"category_name": sorted(df["category"].unique())})

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT NOT NULL UNIQUE
            )"""
        )
        conn.execute(
            """CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id) REFERENCES categories(category_id)
            )"""
        )
        categories.to_sql("categories", conn, if_exists="append", index=False)
        category_lookup = dict(conn.execute("SELECT category_name, category_id FROM categories"))
        books_for_sql = df.copy()
        books_for_sql["category_id"] = books_for_sql["category"].map(category_lookup)
        books_for_sql["in_stock"] = books_for_sql["in_stock"].astype(int)
        books_for_sql = books_for_sql[
            ["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]
        ]
        books_for_sql.to_sql("books", conn, if_exists="append", index=False)


def execute_queries(db_path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    queries = {
        "q1_select_where": """
SELECT title, price_gbp, rating
FROM books
WHERE in_stock = 1 AND price_gbp > 40
ORDER BY price_gbp DESC;
""".strip(),
        "q2_order_limit": """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10;
""".strip(),
        "q3_distinct": """
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;
""".strip(),
        "q4_in": """
SELECT title, category_id, rating
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, title
LIMIT 15;
""".strip(),
        "q5_join_between": """
SELECT c.category_name, b.title, b.rating, b.price_inr
FROM books AS b
JOIN categories AS c ON b.category_id = c.category_id
WHERE b.price_gbp BETWEEN 20 AND 40
ORDER BY c.category_name, b.rating DESC, b.title
LIMIT 20;
""".strip(),
        "q6_join_top10_per_category": """
WITH ranked AS (
    SELECT
        c.category_name,
        b.title,
        b.rating,
        b.price_inr,
        ROW_NUMBER() OVER (
            PARTITION BY c.category_id
            ORDER BY b.rating DESC, b.price_gbp DESC, b.title
        ) AS category_rank
    FROM books AS b
    JOIN categories AS c ON b.category_id = c.category_id
)
SELECT category_name, title, rating, price_inr
FROM ranked
WHERE category_rank <= 10
ORDER BY category_name, category_rank;
""".strip(),
    }

    results: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(db_path) as conn:
        for name, query in queries.items():
            results[name] = pd.read_sql(query, conn)

    with SQL_OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for name, query in queries.items():
            out.write(f"[{name}]\n{query}\n\nOutput:\n")
            out.write(results[name].to_string(index=False))
            out.write("\n\n" + "=" * 80 + "\n\n")
    return results


def pandas_equivalent_join(df: pd.DataFrame, db_path: Path = DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(db_path) as conn:
        books_sql = pd.read_sql("SELECT * FROM books", conn)
        categories_sql = pd.read_sql("SELECT * FROM categories", conn)
        sql_join = pd.read_sql(
            """WITH ranked AS (
                SELECT c.category_name, b.title, b.rating, b.price_inr,
                       ROW_NUMBER() OVER (
                           PARTITION BY c.category_id
                           ORDER BY b.rating DESC, b.price_gbp DESC, b.title
                       ) AS category_rank
                FROM books b
                JOIN categories c ON b.category_id = c.category_id
            )
            SELECT category_name, title, rating, price_inr
            FROM ranked
            WHERE category_rank <= 10
            ORDER BY category_name, category_rank""",
            conn,
        )

    merged = books_sql.merge(categories_sql, on="category_id", how="inner")
    merged = merged.sort_values(
        ["category_name", "rating", "price_gbp", "title"],
        ascending=[True, False, False, True],
    )
    merged["category_rank"] = merged.groupby("category_id").cumcount() + 1
    pandas_join = merged.loc[merged["category_rank"] <= 10, ["category_name", "title", "rating", "price_inr", "category_rank"]]
    pandas_join = pandas_join.sort_values(["category_name", "category_rank"], kind="stable").drop(columns=["category_rank"])
    pandas_join = pandas_join.reset_index(drop=True)
    sql_join = sql_join.reset_index(drop=True)

    # The SQL result and pandas result use the same tie-break order.
    sql_compare = sql_join.copy()
    pandas_compare = pandas_join.copy()
    equivalent = sql_compare.equals(pandas_compare)

    with PANDAS_OUTPUT_PATH.open("w", encoding="utf-8") as out:
        out.write("pd.read_sql JOIN result:\n")
        out.write(sql_join.to_string(index=False))
        out.write("\n\n" + "-" * 80 + "\n\n")
        out.write("pd.merge JOIN result:\n")
        out.write(pandas_join.to_string(index=False))
        out.write(f"\n\nEquivalent: {equivalent}\n")
    return sql_join, pandas_join


def run(source: str, pages: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    actual_source = source
    if source == "live":
        raw = scrape_pages(pages=pages)
    elif source == "snapshot":
        raw = read_snapshot()
    else:
        try:
            raw = scrape_pages(pages=pages)
            actual_source = "live"
        except (requests.RequestException, OSError, RuntimeError) as exc:
            print(f"Live scrape unavailable; using bundled source snapshot: {exc}")
            raw = read_snapshot()
            actual_source = "snapshot-fallback"

    cleaned, stats = clean_books(raw)
    if len(cleaned) < 60:
        raise RuntimeError(f"Acceptance criterion failed: only {len(cleaned)} clean rows")
    if cleaned["category"].nunique() < 3:
        raise RuntimeError("Acceptance criterion failed: fewer than 3 categories")

    cleaned.to_csv(CLEANED_PATH, index=False)
    create_database(cleaned)
    results = execute_queries()
    sql_join, pandas_join = pandas_equivalent_join(cleaned)

    print("Pipeline complete")
    print(f"Source mode: {actual_source}")
    print(f"Rows: {stats['clean_rows']}")
    print(f"Categories: {stats['categories']}")
    print(f"GBP -> INR fixed rate: {FIXED_GBP_TO_INR:.2f}")
    print(f"Dropped rows: {stats['dropped_rows']}")
    print(f"Median price imputations: {stats['price_median_imputations']}")
    print(f"Median rating imputations: {stats['rating_median_imputations']}")
    print(f"SQL query outputs: {SQL_OUTPUT_PATH}")
    print(f"Pandas comparison: {PANDAS_OUTPUT_PATH}")
    print(f"JOIN equivalent: {sql_join.equals(pandas_join)}")
    print(f"Database: {DB_PATH}")
    print(f"Query result sizes: { {k: len(v) for k, v in results.items()} }")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Books to Scrape catalog pipeline")
    parser.add_argument("--source", choices=["auto", "live", "snapshot"], default="auto")
    parser.add_argument("--pages", type=int, default=5)
    args = parser.parse_args()
    run(source=args.source, pages=args.pages)
