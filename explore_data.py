import duckdb

DB = "data/transfermarkt.duckdb"

con = duckdb.connect(DB, read_only=True)

print("\n=== COMPETITIONS ===")

competitions = con.execute("""
    SELECT *
    FROM competitions
    LIMIT 20
""").fetchdf()

print(competitions.to_string(index=False))


print("\n=== APPEARANCE COMPETITIONS ===")

competition_ids = con.execute("""
    SELECT
        competition_id,
        COUNT(*) AS appearances
    FROM appearances
    GROUP BY competition_id
    ORDER BY appearances DESC
""").fetchdf()

print(competition_ids.to_string(index=False))


print("\n=== PLAYER SEASONS ===")

seasons = con.execute("""
    SELECT
        last_season,
        COUNT(*) AS players
    FROM players
    GROUP BY last_season
    ORDER BY last_season DESC
    LIMIT 15
""").fetchdf()

print(seasons.to_string(index=False))

con.close()