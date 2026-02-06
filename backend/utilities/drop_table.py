import argparse
import logging
import os

import psycopg

# Database connection configuration
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "6543")
PG_USER = os.getenv("POSTGRES_USER", "user")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "password")
PG_DB = os.getenv("POSTGRES_DB", "beer_rag")

connection_string = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

logger = logging.getLogger(__name__)


def drop_table(table_name: str):
    logger.info(f"Connecting to database to drop table '{table_name}'...")
    try:
        with psycopg.connect(connection_string) as conn:
            with conn.cursor() as cur:
                # Use cascade to drop dependent objects if any
                cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
                logger.info(
                    f"Table '{table_name}' dropped successfully (if it existed)."
                )
    except Exception as e:
        logger.error(f"Error dropping table: {e}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Drop a PGVector table (collection)")
    parser.add_argument("table_name", help="Name of the table/collection to drop")
    args = parser.parse_args()

    # Confirm action
    confirm = input(
        f"Are you sure you want to DROP the table '{args.table_name}'? This cannot be undone. [y/N]: "
    )
    if confirm.lower() == "y":
        drop_table(args.table_name)
    else:
        logger.info("Operation cancelled.")


if __name__ == "__main__":
    main()
