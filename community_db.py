import sqlite3
from pathlib import Path


DB_PATH = Path("data") / "community.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    connection.row_factory = sqlite3.Row

    return connection


def init_db():
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            club TEXT NOT NULL,
            category TEXT NOT NULL,
            text TEXT NOT NULL,
            likes INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (post_id) REFERENCES posts(id)
        )
        """
    )

    connection.commit()
    connection.close()


def create_post(club, category, text):
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO posts (club, category, text)
        VALUES (?, ?, ?)
        """,
        (club, category, text)
    )

    connection.commit()

    post_id = cursor.lastrowid

    connection.close()

    return post_id


def get_posts(club):
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            club,
            category,
            text,
            likes,
            created_at
        FROM posts
        WHERE club = ?
        ORDER BY id DESC
        """,
        (club,)
    )

    posts = [dict(row) for row in cursor.fetchall()]

    connection.close()

    return posts


def like_post(post_id):
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE posts
        SET likes = likes + 1
        WHERE id = ?
        """,
        (post_id,)
    )

    connection.commit()
    connection.close()


def create_comment(post_id, text):
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO comments (post_id, text)
        VALUES (?, ?)
        """,
        (post_id, text)
    )

    connection.commit()
    connection.close()


def get_comments(post_id):
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            post_id,
            text,
            created_at
        FROM comments
        WHERE post_id = ?
        ORDER BY id ASC
        """,
        (post_id,)
    )

    comments = [dict(row) for row in cursor.fetchall()]

    connection.close()

    return comments


init_db()