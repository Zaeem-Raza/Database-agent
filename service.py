import sqlite3
from User import User
import sounddevice as sd
import numpy as np

DB_NAME = "users.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            name TEXT PRIMARY KEY,
            age INTEGER
        )
    """)
    conn.commit()
    conn.close()

def insert_user(user: User) -> str:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (name, age) VALUES (?, ?)",
            (user.name, user.age)
        )
        conn.commit()
        return f"{user.name}, age {user.age}, has been added to the database."
    except sqlite3.IntegrityError:
        return f"User '{user.name}' already exists. Use update to modify the age."
    finally:
        conn.close()

def get_all_users() -> list[User]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name, age FROM users")
    rows = cursor.fetchall()
    conn.close()
    return [User(name=row[0], age=row[1]) for row in rows]


def record_audio(duration, sample_rate):
    print("Recording... Speak now!")
    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return np.squeeze(audio)
