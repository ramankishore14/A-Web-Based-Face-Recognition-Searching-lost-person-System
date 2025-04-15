import sqlite3

def add_column():
    conn = sqlite3.connect('raman.db')
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE complaints ADD COLUMN face_encoding BLOB")
        conn.commit()
        print("Column 'face_encoding' added successfully.")
    except sqlite3.OperationalError:
        print("Column 'face_encoding' already exists.")

    conn.close()

if __name__ == "__main__":
    add_column()
