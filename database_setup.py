import sqlite3

def init_db():
    conn = sqlite3.connect('raman.db')
    cursor = conn.cursor()

    # Create the complaints table with face_encoding column to store binary data
    cursor.execute('''CREATE TABLE IF NOT EXISTS complaints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fname TEXT,
                        lname TEXT,
                        age INTEGER,
                        address TEXT,
                        aadhaar TEXT,
                        missing_date TEXT,
                        missing_location TEXT,
                        photo_path TEXT,
                        status TEXT DEFAULT "active",
                        face_encoding BLOB)''')  # Added face_encoding column for binary data

    cursor.execute('''CREATE TABLE IF NOT EXISTS responses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        complaint_id INTEGER,
                        responder_name TEXT,
                        response_photo_path TEXT,
                        matched BOOLEAN)''')

    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
