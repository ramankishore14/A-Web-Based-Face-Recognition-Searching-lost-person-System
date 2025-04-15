import sqlite3

def check_columns():
    conn = sqlite3.connect('raman.db')
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(complaints)")
    columns = cursor.fetchall()

    print("Columns in complaints table:")
    for col in columns:
        print(col[1])  # col[1] is the column name

    conn.close()

if __name__ == "__main__":
    check_columns()
