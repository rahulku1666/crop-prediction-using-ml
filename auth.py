import sqlite3
import hashlib
import os

class Auth:
    def __init__(self):
        self.db_path = 'users.db'
        self.secret_key = os.getenv('SECRET_KEY', 'your-secret-key')
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                    (username TEXT PRIMARY KEY, 
                     password TEXT)''')
        conn.commit()
        conn.close()

    def _hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def create_user(self, username, email, password):
        try:
            hashed_password = self._hash_password(password)
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                     (username, hashed_password))
            conn.commit()
            conn.close()
            return True, "User created successfully"
        except sqlite3.IntegrityError:
            return False, "Username already exists"
        except Exception as e:
            return False, str(e)

    def verify_user(self, username, password):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username=?", (username,))
            user = c.fetchone()
            conn.close()
            
            if user and user[1] == self._hash_password(password):
                return True, {'username': user[0]}
            return False, None
        except Exception as e:
            return False, None