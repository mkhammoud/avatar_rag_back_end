import sqlite3

from app.core.pipes.SQLRetrieval import InnerSQL


class InnerSQLite(InnerSQL):
    def __init__(self, sql_conn):
        self.conn = None
        self.sql_conn = sql_conn
        self.conn = sqlite3.connect(self.sql_conn, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def get_conn(self):
        return self.conn

    def query(self, q) -> any:
        cursor = self.conn.cursor()
        cursor.execute(q)
        results = [dict(row) for row in cursor.fetchall()]
        return results
