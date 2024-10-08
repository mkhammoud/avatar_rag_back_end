import sqlite3

from app.core.pipes.SQLRetrieval import InnerSQL


class InnerSQLite(InnerSQL):
    def __init__(self, sql_conn):
        self.conn = None
        self.sql_conn = sql_conn

    def init(self) -> any:
        self.conn = sqlite3.connect(self.sql_conn)

    def query(self, q) -> any:
        cursor = self.conn.cursor()
        results = []
        for row in cursor.execute(q):
            results.append(row)
        return results
