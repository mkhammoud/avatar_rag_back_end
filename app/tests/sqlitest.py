from app.core.InnerSQLs import InnerSQLite

sqlite = InnerSQLite('../../flights.db')
sqlite.init()
print(sqlite.query("select * from flights_fts"))
