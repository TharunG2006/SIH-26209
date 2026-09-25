import os
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from pathlib import Path
from config import ROOT, SPACECRAFT, SUBSYSTEM

DB_PATH = ROOT / "satellite_health.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS Satellite (
    id          SERIAL PRIMARY KEY,
    name        TEXT    NOT NULL UNIQUE,
    norad_id    INTEGER,
    launch_date TEXT
);

CREATE TABLE IF NOT EXISTS Channel (
    id           SERIAL PRIMARY KEY,
    satellite_id INTEGER NOT NULL REFERENCES Satellite(id) ON DELETE CASCADE,
    subsystem    TEXT,
    name         TEXT    NOT NULL,
    unit         TEXT,
    UNIQUE (satellite_id, name)
);

CREATE TABLE IF NOT EXISTS TelemetryReading (
    id         SERIAL PRIMARY KEY,
    channel_id INTEGER NOT NULL REFERENCES Channel(id) ON DELETE CASCADE,
    timestep   INTEGER NOT NULL,
    timestamp  TEXT,
    value      REAL    NOT NULL,
    UNIQUE (channel_id, timestep)
);

CREATE TABLE IF NOT EXISTS Anomaly (
    id             SERIAL PRIMARY KEY,
    satellite_id   INTEGER NOT NULL REFERENCES Satellite(id) ON DELETE CASCADE,
    detected_at    INTEGER NOT NULL,
    detected_utc   TEXT,
    ended_at       INTEGER,
    peak_at        INTEGER,
    severity_score REAL    NOT NULL,
    status         TEXT    NOT NULL DEFAULT 'open'
                   CHECK (status IN ('open', 'acknowledged', 'resolved',
                                     'false_positive')),
    UNIQUE (satellite_id, detected_at, severity_score)
);

CREATE TABLE IF NOT EXISTS AnomalyContribution (
    id              SERIAL PRIMARY KEY,
    anomaly_id      INTEGER NOT NULL REFERENCES Anomaly(id) ON DELETE CASCADE,
    channel_id      INTEGER NOT NULL REFERENCES Channel(id) ON DELETE CASCADE,
    deviation_score REAL    NOT NULL,
    share_pct       REAL,
    onset           INTEGER,
    lag             INTEGER,
    chain_rank      INTEGER,
    actual          REAL,
    predicted       REAL,
    UNIQUE (anomaly_id, channel_id)
);

CREATE TABLE IF NOT EXISTS Operator (
    id   SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    role TEXT
);

CREATE TABLE IF NOT EXISTS Alert (
    id              SERIAL PRIMARY KEY,
    anomaly_id      INTEGER NOT NULL REFERENCES Anomaly(id) ON DELETE CASCADE,
    sent_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    acknowledged_by INTEGER REFERENCES Operator(id),
    acknowledged_at TIMESTAMP,
    UNIQUE (anomaly_id)
);

CREATE INDEX IF NOT EXISTS idx_reading_channel ON TelemetryReading(channel_id);
CREATE INDEX IF NOT EXISTS idx_contrib_anomaly ON AnomalyContribution(anomaly_id);
CREATE INDEX IF NOT EXISTS idx_contrib_channel ON AnomalyContribution(channel_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_satellite ON Anomaly(satellite_id);
"""

class SQLiteMockConnection:
    def __init__(self):
        host = os.environ.get('RDS_HOST', 'localhost')
        db = os.environ.get('RDS_DB', 'postgres')
        user = os.environ.get('RDS_USER', 'postgres')
        password = os.environ.get('RDS_PASSWORD', '')
        self.conn = psycopg2.connect(host=host, database=db, user=user, password=password)
    
    def execute(self, sql, params=None):
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        if params:
            cur.execute(sql.replace('?', '%s'), params)
        else:
            cur.execute(sql)
        return cur
        
    def executemany(self, sql, params_list):
        cur = self.conn.cursor()
        cur.executemany(sql.replace('?', '%s'), params_list)
        return cur
        
    def executescript(self, sql):
        cur = self.conn.cursor()
        cur.execute(sql)
        
    def commit(self):
        self.conn.commit()
        
    def close(self):
        self.conn.close()

@contextmanager
def connect(path: Path | str = DB_PATH):
    conn = SQLiteMockConnection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db(path: Path | str = DB_PATH) -> Path:
    with connect(path) as conn:
        conn.executescript(SCHEMA)
    return Path(path)

def upsert_satellite(conn, name: str, norad_id: int | None = None,
                     launch_date: str | None = None) -> int:
    conn.execute(
        "INSERT INTO Satellite (name, norad_id, launch_date) VALUES (?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET norad_id = COALESCE(EXCLUDED.norad_id, Satellite.norad_id), "
        "launch_date = COALESCE(EXCLUDED.launch_date, Satellite.launch_date)",
        (name, norad_id, launch_date),
    )
    return conn.execute("SELECT id FROM Satellite WHERE name = ?",
                        (name,)).fetchone()["id"]

def upsert_channels(conn, satellite_id: int, channels: list[str]) -> dict[str, int]:
    rows = [(satellite_id, SUBSYSTEM.get(c.split("-")[0], c.split("-")[0]), c)
            for c in channels]
    conn.executemany(
        "INSERT INTO Channel (satellite_id, subsystem, name, unit) "
        "VALUES (?, ?, ?, NULL) ON CONFLICT(satellite_id, name) DO NOTHING",
        rows,
    )
    return {r["name"]: r["id"] for r in conn.execute(
        "SELECT id, name FROM Channel WHERE satellite_id = ?", (satellite_id,))}

def store_readings(conn, channel_ids: dict[str, int], series: dict[str, list],
                   batch: int = 20000, stamps=None) -> int:
    total = 0
    for ch, values in series.items():
        cid = channel_ids[ch]
        rows = [
            (cid, i, float(v),
             str(stamps[i]) if stamps is not None and i < len(stamps) else None)
            for i, v in enumerate(values)
        ]
        for i in range(0, len(rows), batch):
            conn.executemany(
                "INSERT INTO TelemetryReading (channel_id, timestep, value, "
                "timestamp) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(channel_id, timestep) DO NOTHING",
                rows[i:i + batch],
            )
        total += len(rows)
    return total

def store_anomalies(conn, satellite_id: int, channel_ids: dict[str, int],
                    anomalies, raise_alerts: bool = True, stamps=None) -> int:
    written = 0
    for a in anomalies:
        when = (str(stamps[min(a.start, len(stamps) - 1)])
                if stamps is not None and len(stamps) else None)
        cur = conn.execute(
            "INSERT INTO Anomaly (satellite_id, detected_at, detected_utc, "
            "ended_at, peak_at, severity_score, status) "
            "VALUES (?, ?, ?, ?, ?, ?, 'open') "
            "ON CONFLICT(satellite_id, detected_at, severity_score) DO NOTHING",
            (satellite_id, a.start, when, a.end, a.peak, a.severity),
        )
        if cur.rowcount == 0:
            row = conn.execute(
                "SELECT id FROM Anomaly WHERE satellite_id = ? AND detected_at = ? "
                "AND severity_score = ?", (satellite_id, a.start, a.severity)
            ).fetchone()
            anomaly_id = row["id"]
        else:
            anomaly_id = cur.execute("SELECT lastval() as id").fetchone()["id"]
            written += 1

        conn.executemany(
            "INSERT INTO AnomalyContribution (anomaly_id, channel_id, "
            "deviation_score, share_pct, onset, lag, chain_rank, actual, predicted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(anomaly_id, channel_id) DO NOTHING",
            [(anomaly_id, channel_ids[c["channel"]], c["z"], c["share_pct"],
              c.get("onset"), c.get("lag"), c.get("chain_rank"),
              c.get("actual"), c.get("predicted")) for c in a.contributions],
        )
        if raise_alerts:
            conn.execute(
                "INSERT INTO Alert (anomaly_id) VALUES (?) "
                "ON CONFLICT(anomaly_id) DO NOTHING", (anomaly_id,))
    return written

EXPLAIN_SQL = """
SELECT  a.id                AS anomaly_id,
        s.name              AS satellite,
        a.detected_at, a.detected_utc, a.ended_at, a.peak_at,
        a.severity_score, a.status,
        c.name              AS channel,
        c.subsystem,
        ac.deviation_score, ac.share_pct, ac.onset, ac.lag, ac.chain_rank,
        ac.actual, ac.predicted
FROM        Anomaly              a
JOIN        Satellite            s  ON s.id  = a.satellite_id
JOIN        AnomalyContribution  ac ON ac.anomaly_id = a.id
JOIN        Channel              c  ON c.id  = ac.channel_id
WHERE       a.id = ?
ORDER BY    ac.share_pct DESC
"""

def explain(conn, anomaly_id: int):
    return conn.execute(EXPLAIN_SQL, (anomaly_id,)).fetchall()

def causal_chain(conn, anomaly_id: int):
    return conn.execute(
        "SELECT c.name AS channel, c.subsystem, ac.onset, ac.lag, "
        "       ac.deviation_score, ac.chain_rank "
        "FROM AnomalyContribution ac JOIN Channel c ON c.id = ac.channel_id "
        "WHERE ac.anomaly_id = ? ORDER BY ac.chain_rank", (anomaly_id,)
    ).fetchall()

def open_alerts(conn):
    return conn.execute(
        "SELECT al.id AS alert_id, al.sent_at, a.id AS anomaly_id, "
        "       s.name AS satellite, a.detected_at, a.severity_score, "
        "       (SELECT c.name FROM AnomalyContribution ac "
        "        JOIN Channel c ON c.id = ac.channel_id "
        "        WHERE ac.anomaly_id = a.id ORDER BY ac.chain_rank LIMIT 1) "
        "        AS first_mover, "
        "       (SELECT c.name FROM AnomalyContribution ac "
        "        JOIN Channel c ON c.id = ac.channel_id "
        "        WHERE ac.anomaly_id = a.id ORDER BY ac.share_pct DESC LIMIT 1) "
        "        AS largest_contributor "
        "FROM Alert al JOIN Anomaly a ON a.id = al.anomaly_id "
        "JOIN Satellite s ON s.id = a.satellite_id "
        "WHERE al.acknowledged_by IS NULL "
        "ORDER BY a.severity_score DESC"
    ).fetchall()

def acknowledge(conn, alert_id: int, operator_name: str,
                role: str | None = None) -> None:
    conn.execute("INSERT INTO Operator (name, role) VALUES (?, ?) "
                 "ON CONFLICT(name) DO NOTHING", (operator_name, role))
    op = conn.execute("SELECT id FROM Operator WHERE name = ?",
                      (operator_name,)).fetchone()["id"]
    conn.execute(
        "UPDATE Alert SET acknowledged_by = ?, acknowledged_at = CURRENT_TIMESTAMP "
        "WHERE id = ?", (op, alert_id))
    conn.execute(
        "UPDATE Anomaly SET status = 'acknowledged' WHERE id = "
        "(SELECT anomaly_id FROM Alert WHERE id = ?)", (alert_id,))

def summary(conn) -> dict:
    def one(sql):
        return conn.execute(sql).fetchone()['count']
    return {
        "satellites": one("SELECT COUNT(*) as count FROM Satellite"),
        "channels": one("SELECT COUNT(*) as count FROM Channel"),
        "telemetry_readings": one("SELECT COUNT(*) as count FROM TelemetryReading"),
        "anomalies": one("SELECT COUNT(*) as count FROM Anomaly"),
        "contributions": one("SELECT COUNT(*) as count FROM AnomalyContribution"),
        "alerts": one("SELECT COUNT(*) as count FROM Alert"),
        "operators": one("SELECT COUNT(*) as count FROM Operator"),
    }
