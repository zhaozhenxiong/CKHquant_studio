"""Transactional derived accounting for the immutable research cache.

The original tables and object bytes remain unchanged. Triggers also cover
maintenance connections and old clients, so accounting does not rely on a
particular Python process observing every mutation. Bootstrap runs once under
the store writer transaction; it never expires or rewrites existing results.
"""


def install_accounting(connection):
    connection.execute('CREATE TABLE IF NOT EXISTS research_accounting_version(version INTEGER NOT NULL)')
    connection.execute('''CREATE TABLE IF NOT EXISTS object_usage(
        sha256 TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, state TEXT NOT NULL,
        ref_count INTEGER NOT NULL DEFAULT 0, pin_count INTEGER NOT NULL DEFAULT 0,
        expires_at REAL NOT NULL DEFAULT 0, accessed_at REAL NOT NULL DEFAULT 0,
        lease_until REAL NOT NULL DEFAULT 0)''')
    connection.execute('''CREATE TABLE IF NOT EXISTS research_totals(
        id INTEGER PRIMARY KEY CHECK(id=1), total_bytes INTEGER NOT NULL,
        cache_bytes INTEGER NOT NULL)''')
    # Startup is O(1) after this one-time, transactional derivation. No CAS
    # bytes, pin owners, lease times, result identities or timestamps change.
    if connection.execute('SELECT version FROM research_accounting_version').fetchone() is None:
        connection.execute('DELETE FROM object_usage')
        connection.execute('''INSERT INTO object_usage
            SELECT o.sha256,o.size_bytes,o.state,
              (SELECT COUNT(*) FROM results r WHERE r.sha256=o.sha256),
              (SELECT COUNT(*) FROM results r JOIN pins p USING(cache_key) WHERE r.sha256=o.sha256),
              COALESCE((SELECT MAX(expires_at) FROM results r WHERE r.sha256=o.sha256),0),
              COALESCE((SELECT MAX(accessed_at) FROM results r WHERE r.sha256=o.sha256),0),
              COALESCE((SELECT MAX(l.expires_at) FROM results r JOIN leases l USING(cache_key) WHERE r.sha256=o.sha256),0)
            FROM objects o''')
        connection.execute('''INSERT OR REPLACE INTO research_totals
            SELECT 1,COALESCE(SUM(size_bytes),0),
              COALESCE(SUM(CASE WHEN pin_count=0 THEN size_bytes ELSE 0 END),0)
            FROM object_usage''')
        connection.execute('INSERT INTO research_accounting_version VALUES(1)')
    for sql in (
        'CREATE INDEX IF NOT EXISTS research_results_namespace_access ON results(namespace,accessed_at DESC)',
        'CREATE INDEX IF NOT EXISTS research_results_namespace_created ON results(namespace,created_at DESC)',
        'CREATE INDEX IF NOT EXISTS research_results_access ON results(accessed_at DESC)',
        'CREATE INDEX IF NOT EXISTS research_usage_expiry ON object_usage(state,pin_count,MAX(expires_at,lease_until),accessed_at,sha256)',
        'CREATE INDEX IF NOT EXISTS research_usage_lru ON object_usage(state,pin_count,accessed_at,sha256)',
        'CREATE INDEX IF NOT EXISTS research_gc_pending ON gc_journal(state,id)',
    ):
        connection.execute(sql)

    triggers = {
        'object_insert': '''AFTER INSERT ON objects BEGIN
            INSERT INTO object_usage(sha256,size_bytes,state) VALUES(NEW.sha256,NEW.size_bytes,NEW.state);
            UPDATE research_totals SET total_bytes=total_bytes+NEW.size_bytes,
                cache_bytes=cache_bytes+NEW.size_bytes WHERE id=1;
        END''',
        'object_delete': '''BEFORE DELETE ON objects BEGIN
            UPDATE research_totals SET total_bytes=total_bytes-OLD.size_bytes,
                cache_bytes=cache_bytes-CASE WHEN (SELECT pin_count FROM object_usage WHERE sha256=OLD.sha256)=0
                    THEN OLD.size_bytes ELSE 0 END WHERE id=1;
            DELETE FROM object_usage WHERE sha256=OLD.sha256;
        END''',
        'object_state': '''AFTER UPDATE OF state ON objects BEGIN
            UPDATE object_usage SET state=NEW.state WHERE sha256=NEW.sha256;
        END''',
        'result_insert': '''AFTER INSERT ON results BEGIN
            UPDATE object_usage SET ref_count=ref_count+1,
                expires_at=MAX(expires_at,NEW.expires_at),accessed_at=MAX(accessed_at,NEW.accessed_at)
                WHERE sha256=NEW.sha256;
        END''',
        # Explicit cascades run while the parent still exists, so pin/lease
        # triggers can resolve the shared SHA (including FK-disabled clients).
        'result_before_delete': '''BEFORE DELETE ON results BEGIN
            DELETE FROM pins WHERE cache_key=OLD.cache_key;
            DELETE FROM leases WHERE cache_key=OLD.cache_key;
        END''',
        'result_delete': '''AFTER DELETE ON results BEGIN
            UPDATE object_usage SET ref_count=ref_count-1,
                expires_at=COALESCE((SELECT MAX(expires_at) FROM results WHERE sha256=OLD.sha256),0),
                accessed_at=COALESCE((SELECT MAX(accessed_at) FROM results WHERE sha256=OLD.sha256),0)
                WHERE sha256=OLD.sha256;
        END''',
        'result_touch': '''AFTER UPDATE OF expires_at,accessed_at ON results BEGIN
            UPDATE object_usage SET
                expires_at=COALESCE((SELECT MAX(expires_at) FROM results WHERE sha256=NEW.sha256),0),
                accessed_at=COALESCE((SELECT MAX(accessed_at) FROM results WHERE sha256=NEW.sha256),0)
                WHERE sha256=NEW.sha256;
        END''',
        'pin_insert': '''AFTER INSERT ON pins BEGIN
            UPDATE research_totals SET cache_bytes=cache_bytes-COALESCE((SELECT size_bytes FROM object_usage
                WHERE sha256=(SELECT sha256 FROM results WHERE cache_key=NEW.cache_key) AND pin_count=0),0) WHERE id=1;
            UPDATE object_usage SET pin_count=pin_count+1
                WHERE sha256=(SELECT sha256 FROM results WHERE cache_key=NEW.cache_key);
        END''',
        'pin_delete': '''BEFORE DELETE ON pins BEGIN
            UPDATE research_totals SET cache_bytes=cache_bytes+COALESCE((SELECT size_bytes FROM object_usage
                WHERE sha256=(SELECT sha256 FROM results WHERE cache_key=OLD.cache_key) AND pin_count=1),0) WHERE id=1;
            UPDATE object_usage SET pin_count=pin_count-1
                WHERE sha256=(SELECT sha256 FROM results WHERE cache_key=OLD.cache_key);
        END''',
    }
    for action, reference in [('INSERT', 'NEW'), ('UPDATE', 'NEW'), ('DELETE', 'OLD')]:
        triggers['lease_' + action.lower()] = f'''AFTER {action} ON leases BEGIN
            UPDATE object_usage SET lease_until=COALESCE((SELECT MAX(l.expires_at)
                FROM leases l JOIN results r USING(cache_key) WHERE r.sha256=object_usage.sha256),0)
                WHERE sha256=(SELECT sha256 FROM results WHERE cache_key={reference}.cache_key);
        END'''
    for name, definition in triggers.items():
        connection.execute(f'CREATE TRIGGER IF NOT EXISTS research_account_{name} {definition}')
