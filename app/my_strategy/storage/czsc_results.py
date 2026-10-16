"""New CZSC run index and atomic reports; no legacy result fallback."""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid
from typing import Any

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, PROCESSED_DATA_ROOT
from my_strategy.core.tz import local_now


class ResultStore:
    def __init__(self, db_path: Path | None = None, runs_root: Path | None = None) -> None:
        self.db_path = db_path or PROCESSED_DATA_ROOT / "czsc" / "results.db"
        self.runs_root = runs_root or ARTIFACT_RUNS_ROOT
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS czsc_runs (run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, symbol TEXT, created_at TEXT NOT NULL, summary TEXT NOT NULL, report_path TEXT NOT NULL)")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scan_views(run_id TEXT PRIMARY KEY,report_sha256 TEXT NOT NULL,
                    report_signature TEXT NOT NULL,header TEXT NOT NULL,row_count INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS scan_view_rows(run_id TEXT NOT NULL,ordinal INTEGER NOT NULL,
                    symbol TEXT,category TEXT,direction TEXT,search_text TEXT,payload TEXT NOT NULL,
                    PRIMARY KEY(run_id,ordinal));
                CREATE INDEX IF NOT EXISTS scan_rows_category ON scan_view_rows(run_id,category,ordinal);
                CREATE INDEX IF NOT EXISTS scan_rows_symbol ON scan_view_rows(run_id,symbol,ordinal);
            """)
            conn.commit()

    def save(self, kind: str, result: dict[str, Any]) -> dict[str, Any]:
        from filelock import FileLock
        token=hashlib.sha256(str(result.get('run_id')).encode()).hexdigest()[:24]
        with FileLock(str(self.db_path)+'.'+token+'.lock'):
            return self._save(kind,result)

    def _save(self, kind: str, result: dict[str, Any]) -> dict[str, Any]:
        run_id = result["run_id"]
        if not re.fullmatch(r"[\w.-]+", run_id) or run_id in {".", ".."}:
            raise ValueError("Invalid run identity")
        target = self.runs_root / run_id / "reports" / "czsc.json"
        # Services may have already committed this exact result. The task layer
        # only needs its reference; avoid rewriting an identical full report.
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if target.is_file() and target.read_text(encoding='utf-8') == encoded:
            with closing(sqlite3.connect(self.db_path)) as conn:
                if conn.execute('SELECT 1 FROM czsc_runs WHERE run_id=?', (run_id,)).fetchone():
                    if kind == 'scan':
                        self._index_scan(conn,result,target,encoded)
                        conn.commit()
                    return result
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(target)
        summary = {key: result[key] for key in ("metrics", "coverage", "data_range", "strategy_version", "model_gate", "model_run_id", "category_counts", "analysis_method", "method_context", "status") if key in result}
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("INSERT INTO czsc_runs VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET summary=excluded.summary, report_path=excluded.report_path", (run_id, kind, result.get("symbol", ""), local_now().isoformat(), json.dumps(summary, ensure_ascii=False), target.relative_to(self.runs_root).as_posix()))
            if kind == 'scan': self._index_scan(conn,result,target,encoded)
            conn.commit()
        return result

    def list(self, kind: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            where, args = ("WHERE kind=?", [kind]) if kind else ("", [])
            rows = conn.execute(f"SELECT run_id,kind,symbol,created_at,summary FROM czsc_runs {where} ORDER BY created_at DESC LIMIT ?", [*args, min(max(limit, 1), 200)]).fetchall()
        return [{**dict(row), "summary": json.loads(row["summary"])} for row in rows]

    def get(self, run_id: str) -> dict[str, Any]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute("SELECT report_path FROM czsc_runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            raise KeyError(run_id)
        path = (self.runs_root / row[0]).resolve()
        if not path.is_relative_to(self.runs_root.resolve()):
            raise ValueError("Report path outside run storage")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _signature(path):
        stat=path.stat()
        return json.dumps([stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns])

    def _index_scan(self, conn, result, path, encoded):
        from my_strategy.storage.scan_view import _scan_row_view
        run_id=result['run_id']; sha=hashlib.sha256(encoded.encode('utf-8')).hexdigest()
        signature=self._signature(path)
        existing=conn.execute('SELECT report_sha256,report_signature FROM scan_views WHERE run_id=?',(run_id,)).fetchone()
        if existing == (sha,signature): return
        # Header + rows publish in the same transaction as the report index.
        header={key:value for key,value in result.items() if key!='rows'}
        conn.execute('INSERT OR REPLACE INTO scan_views VALUES(?,?,?,?,?)',
            (run_id,sha,signature,json.dumps(header,ensure_ascii=False,allow_nan=False),len(result.get('rows',[]))))
        conn.execute('DELETE FROM scan_view_rows WHERE run_id=?',(run_id,))
        rows=[]
        for ordinal,row in enumerate(result.get('rows',[])):
            view=_scan_row_view(row)
            category=row.get('category') or {'BUY':'buy','SELL':'sell','WAIT':'wait','HOLD':'hold'}.get(row.get('action'),'')
            direction=row.get('reference_action') or (row.get('method_evaluation') or {}).get('direction','')
            payload=json.dumps(view,ensure_ascii=False,allow_nan=False)
            rows.append((run_id,ordinal,row.get('symbol'),category,direction,payload.lower(),payload))
        conn.executemany('INSERT INTO scan_view_rows VALUES(?,?,?,?,?,?,?)',rows)

    def get_scan_page(self, run_id, **options):
        from filelock import FileLock
        token=hashlib.sha256(str(run_id).encode()).hexdigest()[:24]
        with FileLock(str(self.db_path)+'.'+token+'.lock'):
            return self._get_scan_page(run_id,**options)

    def _get_scan_page(self, run_id, *, page=None, page_size=40, query='', category='all', symbols=None):
        if page is not None and (page<1 or not 1<=page_size<=100): raise ValueError('扫描分页无效')
        with closing(sqlite3.connect(self.db_path,timeout=30)) as conn:
            record=conn.execute('SELECT kind,report_path FROM czsc_runs WHERE run_id=?',(run_id,)).fetchone()
            if not record or record[0]!='scan': raise KeyError(run_id)
            path=(self.runs_root/record[1]).resolve()
            if not path.is_relative_to(self.runs_root.resolve()): raise ValueError('Report path outside run storage')
            indexed=conn.execute('SELECT report_sha256,report_signature,header,row_count FROM scan_views WHERE run_id=?',(run_id,)).fetchone()
            if not indexed or indexed[1]!=self._signature(path):
                # Old reports are migrated lazily without changing their bytes.
                encoded=path.read_text(encoding='utf-8'); result=json.loads(encoded)
                if result.get('run_id')!=run_id: raise ValueError('Report identity mismatch')
                self._index_scan(conn,result,path,encoded);conn.commit()
            conn.execute('BEGIN')
            indexed=conn.execute('SELECT report_sha256,report_signature,header,row_count FROM scan_views WHERE run_id=?',(run_id,)).fetchone()
            clauses=['run_id=?'];args=[run_id]
            if query:
                clauses.append('instr(search_text,?)>0');args.append(query.lower())
            if category in ('exit','sell'):
                clauses.append("(category='exit' OR direction='SELL' OR category=?)");args.append(category)
            elif category not in ('all','favorites'):
                clauses.append('category=?');args.append(category)
            if symbols is not None:
                clauses.append('symbol IN ('+','.join('?' for _ in symbols)+')' if symbols else '0');args.extend(symbols)
            where=' WHERE '+' AND '.join(clauses)
            total=conn.execute('SELECT COUNT(*) FROM scan_view_rows'+where,args).fetchone()[0]
            limit=' LIMIT ? OFFSET ?' if page is not None else ''
            page_args=[page_size,(page-1)*page_size] if page is not None else []
            rows=conn.execute('SELECT payload FROM scan_view_rows'+where+' ORDER BY ordinal'+limit,[*args,*page_args]).fetchall()
            result={**json.loads(indexed[2]),'rows':[json.loads(r[0]) for r in rows],
                'scan_view':{'schema':'current_scan_view','compact':True,'analysis_mode':'reference',
                    'raw_evidence':'per_stock_analysis','report_sha256':indexed[0]}}
            if page is not None:
                result['scan_view'].update(page=page,page_size=page_size,total=total,total_rows=indexed[3],paged=True)
            conn.commit()
            return result
