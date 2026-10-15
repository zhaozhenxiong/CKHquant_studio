"""Immutable current board rows, two publication levels and process leases."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

import psutil

from my_strategy.core.paths import BASIC_FLUENT_DB_PATH
from my_strategy.core.run_context import stable_hash

VERSION = 1
APPLICATION_ID = 0x4B484246
TABLES = {'sync_runs', 'feature_generations', 'board_daily_features', 'stock_daily_features',
          'native_stock_summaries', 'board_structure_features', 'board_rankings', 'feature_input_refs'}
BOARD_COLUMNS = ('board_return_1','board_return_5','board_return_10','board_return_20',
    'board_advance_ratio','board_above_ma20_ratio','board_breakout20_ratio','board_amount',
    'board_market_amount_share','board_relative_amount20','board_volatility20','board_drawdown20')
STOCK_COLUMNS = tuple('industry_'+name[6:] for name in BOARD_COLUMNS)+(
    'relative_industry_1','relative_industry_5','relative_industry_20','industry_peer_rank20',
    'concept_return_1_mean','concept_return_5_mean','concept_return_20_mean','concept_advance_ratio_mean',
    'concept_positive_ratio20','concept_known_count')
RANK_POLICY = {'version': 'board_ranking_v1', 'activity': '50*average_rank_percentile(relative_amount20)+50*average_rank_percentile(abs(return1))',
               'percentile': '(average_1_based_rank-1)/(n-1);singleton=.5',
               'capital': 'qualified_member_amount_desc;tie=board_id', 'limit': 30}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _json(text):
    def invalid(value): raise ValueError('nonfinite Fluent JSON: '+value)
    return json.loads(text, parse_constant=invalid)


def _verified(text, sha):
    if hashlib.sha256(text.encode()).hexdigest()!=sha: raise ValueError('Fluent payload SHA drift')
    return _json(text)


def _finite(value):
    return value is not None and not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)


def rank_boards(rows):
    """Deterministic, equal weight, separate industry/concept populations."""
    result=[]
    for kind in ('industry','concept'):
        population=[row for row in rows if row.get('kind')==kind]
        active=[row for row in population if _finite(row.get('board_relative_amount20')) and _finite(row.get('board_return_1'))]
        percentiles={}
        for field in ('board_relative_amount20','board_return_1'):
            ordered=sorted((abs(row[field]) if field=='board_return_1' else row[field],row['board_id']) for row in active)
            start=0
            while start<len(ordered):
                end=start+1
                while end<len(ordered) and ordered[end][0]==ordered[start][0]: end+=1
                percentile=.5 if len(ordered)==1 else ((start+end-1)/2)/(len(ordered)-1)
                for _,board in ordered[start:end]: percentiles[(board,field)]=percentile
                start=end
        scored=[(50*(percentiles[(row['board_id'],'board_relative_amount20')]+percentiles[(row['board_id'],'board_return_1')]),row) for row in active]
        for position,(score,row) in enumerate(sorted(scored,key=lambda pair:(-pair[0],pair[1]['board_id']))[:30],1):
            result.append({'kind':kind,'ranking_type':'activity','rank':position,'board_id':row['board_id'],
                           'score':score,'capital_basis':'turnover_amount_proxy','policy':RANK_POLICY})
        capital=[row for row in population if _finite(row.get('board_amount')) and row['board_amount']>=0]
        for position,row in enumerate(sorted(capital,key=lambda item:(-item['board_amount'],item['board_id']))[:30],1):
            result.append({'kind':kind,'ranking_type':'capital','rank':position,'board_id':row['board_id'],
                           'score':row['board_amount'],'capital_basis':'turnover_amount_proxy','policy':RANK_POLICY})
    return result


_SCHEMA = f'''
CREATE TABLE sync_runs(sync_id TEXT PRIMARY KEY,fingerprint TEXT NOT NULL,status TEXT NOT NULL,
 owner_pid INTEGER,owner_started REAL,heartbeat REAL,lease_seconds REAL NOT NULL,payload_json TEXT NOT NULL,payload_sha TEXT NOT NULL);
CREATE INDEX sync_fingerprint ON sync_runs(fingerprint,heartbeat DESC);
CREATE UNIQUE INDEX one_active_sync ON sync_runs(fingerprint) WHERE status IN ('running','queued');
CREATE UNIQUE INDEX one_sync_owner ON sync_runs((1)) WHERE status IN ('running','queued');
CREATE TABLE feature_generations(generation_id TEXT PRIMARY KEY,cache_key TEXT NOT NULL,contract_hash TEXT NOT NULL,
 price_date TEXT NOT NULL,membership_date TEXT,identity_json TEXT NOT NULL,identity_sha TEXT NOT NULL,
 top_json TEXT NOT NULL,top_sha TEXT NOT NULL,bundle_sha TEXT NOT NULL,public_json TEXT NOT NULL,public_sha TEXT NOT NULL,
 market_ready INTEGER NOT NULL DEFAULT 0,full_ready INTEGER NOT NULL DEFAULT 0,market_active INTEGER NOT NULL DEFAULT 0,
 full_active INTEGER NOT NULL DEFAULT 0,native_identity_json TEXT,native_identity_sha TEXT,created_at REAL NOT NULL);
CREATE UNIQUE INDEX market_publication ON feature_generations(market_active) WHERE market_active=1;
CREATE UNIQUE INDEX full_publication ON feature_generations(full_active) WHERE full_active=1;
CREATE INDEX market_cache ON feature_generations(cache_key,contract_hash);
CREATE TABLE board_daily_features(generation_id TEXT NOT NULL,board_id TEXT NOT NULL,kind TEXT NOT NULL,price_date TEXT NOT NULL,
 board_return_1 REAL,board_return_5 REAL,board_return_10 REAL,board_return_20 REAL,board_advance_ratio REAL,board_above_ma20_ratio REAL,
 board_breakout20_ratio REAL,board_amount REAL,board_market_amount_share REAL,board_relative_amount20 REAL,board_volatility20 REAL,
 board_drawdown20 REAL,activity_score REAL,coverage REAL,effective_members INTEGER,derived_sha TEXT NOT NULL,payload_json TEXT NOT NULL,payload_sha TEXT NOT NULL,
 PRIMARY KEY(generation_id,board_id),FOREIGN KEY(generation_id) REFERENCES feature_generations(generation_id));
CREATE INDEX board_entity_date ON board_daily_features(board_id,price_date,generation_id);
CREATE TABLE stock_daily_features(generation_id TEXT NOT NULL,symbol TEXT NOT NULL,price_date TEXT NOT NULL,
 {','.join(name+' REAL' for name in STOCK_COLUMNS)},
 memberships_json TEXT NOT NULL,memberships_sha TEXT NOT NULL,payload_json TEXT NOT NULL,payload_sha TEXT NOT NULL,
 PRIMARY KEY(generation_id,symbol),FOREIGN KEY(generation_id) REFERENCES feature_generations(generation_id));
CREATE INDEX stock_entity_date ON stock_daily_features(symbol,price_date,generation_id);
CREATE TABLE native_stock_summaries(generation_id TEXT NOT NULL,symbol TEXT NOT NULL,price_date TEXT NOT NULL,
 native_cache_key TEXT,payload_json TEXT NOT NULL,payload_sha TEXT NOT NULL,PRIMARY KEY(generation_id,symbol,price_date),
 FOREIGN KEY(generation_id) REFERENCES feature_generations(generation_id));
CREATE INDEX native_entity_date ON native_stock_summaries(symbol,price_date,generation_id);
CREATE TABLE board_structure_features(generation_id TEXT NOT NULL,board_id TEXT NOT NULL,structure_basis TEXT NOT NULL,
 price_date TEXT NOT NULL,payload_json TEXT NOT NULL,payload_sha TEXT NOT NULL,PRIMARY KEY(generation_id,board_id,structure_basis),
 FOREIGN KEY(generation_id) REFERENCES feature_generations(generation_id));
CREATE TABLE board_rankings(generation_id TEXT NOT NULL,kind TEXT NOT NULL,ranking_type TEXT NOT NULL,rank INTEGER NOT NULL,
 board_id TEXT NOT NULL,score REAL NOT NULL,payload_json TEXT NOT NULL,payload_sha TEXT NOT NULL,
 PRIMARY KEY(generation_id,kind,ranking_type,rank),FOREIGN KEY(generation_id,board_id) REFERENCES board_daily_features(generation_id,board_id));
CREATE INDEX rank_read ON board_rankings(generation_id,kind,ranking_type,rank);
CREATE TABLE feature_input_refs(generation_id TEXT NOT NULL,ref_type TEXT NOT NULL,entity_id TEXT NOT NULL,
 payload_json TEXT NOT NULL,payload_sha TEXT NOT NULL,PRIMARY KEY(generation_id,ref_type,entity_id),
 FOREIGN KEY(generation_id) REFERENCES feature_generations(generation_id));
'''


class BoardFluentStore:
    def __init__(self, db_path=None, readonly=False):
        self.path=Path(db_path or BASIC_FLUENT_DB_PATH).expanduser().resolve();self.readonly=bool(readonly)
        if self.readonly:
            if self.path.is_file():
                with self._read() as conn:self._verify_schema(conn)
            return
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._write(initialize=True) as conn:
            tables={row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not tables and conn.execute('PRAGMA user_version').fetchone()[0]==0:
                for statement in _SCHEMA.split(';'):
                    if statement.strip():conn.execute(statement)
                for table in TABLES-{'sync_runs','feature_generations'}:
                    for operation in ('UPDATE','DELETE'):
                        conn.execute(f"CREATE TRIGGER immutable_{table}_{operation.lower()} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT,'immutable Fluent output'); END")
                conn.execute('''CREATE TRIGGER immutable_generation_inputs BEFORE UPDATE ON feature_generations
                    WHEN OLD.identity_json<>NEW.identity_json OR OLD.top_json<>NEW.top_json OR OLD.bundle_sha<>NEW.bundle_sha
                    OR OLD.public_json<>NEW.public_json OR OLD.cache_key<>NEW.cache_key OR OLD.contract_hash<>NEW.contract_hash
                    BEGIN SELECT RAISE(ABORT,'immutable Fluent generation input'); END''')
                conn.execute(f'PRAGMA user_version={VERSION}');conn.execute(f'PRAGMA application_id={APPLICATION_ID}')
            self._verify_schema(conn)

    @staticmethod
    def _verify_schema(conn):
        if (conn.execute('PRAGMA user_version').fetchone()[0]!=VERSION or conn.execute('PRAGMA application_id').fetchone()[0]!=APPLICATION_ID or
            {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}!=TABLES):
            raise ValueError('Fluent database schema identity mismatch')

    @contextmanager
    def _read(self):
        if not self.path.is_file():yield None;return
        conn=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,timeout=30);conn.row_factory=sqlite3.Row
        conn.execute('PRAGMA query_only=ON')
        try:yield conn
        finally:conn.close()

    @contextmanager
    def _write(self, initialize=False):
        if self.readonly:raise ValueError('Fluent store is read-only')
        conn=sqlite3.connect(str(self.path),timeout=30);conn.row_factory=sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON');conn.execute('PRAGMA busy_timeout=30000')
        if initialize:conn.execute('PRAGMA journal_mode=WAL');conn.execute('PRAGMA synchronous=FULL')
        try:
            conn.execute('BEGIN IMMEDIATE');yield conn;conn.commit()
        except BaseException:conn.rollback();raise
        finally:conn.close()

    @staticmethod
    def _put(conn,table,columns,values):
        placeholders=','.join('?' for _ in columns)
        conn.execute(f"INSERT OR IGNORE INTO {table} ({','.join(columns)}) VALUES({placeholders})",values)
        keys={'board_daily_features':('generation_id','board_id'),'stock_daily_features':('generation_id','symbol'),
            'native_stock_summaries':('generation_id','symbol','price_date'),'board_structure_features':('generation_id','board_id','structure_basis'),
            'board_rankings':('generation_id','kind','ranking_type','rank'),'feature_input_refs':('generation_id','ref_type','entity_id')}[table]
        lookup=dict(zip(columns,values));prior=conn.execute(f"SELECT * FROM {table} WHERE "+' AND '.join(key+'=?' for key in keys),[lookup[key] for key in keys]).fetchone()
        if any(prior[key]!=lookup[key] for key in columns):raise ValueError('same immutable Fluent input produced different output')

    @staticmethod
    def _generation(conn,generation_id):
        row=conn.execute('SELECT * FROM feature_generations WHERE generation_id=?',(generation_id,)).fetchone()
        if row is None:return None
        public=_verified(row['public_json'],row['public_sha'])
        identity=_verified(row['identity_json'],row['identity_sha'])
        if digest(identity)!=row['generation_id'] or identity['cache_key']!=row['cache_key'] or identity['contract_hash']!=row['contract_hash']:
            raise ValueError('Fluent generation identity drift')
        if public['price_date']!=row['price_date']:raise ValueError('Fluent generation date drift')
        native=None
        if row['native_identity_json'] is not None:native=_verified(row['native_identity_json'],row['native_identity_sha'])
        return {**public,'generation_id':generation_id,'market_ready':bool(row['market_ready']),'full_ready':bool(row['full_ready']),
            'market_published':bool(row['market_active']),'full_published':bool(row['full_active']),
            'native_identity':native,'created_at':row['created_at']}

    def get_generation(self,generation_id):
        with self._read() as conn:return self._generation(conn,generation_id) if conn else None

    def current_generation(self,level='market'):
        if level not in {'market','full'}:raise ValueError('unknown Fluent publication level')
        with self._read() as conn:
            if conn is None:return None
            row=conn.execute(f'SELECT generation_id FROM feature_generations WHERE {level}_active=1').fetchone()
            return self._generation(conn,row[0]) if row else None

    def ready_full_generation(self,price_date):
        """Find an immutable completed index even after its publication moved."""
        with self._read() as conn:
            if conn is None:return None
            row=conn.execute('SELECT generation_id FROM feature_generations WHERE price_date=? AND full_ready=1 ORDER BY created_at DESC LIMIT 1',(price_date,)).fetchone()
            return self._generation(conn,row[0]) if row else None

    def get_native_component(self,generation_id,*,allow_partial=False):
        """Read verified indexed summaries and refs; never decode producer blobs."""
        generation=self.verify_generation(generation_id,level='market' if allow_partial else 'full')
        with self._read() as conn:
            if allow_partial:self._verify_full_population(conn,generation_id,generation['native_identity'])
            summaries={row['symbol']:self._row(row) for row in conn.execute('SELECT * FROM native_stock_summaries WHERE generation_id=? ORDER BY symbol',(generation_id,))}
            refs=[_verified(row['payload_json'],row['payload_sha']) for row in conn.execute("SELECT payload_json,payload_sha FROM feature_input_refs WHERE generation_id=? AND ref_type='native_stock' ORDER BY entity_id",(generation_id,))]
            coverage=conn.execute("SELECT payload_json,payload_sha FROM feature_input_refs WHERE generation_id=? AND ref_type='native' AND entity_id='native_coverage'",(generation_id,)).fetchone()
            diagnostics=_verified(coverage['payload_json'],coverage['payload_sha']) if coverage else {}
            return {'identity':generation['native_identity'],'summaries':summaries,'input_refs':refs,'complete':generation['full_ready'],'diagnostics':diagnostics}

    def save_market(self,bundle,*,contract=None,sync_id=None,publish=True,migration=None,publication_inputs=None):
        if bundle.get('mode')!='current_reference' or bundle.get('training_eligible') is not False or bundle.get('qualification')!='pending':
            raise ValueError('Fluent accepts current reference only')
        boards=bundle['boards'];stocks=bundle['stocks']
        if len({row['board_id'] for row in boards})!=len(boards) or len({row['symbol'] for row in stocks})!=len(stocks):raise ValueError('duplicate Fluent entity')
        if any(row.get('training_eligible') is not False or row.get('mode')!='current_reference' or row.get('price_date')!=bundle['price_date'] for row in boards+stocks):
            raise ValueError('Fluent row is historical or incompatible')
        contract=contract or {'rank_policy':RANK_POLICY};contract_hash=digest(contract)
        identity={'cache_key':bundle['cache_key'],'contract_hash':contract_hash,
                  'publication_inputs':publication_inputs or {}};generation_id=digest(identity)
        top={key:value for key,value in bundle.items() if key not in {'boards','stocks'}}
        public={key:value for key,value in top.items() if key not in {'cache_identity'}}
        public.update(board_count=len(boards),stock_count=len(stocks),ranking_policy=RANK_POLICY,
                      fluent_contract=contract,publication_inputs=publication_inputs or {},
                      capital_basis='turnover_amount_proxy',training_started=False)
        rankings=rank_boards(boards);scores={row['board_id']:row['score'] for row in rankings if row['ranking_type']=='activity'}
        # Scores for boards outside Top30 are still queryable.
        for kind in ('industry','concept'):
            candidates=[row for row in boards if row['kind']==kind]
            complete=[row for row in candidates if _finite(row.get('board_relative_amount20')) and _finite(row.get('board_return_1'))]
            for row in complete:
                if row['board_id'] not in scores:
                    def percentile(field):
                        own=abs(row[field]) if field=='board_return_1' else row[field]
                        values=[abs(item[field]) if field=='board_return_1' else item[field] for item in complete]
                        rank=sum(value<own for value in values)+(sum(value==own for value in values)-1)/2
                        return .5 if len(values)==1 else rank/(len(values)-1)
                    scores[row['board_id']]=50*(percentile('board_relative_amount20')+percentile('board_return_1'))
        with self._write() as conn:
            encoded=canonical(identity);top_json=canonical(top);public_json=canonical(public)
            conn.execute('''INSERT OR IGNORE INTO feature_generations(generation_id,cache_key,contract_hash,price_date,membership_date,
                identity_json,identity_sha,top_json,top_sha,bundle_sha,public_json,public_sha,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (generation_id,bundle['cache_key'],contract_hash,bundle['price_date'],bundle.get('membership_date'),encoded,digest(identity),
                 top_json,digest(top),digest(bundle),public_json,digest(public),time.time()))
            prior=conn.execute('SELECT bundle_sha,top_json,public_json FROM feature_generations WHERE generation_id=?',(generation_id,)).fetchone()
            if prior['bundle_sha']!=digest(bundle) or prior['top_json']!=top_json or prior['public_json']!=public_json:raise ValueError('immutable Fluent generation drift')
            for row in boards:
                derived={'activity_score':scores.get(row['board_id']),'coverage':row.get('coverage'),'effective_members':row.get('effective_members')}
                fields=('generation_id','board_id','kind','price_date',*BOARD_COLUMNS,'activity_score','coverage','effective_members','derived_sha','payload_json','payload_sha')
                values=(generation_id,row['board_id'],row['kind'],row['price_date'],*[row.get(name) for name in BOARD_COLUMNS],
                        scores.get(row['board_id']),row.get('coverage'),row.get('effective_members'),digest(derived),canonical(row),digest(row))
                self._put(conn,'board_daily_features',fields,values)
            for row in stocks:
                industries=sorted({item['board_id'] for item in row.get('industry_matches',[])})
                memberships={'all_industry_ids':industries,'board_ids':sorted(set(industries+row.get('concept_ids',[])))}
                self._put(conn,'stock_daily_features',('generation_id','symbol','price_date',*STOCK_COLUMNS,'memberships_json','memberships_sha','payload_json','payload_sha'),
                          (generation_id,row['symbol'],row['price_date'],*[row.get(name) for name in STOCK_COLUMNS],
                           canonical(memberships),digest(memberships),canonical(row),digest(row)))
            for row in rankings:self._put(conn,'board_rankings',('generation_id','kind','ranking_type','rank','board_id','score','payload_json','payload_sha'),
                                         (generation_id,row['kind'],row['ranking_type'],row['rank'],row['board_id'],row['score'],canonical(row),digest(row)))
            refs=[('market','bundle',{'cache_identity':bundle['cache_identity'],'cache_key':bundle['cache_key']})]
            if migration is not None:refs.append(('migration','legacy_bulk',migration))
            for ref_type,entity_id,value in refs:self._put(conn,'feature_input_refs',('generation_id','ref_type','entity_id','payload_json','payload_sha'),
                                                         (generation_id,ref_type,entity_id,canonical(value),digest(value)))
            conn.execute('UPDATE feature_generations SET market_ready=1 WHERE generation_id=?',(generation_id,))
            if publish:self._publish(conn,generation_id,'market')
        return generation_id

    @staticmethod
    def _publish(conn,generation_id,level):
        row=conn.execute('SELECT market_ready,full_ready,price_date,created_at FROM feature_generations WHERE generation_id=?',(generation_id,)).fetchone()
        if row is None or not row[level+'_ready']:raise ValueError('Fluent component is not ready for publication')
        active=conn.execute(f'SELECT generation_id,price_date,created_at FROM feature_generations WHERE {level}_active=1').fetchone()
        if active and active['price_date']>row['price_date']:raise ValueError('cannot replace latest Fluent publication with an older price date')
        conn.execute(f'UPDATE feature_generations SET {level}_active=0 WHERE {level}_active=1')
        conn.execute(f'UPDATE feature_generations SET {level}_active=1 WHERE generation_id=?',(generation_id,))
        if level=='market':conn.execute('UPDATE feature_generations SET full_active=0 WHERE generation_id<>?',(generation_id,))

    def publish(self,generation_id,level='market'):
        if level not in {'market','full'}:raise ValueError('unknown publication level')
        with self._write() as conn:
            if level=='full':
                generation=self._generation(conn,generation_id)
                if generation is None:raise ValueError('Fluent generation is missing')
                self._verify_full_population(conn,generation_id,generation['native_identity'])
            self._publish(conn,generation_id,level)

    def publish_full(self,generation_id):
        """Publish both levels of one generation in a single SQLite transaction."""
        with self._write() as conn:
            ready=conn.execute('SELECT full_ready FROM feature_generations WHERE generation_id=?',(generation_id,)).fetchone()
            if ready is None or not ready[0]:raise ValueError('full Fluent component is not ready')
            generation=self._generation(conn,generation_id)
            self._verify_full_population(conn,generation_id,generation['native_identity'])
            self._publish(conn,generation_id,'market');self._publish(conn,generation_id,'full')

    def load_market(self,cache_key,contract=None,generation_id=None):
        contract_hash=digest(contract or {'rank_policy':RANK_POLICY})
        with self._read() as conn:
            if conn is None:return None
            sql='SELECT * FROM feature_generations WHERE cache_key=? AND contract_hash=? AND market_ready=1'
            params=[cache_key,contract_hash]
            if generation_id is not None:sql+=' AND generation_id=?';params.append(generation_id)
            row=conn.execute(sql+' ORDER BY created_at DESC LIMIT 1',params).fetchone()
            if row is None:return None
            self._generation(conn,row['generation_id']);top=_verified(row['top_json'],row['top_sha'])
            boards=[self._row(item,board=True) for item in conn.execute('SELECT * FROM board_daily_features WHERE generation_id=? ORDER BY board_id',(row['generation_id'],))]
            stocks=[self._row(item) for item in conn.execute('SELECT * FROM stock_daily_features WHERE generation_id=? ORDER BY symbol',(row['generation_id'],))]
            result={**top,'boards':boards,'stocks':stocks}
            if digest(result)!=row['bundle_sha']:raise ValueError('Fluent market aggregate SHA drift')
            return result

    @staticmethod
    def _row(row,board=False):
        value=_verified(row['payload_json'],row['payload_sha'])
        entity='board_id' if 'board_id' in row.keys() else 'symbol'
        if value.get(entity)!=row[entity] or value.get('price_date')!=row['price_date']:raise ValueError('Fluent indexed row identity drift')
        if board and (value.get('kind')!=row['kind'] or any(value.get(name)!=row[name] for name in BOARD_COLUMNS)):
            raise ValueError('Fluent numeric columns drift')
        if board and digest({key:row[key] for key in ('activity_score','coverage','effective_members')})!=row['derived_sha']:
            raise ValueError('Fluent derived metadata SHA drift')
        if all(name in row.keys() for name in STOCK_COLUMNS) and any(value.get(name)!=row[name] for name in STOCK_COLUMNS):
            raise ValueError('Fluent stock numeric columns drift')
        return value

    def verify_generation(self,generation_id,level='market'):
        generation=self.get_generation(generation_id)
        if generation is None or not generation['market_ready']:raise ValueError('Fluent generation is missing or incomplete')
        self.load_market(generation['cache_key'],generation['fluent_contract'],generation_id=generation_id)
        with self._read() as conn:
            for table in ('feature_input_refs','board_rankings'):
                for row in conn.execute('SELECT payload_json,payload_sha FROM '+table+' WHERE generation_id=?',(generation_id,)):
                    _verified(row['payload_json'],row['payload_sha'])
            if level=='full':
                if not generation['full_ready']:raise ValueError('Fluent full component is not ready')
                for table in ('native_stock_summaries','board_structure_features'):
                    for row in conn.execute('SELECT * FROM '+table+' WHERE generation_id=?',(generation_id,)):self._row(row)
                self._verify_full_population(conn,generation_id,generation['native_identity'])
        return generation

    @staticmethod
    def _verify_full_population(conn,generation_id,identity):
        market_boards={row[0] for row in conn.execute('SELECT board_id FROM board_daily_features WHERE generation_id=?',(generation_id,))}
        native_boards={row[0] for row in conn.execute('SELECT board_id FROM board_structure_features WHERE generation_id=?',(generation_id,))}
        if native_boards!=market_boards:raise ValueError('native board component omitted required entities')
        if not isinstance(identity,dict):raise ValueError('native component identity missing')
        if identity.get('format') in {'native_summary_collection_v1','incremental_native_summary_collection_v1'}:
            required=identity.get('required_symbols')
            if not isinstance(required,list) or any(not isinstance(symbol,str) for symbol in required) or required!=sorted(set(required)):
                raise ValueError('native required symbol population identity invalid')
            actual=[row[0] for row in conn.execute('SELECT symbol FROM native_stock_summaries WHERE generation_id=? ORDER BY symbol',(generation_id,))]
            if actual!=required:raise ValueError('native summary component omitted required entities')
            refs=[]
            for row in conn.execute("SELECT entity_id,payload_json,payload_sha FROM feature_input_refs WHERE generation_id=? AND ref_type='native_stock' ORDER BY entity_id",(generation_id,)):
                value=_verified(row['payload_json'],row['payload_sha'])
                if value.get('entity_id')!=row['entity_id']:raise ValueError('native input reference indexed identity drift')
                refs.append(value)
            if stable_hash(refs)!=identity.get('input_refs_hash'):raise ValueError('native input references aggregate SHA drift')

    def _selected(self,conn,generation_id):
        if generation_id is not None:return generation_id
        row=conn.execute('SELECT generation_id FROM feature_generations WHERE market_active=1').fetchone()
        return row[0] if row else None

    def board(self,board_id,generation_id=None,include_structure=True):
        with self._read() as conn:
            if conn is None:return None
            generation_id=self._selected(conn,generation_id)
            if generation_id is None:return None
            generation=self._generation(conn,generation_id)
            row=conn.execute('SELECT * FROM board_daily_features WHERE generation_id=? AND board_id=?',(generation_id,board_id)).fetchone()
            if row is None:return None
            value={**self._row(row,board=True),'generation_id':generation_id,'activity_score':row['activity_score'],'capital_basis':'turnover_amount_proxy'}
            if include_structure:
                structure=conn.execute('SELECT * FROM board_structure_features WHERE generation_id=? AND board_id=? ORDER BY structure_basis',(generation_id,board_id)).fetchall()
                value['native_structure']=[self._row(item) for item in structure] if structure else None
                value['native_component_complete']=generation['full_ready']
            return value

    def stock(self,symbol,generation_id=None):
        with self._read() as conn:
            if conn is None:return None
            generation_id=self._selected(conn,generation_id)
            if generation_id is None:return None
            self._generation(conn,generation_id)
            row=conn.execute('SELECT * FROM stock_daily_features WHERE generation_id=? AND symbol=?',(generation_id,symbol)).fetchone()
            if row is None:return None
            value={**self._row(row),'generation_id':generation_id}
            value.update(_verified(row['memberships_json'],row['memberships_sha']))
            native=conn.execute('SELECT * FROM native_stock_summaries WHERE generation_id=? AND symbol=?',(generation_id,symbol)).fetchone()
            value['native_summary']=self._row(native) if native else None
            return value

    def boards(self,kind,generation_id=None):
        """Read SHA-verified published market rows for visual comparisons."""
        if kind not in {'industry','concept'}:raise ValueError('unknown board kind')
        with self._read() as conn:
            if conn is None:return []
            generation_id=self._selected(conn,generation_id)
            if generation_id is None:return []
            generation=self._generation(conn,generation_id)
            if not generation or not generation['market_ready']:return []
            return [{**self._row(row,board=True),'generation_id':generation_id,
                'activity_score':row['activity_score']} for row in conn.execute(
                    'SELECT * FROM board_daily_features WHERE generation_id=? AND kind=? ORDER BY board_id',
                    (generation_id,kind))]

    def rankings(self,kind,ranking_type='activity',limit=30,generation_id=None):
        if kind not in {'industry','concept'} or ranking_type not in {'activity','capital'}:raise ValueError('unknown board ranking')
        with self._read() as conn:
            if conn is None:return []
            generation_id=self._selected(conn,generation_id)
            if generation_id is None:return []
            self._generation(conn,generation_id);result=[]
            for row in conn.execute('SELECT * FROM board_rankings WHERE generation_id=? AND kind=? AND ranking_type=? ORDER BY rank LIMIT ?',
                                    (generation_id,kind,ranking_type,min(30,max(0,int(limit))))):
                ranking=_verified(row['payload_json'],row['payload_sha'])
                if any(ranking.get(key)!=row[key] for key in ('board_id','rank','kind','ranking_type','score')):raise ValueError('Fluent ranking index drift')
                board=conn.execute('SELECT * FROM board_daily_features WHERE generation_id=? AND board_id=?',(generation_id,row['board_id'])).fetchone()
                result.append({**self._row(board,board=True),**ranking,'generation_id':generation_id,'activity_score':board['activity_score']})
            return result

    def attach_native(self,generation_id,native_result,board_rows,*,publish=True,allow_partial=False):
        generation=self.get_generation(generation_id)
        if generation is None or not generation['market_ready']:raise ValueError('market generation is not ready')
        identity=native_result['identity']
        complete=native_result.get('complete',identity.get('complete')) is True
        partial_allowed=allow_partial and identity.get('format')=='incremental_native_summary_collection_v1'
        if (identity.get('price_date')!=generation['price_date'] or (not complete and not partial_allowed) or
            identity.get('mode')!='current_reference' or identity.get('training_eligible') is not False):
            raise ValueError('native component is incomplete or has a different price date')
        summaries=native_result.get('summaries',native_result.get('by_symbol',{}))
        summaries=list(summaries.values()) if isinstance(summaries,dict) else summaries
        with self._write() as conn:
            existing=conn.execute('SELECT native_identity_json,native_identity_sha FROM feature_generations WHERE generation_id=?',(generation_id,)).fetchone()
            if existing['native_identity_json'] is not None and _verified(existing['native_identity_json'],existing['native_identity_sha'])!=identity:
                raise ValueError('same market generation cannot mix native identities')
            for row in summaries:
                if row.get('price_date')!=generation['price_date'] or row.get('training_eligible',False) is not False:raise ValueError('native summary is not current reference')
                if row.get('row_hash')!=stable_hash({key:value for key,value in row.items() if key!='row_hash'}):raise ValueError('native summary row hash drift')
                self._put(conn,'native_stock_summaries',('generation_id','symbol','price_date','native_cache_key','payload_json','payload_sha'),
                    (generation_id,row['symbol'],row['price_date'],row.get('native_cache_key',row.get('cache_key')),canonical(row),digest(row)))
            for row in board_rows:
                if row.get('price_date')!=generation['price_date'] or row.get('training_eligible') is not False:raise ValueError('board native date mismatch')
                if row.get('row_hash')!=stable_hash({key:value for key,value in row.items() if key!='row_hash'}):raise ValueError('board native row hash drift')
                if conn.execute('SELECT 1 FROM board_daily_features WHERE generation_id=? AND board_id=?',(generation_id,row['board_id'])).fetchone() is None:
                    raise ValueError('board native entity missing from market generation')
                basis=row.get('structure_basis','member_native')
                self._put(conn,'board_structure_features',('generation_id','board_id','structure_basis','price_date','payload_json','payload_sha'),
                    (generation_id,row['board_id'],basis,row['price_date'],canonical(row),digest(row)))
            expected=conn.execute('SELECT COUNT(*) FROM board_daily_features WHERE generation_id=?',(generation_id,)).fetchone()[0]
            if len({row['board_id'] for row in board_rows})!=expected:raise ValueError('native board component omitted required entities')
            for entity,value in [('native_component',identity),('native_coverage',native_result.get('diagnostics',{}))]:
                self._put(conn,'feature_input_refs',('generation_id','ref_type','entity_id','payload_json','payload_sha'),
                    (generation_id,'native',entity,canonical(value),digest(value)))
            refs=native_result.get('input_refs',[])
            refs=list(refs.values()) if isinstance(refs,dict) else refs
            for value in refs:
                self._put(conn,'feature_input_refs',('generation_id','ref_type','entity_id','payload_json','payload_sha'),
                    (generation_id,'native_stock',value['entity_id'],canonical(value),digest(value)))
            self._verify_full_population(conn,generation_id,identity)
            conn.execute('UPDATE feature_generations SET native_identity_json=?,native_identity_sha=?,full_ready=? WHERE generation_id=?',
                         (canonical(identity),digest(identity),int(complete),generation_id))
            if publish:
                if not complete:raise ValueError('partial native component cannot publish full')
                current=conn.execute('SELECT generation_id FROM feature_generations WHERE market_active=1').fetchone()
                if not current or current[0]!=generation_id:raise ValueError('cannot publish full component for a different market generation')
                self._publish(conn,generation_id,'full')

    def status(self):
        with self._read() as conn:
            counts={table:conn.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in TABLES} if conn else {}
        return {'available':bool(counts),'path':str(self.path),'counts':counts,'market':self.current_generation('market'),
                'full':self.current_generation('full'),'latest_sync':self.latest_sync()}

    @staticmethod
    def _owner():return os.getpid(),psutil.Process().create_time()

    @staticmethod
    def _alive(pid,started):
        try:return abs(psutil.Process(pid).create_time()-started)<.02 and psutil.Process(pid).is_running()
        except (psutil.Error,TypeError,ValueError):return False

    @staticmethod
    def _sync(row):
        if row is None:return None
        value=_verified(row['payload_json'],row['payload_sha'])
        if any(value.get(key)!=row[key] for key in ('sync_id','fingerprint','status','owner_pid','owner_started','heartbeat','lease_seconds')):
            raise ValueError('Fluent sync lease index drift')
        return value

    def acquire_sync(self,fingerprint,input_identity,*,sync_id=None,parent_update_id=None,lease_seconds=120,reuse_completed=True):
        if not _finite(lease_seconds) or lease_seconds<=0:raise ValueError('positive sync lease timeout required')
        pid,started=self._owner();now=time.time()
        with self._write() as conn:
            row=conn.execute('SELECT * FROM sync_runs WHERE '+('sync_id=?' if sync_id else 'fingerprint=? ORDER BY heartbeat DESC LIMIT 1'),(sync_id or fingerprint,)).fetchone()
            value=self._sync(row)
            active=conn.execute("SELECT * FROM sync_runs WHERE status IN ('running','queued')").fetchone()
            if active is not None and (value is None or active['sync_id']!=value['sync_id']):
                blocker=self._sync(active)
                if self._alive(blocker['owner_pid'],blocker['owner_started']):
                    return {**blocker,'owned':False,'reused':False,'blocked_by_active_sync':True}
                blocker.update(status='failed',error='owner_process_exited',finished_at=now)
                conn.execute('UPDATE sync_runs SET status=?,payload_json=?,payload_sha=? WHERE sync_id=?',
                             ('failed',canonical(blocker),digest(blocker),blocker['sync_id']))
            if value is not None:
                if value['fingerprint']!=fingerprint or value['input_identity']!=input_identity:raise ValueError('resume sync frozen input mismatch')
                if value['status']=='cancelled':
                    if sync_id is not None:raise ValueError('cancelled sync is terminal; create a new explicit plan')
                    value=None
            if value is not None:
                if value['status']=='completed':
                    if reuse_completed:return {**value,'reused':True,'owned':False}
                    if sync_id is not None:raise ValueError('completed sync cannot be overwritten')
                    value=None
            if value is not None:
                alive=self._alive(value['owner_pid'],value['owner_started'])
                mine=value['owner_pid']==pid and abs(value['owner_started']-started)<.02
                if value['status'] in {'running','queued'} and alive:
                    return {**value,'reused':True,'owned':False,'lease_stalled':now-value['heartbeat']>value['lease_seconds']}
            else:value={'sync_id':sync_id or uuid.uuid4().hex,'fingerprint':fingerprint,'input_identity':input_identity,
                        'parent_update_id':parent_update_id,'created_at':now,'stage':'queued','checkpoint':{},'result':None,'error':None}
            value.update(status='running',owner_pid=pid,owner_started=started,heartbeat=now,lease_seconds=float(lease_seconds))
            text=canonical(value)
            conn.execute('INSERT INTO sync_runs VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(sync_id) DO UPDATE SET status=excluded.status,owner_pid=excluded.owner_pid,owner_started=excluded.owner_started,heartbeat=excluded.heartbeat,lease_seconds=excluded.lease_seconds,payload_json=excluded.payload_json,payload_sha=excluded.payload_sha',
                         (value['sync_id'],fingerprint,value['status'],pid,started,now,float(lease_seconds),text,digest(value)))
            return {**value,'reused':row is not None,'owned':True}

    def _update_sync(self,sync_id,**changes):
        pid,started=self._owner()
        with self._write() as conn:
            value=self._sync(conn.execute('SELECT * FROM sync_runs WHERE sync_id=?',(sync_id,)).fetchone())
            if value is None:raise KeyError(sync_id)
            if value['owner_pid']!=pid or abs(value['owner_started']-started)>=.02:raise ValueError('sync lease belongs to another process attempt')
            if value['status'] not in {'running','queued'}:raise ValueError('sync lease is no longer active')
            if changes.get('checkpoint') is not None:changes['checkpoint']={**value.get('checkpoint',{}),**changes['checkpoint']}
            value.update(changes,heartbeat=time.time());text=canonical(value)
            conn.execute('UPDATE sync_runs SET status=?,heartbeat=?,payload_json=?,payload_sha=? WHERE sync_id=?',
                         (value['status'],value['heartbeat'],text,digest(value),sync_id))
        return value

    def heartbeat_sync(self,sync_id,stage=None,checkpoint=None):
        changes={}
        if stage is not None:changes['stage']=stage
        if checkpoint is not None:changes['checkpoint']=checkpoint
        return self._update_sync(sync_id,**changes)

    def finish_sync(self,sync_id,status='completed',result=None,error=None):
        if status not in {'completed','partial','failed','cancelled'}:raise ValueError('unknown sync outcome')
        return self._update_sync(sync_id,status=status,result=result,error=error,finished_at=time.time())

    def get_sync(self,sync_id):
        with self._read() as conn:return self._sync(conn.execute('SELECT * FROM sync_runs WHERE sync_id=?',(sync_id,)).fetchone()) if conn else None

    def latest_sync(self):
        with self._read() as conn:return self._sync(conn.execute('SELECT * FROM sync_runs ORDER BY heartbeat DESC LIMIT 1').fetchone()) if conn else None
