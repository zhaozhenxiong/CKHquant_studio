"""Read-only views of saved method scans and verified board membership.

No model invocation, result-index migration, cache publication or raw-price read
occurs here. Current membership visible at the cutoff remains a reference, not
verified historical membership or a historical board backtest.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import closing
from copy import deepcopy
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import threading

from my_strategy.core.run_context import stable_hash
from my_strategy.services.board_calculation_scope import CALCULATION_SCOPE, calculation_membership, is_calculation_symbol
from my_strategy.services.method_display import COMBINED, classify_method_row
from my_strategy.storage.board_knowledge import BoardKnowledgeStore, _instant
from my_strategy.storage.scan_view import _scan_row_view

SCHEMA = 'board_method_summary_v1'
CATEGORIES = frozenset({'all', 'buy', 'sell', 'wait', 'unavailable', 'failed', 'deferred', 'not_in_scan'})
POLICY = {'minimum_members': 5, 'minimum_complete_fraction': .8, 'weighting': 'equal_member'}
LIMITATIONS = [
    '当前成员仅作为截止时点可见的成员参考，未核验历史成员，不用于历史训练或回测。',
    '板块成员可能重叠，板块计数与统计不能相加作为全市场统计。',
    '采样上涨比例是未校准模型采样频率，不是胜率或收益保证。',
    '预测统计是成员预测摘要的横截面统计，不是板块 Kronos 推理或联合收益区间。',
    '方法方向是截止日研究参考，不是首次触发、目标仓位或模拟成交。',
]



def _json(encoded):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate saved JSON key')
            result[key] = value
        return result
    return json.loads(encoded, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('non-finite saved JSON')))


def _day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('invalid method cutoff date')
    return value


def _signature(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _page(rows, page, page_size, query='', category='all', symbols=None):
    if type(page) is not int or page < 1 or type(page_size) is not int or not 1 <= page_size <= 100:
        raise ValueError('invalid method result pagination')
    if category not in CATEGORIES:
        raise ValueError('invalid method result category')
    query = query.strip().casefold()
    selected = []
    for row in rows:
        view = row.get('method_view') or {}
        actual = view['direction'].lower() if view.get('valid') else view.get('status', 'not_in_scan')
        if category != 'all' and actual != category:
            continue
        if symbols is not None and row['symbol'] not in symbols:
            continue
        if query and query not in ' '.join(str(row.get(key) or '') for key in ('symbol', 'name')).casefold():
            continue
        selected.append(row)
    total = len(selected)
    return deepcopy(selected[(page-1)*page_size:page*page_size]), {
        'page': page, 'page_size': page_size, 'total': total, 'total_rows': len(rows),
        'pages': math.ceil(total/page_size), 'paged': True}


def summarize(rows, expected=None):
    """Count genuine valid WAIT separately from all missing/terminal branches."""
    expected = len(rows) if expected is None else expected
    counts = {key: 0 for key in ('matched', 'complete', 'buy', 'sell', 'wait', 'unavailable', 'failed', 'deferred', 'not_in_scan')}
    medians, fractions, scores = [], [], []
    source_counts = {key: 0 for key in ('complete', 'partial', 'failed', 'deferred')}
    for row in rows:
        view = row.get('method_view')
        if not view or view.get('status') == 'not_in_scan':
            counts['not_in_scan'] += 1
            continue
        counts['matched'] += 1
        if view.get('source_status') in source_counts:
            source_counts[view['source_status']] += 1
        if view['valid']:
            counts['complete'] += 1
            counts[view['direction'].lower()] += 1
        else:
            status = view['status'] if view['status'] in {'failed', 'deferred'} else 'unavailable'
            counts[status] += 1
        if view.get('forecast_valid'):
            medians.append(view['forecast_summary']['terminal_return_median'])
            fractions.append(view['forecast_summary']['up_path_fraction'])
        if view.get('score_valid'):
            scores.append(view['score_value'])
    counts['expected'] = expected
    ratio = lambda number: number / expected if expected else None
    valid_ratio = lambda number: number / counts['complete'] if counts['complete'] else None
    return {'counts': counts, 'source_status_counts': source_counts,
        'coverage': {'method_complete_ratio': ratio(counts['complete']), 'forecast_ratio': ratio(len(medians)), 'score_ratio': ratio(len(scores))},
        'direction_ratios': {'denominator': counts['complete'], 'buy': valid_ratio(counts['buy']), 'sell': valid_ratio(counts['sell']), 'wait': valid_ratio(counts['wait'])},
        'forecast': {'count': len(medians), 'denominator': 'forecast_valid_members',
            'mean_member_return_median': statistics.fmean(medians) if medians else None,
            'median_member_return_median': statistics.median(medians) if medians else None,
            'mean_member_up_path_fraction': statistics.fmean(fractions) if fractions else None,
            'semantics': 'member_cross_section_not_board_forecast', 'calibrated_probability': False,
            'win_rate': None, 'joint_return_interval': None},
        'score': {'count': len(scores), 'denominator': 'score_valid_members', 'mean': statistics.fmean(scores) if scores else None}}


class BoardMethodSummary:
    """Bounded in-memory projections; SQLite and saved reports are read only."""
    def __init__(self, result_db, runs_root, kb_path=None, *, cache_bytes=96*1024*1024, cache_entries=3):
        self.result_db = Path(result_db)
        self.runs_root = Path(runs_root).resolve()
        self.kb_path = kb_path
        self.cache_bytes = max(0, cache_bytes)
        self.cache_entries = max(0, cache_entries)
        self._cache = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    def _connect(self):
        con = sqlite3.connect(self.result_db.resolve().as_uri()+'?mode=ro', uri=True)
        con.row_factory = sqlite3.Row
        con.execute('PRAGMA query_only=ON')
        return con

    def _remember(self, key, value):
        """Reports and membership projections share the same LRU/byte budget."""
        size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode())
        if size <= self.cache_bytes and self.cache_entries:
            while self._cache and (len(self._cache) >= self.cache_entries or self._bytes+size > self.cache_bytes):
                _, (_, removed) = self._cache.popitem(last=False)
                self._bytes -= removed
            self._cache[key] = (value, size)
            self._bytes += size

    def _knowledge_signature(self):
        # A committed SQLite write can live entirely in WAL. Main-file mtime
        # alone would hide new manifests, memberships or corrupted source proofs.
        # Include inode/ctime as well to notice replacement and same-size edits.
        path = BoardKnowledgeStore(self.kb_path, readonly=True).path
        signatures = []
        for item in (path, Path(str(path)+'-wal'), Path(str(path)+'-journal')):
            try:
                signatures.append(_signature(item))
            except FileNotFoundError:
                signatures.append(None)
        return (str(path), *signatures)

    def list_runs(self, *, method, as_of, page=1, page_size=20):
        self._selection(method, as_of)
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError('invalid method run pagination')
        rows = []
        if self.result_db.exists():
            with closing(self._connect()) as con:
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if {'czsc_runs', 'scan_views'} <= tables:
                    # Index-only discovery; the selected report is authenticated on read.
                    query = """SELECT r.run_id,r.created_at,r.summary,s.report_sha256,s.row_count
                        FROM czsc_runs r JOIN scan_views s ON s.run_id=r.run_id WHERE r.kind='scan'
                        AND json_extract(r.summary,'$.analysis_method')=?
                        AND json_extract(r.summary,'$.method_context.end')=? ORDER BY r.created_at DESC,r.run_id DESC"""
                    records = con.execute(query, (method, as_of)).fetchall()
                    for row in records:
                        summary = _json(row['summary']); context = summary.get('method_context') or {}
                        rows.append({'run_id': row['run_id'], 'method': method, 'as_of': as_of,
                            'created_at': row['created_at'], 'status': summary.get('status'),
                            'coverage': summary.get('coverage'), 'context': context,
                            'report_sha256': row['report_sha256'], 'row_count': row['row_count'],
                            'verification': 'indexed_candidate_requires_selected_report_verification'})
        return {'schema': SCHEMA, 'selection': {'method': method, 'as_of': as_of, 'status': 'available' if rows else 'pending'},
            'runs': rows[(page-1)*page_size:page*page_size], 'pagination': {'page': page, 'page_size': page_size, 'total': len(rows)}}

    @staticmethod
    def _selection(method, as_of):
        if method not in COMBINED:
            raise ValueError('unsupported combined method')
        _day(as_of)

    def _read(self, run_id, method, as_of):
        self._selection(method, as_of)
        if not re.fullmatch(r'[\w.-]+', run_id) or run_id in {'.', '..'}:
            raise ValueError('invalid saved scan identity')
        if not self.result_db.exists():
            raise KeyError(run_id)
        with closing(self._connect()) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'czsc_runs', 'scan_views'} <= tables:
                raise KeyError(run_id)
            row = con.execute('''SELECT r.report_path,s.report_sha256,s.row_count FROM czsc_runs r
                JOIN scan_views s ON s.run_id=r.run_id WHERE r.run_id=? AND r.kind='scan' ''', (run_id,)).fetchone()
        if not row:
            raise KeyError(run_id)
        relative = Path(row['report_path'])
        path = self.runs_root / relative
        if relative.is_absolute() or '..' in relative.parts or path.resolve() != path or not path.is_relative_to(self.runs_root):
            raise ValueError('saved scan path is outside owned storage or a symlink')
        if not re.fullmatch(r'[a-f0-9]{64}', row['report_sha256']):
            raise ValueError('saved scan index SHA invalid')
        signature = _signature(path)
        key = (run_id, row['report_sha256'], row['row_count'], signature)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                result = cached[0]
                if result['analysis_method'] != method or result['as_of'] != as_of:
                    raise ValueError('selected method/date differs from saved scan')
                self._cache.move_to_end(key)
                return result
            if signature[2] > 512*1024*1024:
                raise ValueError('saved scan exceeds read limit')
            fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            with os.fdopen(fd, 'rb') as stream:
                before = os.fstat(stream.fileno())
                encoded = stream.read(512*1024*1024+1)
                after = os.fstat(stream.fileno())
            if (before.st_size != len(encoded) or
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) or
                _signature(path) != signature or path.resolve() != path):
                raise ValueError('saved scan changed during read')
            if hashlib.sha256(encoded).hexdigest() != row['report_sha256']:
                raise ValueError('saved scan report SHA mismatch')
            report = _json(encoded)
            result = self._validate(report, run_id, method, as_of, row['row_count'], row['report_sha256'])
            self._remember(key, result)
            return result

    @staticmethod
    def _validate(report, run_id, method, as_of, count, sha):
        if not isinstance(report, dict):
            raise ValueError('saved scan is not an object')
        context = report.get('method_context'); frozen = report.get('frozen_identity') or {}
        population = report.get('population'); rows = report.get('rows'); coverage = report.get('coverage') or {}
        if not isinstance(frozen, dict) or not isinstance(coverage, dict):
            raise ValueError('saved scan identity/coverage malformed')
        if (report.get('run_id') != run_id or report.get('analysis_method') != method or
            report.get('as_of') != as_of or report.get('data_end') != as_of or
            report.get('strategy_version') != 'account_free_methods_v1' or not isinstance(context, dict) or
            context.get('analysis_method') != method or context.get('end') != as_of or frozen.get('context') != context):
            raise ValueError('saved scan method/date/context identity mismatch')
        if (not isinstance(population, list) or not population or len(population) != len(set(population)) or
            not all(is_calculation_symbol(s) and re.fullmatch(r'\d{6}\.(SH|SZ)', str(s)) for s in population) or
            population != frozen.get('population') or not isinstance(rows, list) or
            not all(isinstance(row, dict) for row in rows) or
            [row.get('symbol') for row in rows] != population or len(rows) != count):
            raise ValueError('saved scan frozen population mismatch')
        if any(not isinstance(row.get('method_evaluation'), dict) for row in rows):
            raise ValueError('saved scan method evaluation malformed')
        source_counts = {status: sum((r.get('method_evaluation') or {}).get('status') == status for r in rows)
                         for status in ('complete', 'partial', 'failed', 'deferred')}
        if sum(source_counts.values()) != len(rows):
            raise ValueError('unknown saved method row status')
        if (any(type(coverage.get(key)) is not int for key in ('requested', 'failed', 'deferred', 'unavailable', 'evaluated')) or
            coverage.get('requested') != len(rows) or coverage.get('deferred') != source_counts['deferred'] or
            coverage.get('failed') != source_counts['failed'] or coverage.get('unavailable') != source_counts['partial'] or
            coverage.get('evaluated') != len(rows)-source_counts['deferred'] or
            coverage.get('complete') is not (source_counts['deferred'] == 0)):
            raise ValueError('saved scan coverage identity mismatch')
        success = len(rows)-source_counts['failed']-source_counts['deferred']
        if any(key in coverage and (type(coverage[key]) is not int or coverage[key] != success)
               for key in ('success', 'successful')):
            raise ValueError('saved scan successful coverage mismatch')
        views = []
        for row in rows:
            evaluation = row.get('method_evaluation') or {}
            if (row.get('analysis_method') != method or evaluation.get('method') != method or evaluation.get('as_of') != as_of or
                any(row.get(key) is not None and row[key] != as_of for key in ('date', 'data_end', 'as_of')) or
                row.get('method_context') is not None and row['method_context'] != context):
                raise ValueError('saved scan row method/date/context identity mismatch')
            view = classify_method_row(row, context=context, as_of=as_of, model_version=frozen.get('model_version'))
            if 'forecast_horizon_identity_mismatch' in view['reason_codes']:
                raise ValueError('saved scan forecast horizon identity mismatch')
            score = evaluation.get('score') or {}
            scored = method != 'structure_kronos'
            expected = {'model_id': context.get('score_model_id'),
                        'feature_profile': 'ma_trend_v1' if scored else None}
            if any(evaluation.get(key) is not None and evaluation[key] != value for key, value in expected.items()):
                raise ValueError('saved scan row model identity mismatch')
            if not isinstance(score, dict):
                raise ValueError('saved scan score malformed')
            head = 'l2' if method.endswith('_l2') else 'mlp'
            expected_score = {'model_version': frozen.get('model_version'), 'variant': 'A',
                'model_type': 'l2' if head == 'l2' else 'torch_mlp',
                'feature_profile': 'ma_trend_v1', 'consumes_kronos': False}
            if frozen.get('model_version') is not None:
                expected_score['scorer_id'] = hashlib.sha256(json.dumps({'model': frozen['model_version'], 'head': head}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if (not scored and score or any(score.get(key) is not None and score[key] != value
                    for key, value in expected_score.items())):
                raise ValueError('saved scan row scorer identity mismatch')
            qualification = score.get('qualification') or {}
            if not isinstance(qualification, dict):
                raise ValueError('saved scan qualification malformed')
            expected_mode = 'historical_candidate' if context.get('usage_mode', 'historical') == 'historical' else 'paper'
            if qualification.get('mode') is not None and qualification['mode'] != expected_mode:
                raise ValueError('saved scan row qualification identity mismatch')
            forecast = evaluation.get('kronos') or {}
            if not isinstance(forecast, dict):
                raise ValueError('saved scan forecast malformed')
            if forecast.get('as_of') is not None and forecast['as_of'] != as_of:
                raise ValueError('saved scan forecast cutoff identity mismatch')
            quality = forecast.get('quality') or {}
            if not isinstance(quality, dict):
                raise ValueError('saved scan forecast quality malformed')
            if quality.get('requested_samples') is not None and quality['requested_samples'] != context.get('samples'):
                raise ValueError('saved scan forecast sample identity mismatch')
            views.append({**_scan_row_view(row), 'method_view': view})
        identity = {'run_id': run_id, 'report_sha256': sha, 'method': method, 'as_of': as_of,
            'context_sha256': stable_hash(context), 'population_sha256': stable_hash(population),
            'frozen_identity_sha256': stable_hash(frozen), 'model_version': frozen.get('model_version'),
            'schema': SCHEMA, 'policy': deepcopy(POLICY),
            'display_code': {'aggregation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'classifier_sha256': hashlib.sha256(Path(__file__).with_name('method_display.py').read_bytes()).hexdigest()}}
        return {key: deepcopy(report.get(key)) for key in ('run_id', 'status', 'strategy_version', 'analysis_mode', 'analysis_method',
            'method_context', 'data_end', 'as_of', 'data_range', 'coverage', 'market_history', 'updated_at')} | {
                'rows': views, 'identity': identity, 'population_summary': summarize(views)}

    def stocks(self, run_id, *, method, as_of, page=1, page_size=40, query='', category='all', symbols=None):
        report = self._read(run_id, method, as_of)
        rows, pagination = _page(report['rows'], page, page_size, query, category, symbols)
        return {key: deepcopy(value) for key, value in report.items() if key != 'rows'} | {
            'rows': rows, 'pagination': pagination, 'scan_view': {'schema': 'current_scan_view', 'compact': True,
                'analysis_mode': 'reference', 'raw_evidence': 'per_stock_analysis',
                'report_sha256': report['identity']['report_sha256'], **pagination}}

    def _boards(self, report):
        cutoff = report['as_of']+'T15:00:00+08:00'
        snapshots = BoardKnowledgeStore(self.kb_path, readonly=True).catalog(as_of=cutoff)
        lookup = {row['symbol']: row for row in report['rows']}
        boards, members = [], {}
        for snapshot in snapshots:
            if any(_instant(snapshot[key]) > _instant(cutoff) for key in ('available_at', 'effective_at', 'observed_at')):
                raise ValueError('board membership was not visible at scan cutoff')
            projection = calculation_membership(snapshot)
            selected = [lookup.get(symbol) or {'symbol': symbol, 'name': '', 'method_view': {
                'status': 'not_in_scan', 'valid': False, 'direction': None, 'reason_codes': ['not_in_frozen_scan_population'],
                'forecast_valid': False, 'score_valid': False, 'method': report['analysis_method'], 'as_of': report['as_of']}}
                for symbol in projection['eligible_members']]
            stats = summarize(selected)
            enough = (projection['usable_membership'] and stats['counts']['complete'] >= POLICY['minimum_members'] and
                (stats['coverage']['method_complete_ratio'] or 0) >= POLICY['minimum_complete_fraction'])
            board = {key: snapshot.get(key) for key in ('board_id', 'name', 'kind', 'source', 'taxonomy', 'observed_at', 'available_at', 'effective_at')}
            board.update(stats, ranking_eligible=enough, availability='available' if enough else 'insufficient_coverage',
                ranking_reason=None if enough else 'minimum_5_complete_members_and_80_percent_complete_coverage',
                membership={'mode': 'current_reference', 'historical_membership_verified': False, 'training_eligible': False,
                    'cutoff_at': cutoff, 'calculation_scope': CALCULATION_SCOPE, 'snapshot_hash': snapshot['snapshot_hash'],
                    'projection_hash': projection['projection_hash'], 'source_member_count': projection['source_member_count'],
                    'eligible_count': projection['eligible_count'], 'excluded_count': len(projection['excluded_members']),
                    'usable_membership': projection['usable_membership']})
            boards.append(board); members[board['board_id']] = selected
        return boards, members

    def _board_projection(self, report):
        """Reuse only a fully authenticated projection of unchanged DB files.

        Keep symbols rather than repeated stock-result dictionaries in the
        projection. Member pagination joins back to the authenticated report.
        The existing lock also combines concurrent misses, avoiding 710 source
        proof checks for every simultaneous summary/member request.
        """
        with self._lock:
            for _ in range(2):
                signature = self._knowledge_signature()
                for stale in list(self._cache):
                    if len(stale) == 3 and stale[0] == 'boards' and stale[2] != signature:
                        _, removed = self._cache.pop(stale)
                        self._bytes -= removed
                key = ('boards', stable_hash(report['identity']), signature)
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    return cached[0]
                boards, members = self._boards(report)
                projection = {'boards': boards, 'members': {
                    board: [row['symbol'] for row in rows] for board, rows in members.items()}}
                if signature != self._knowledge_signature():
                    # Never publish a projection assembled while a manifest or
                    # source changed. Retry once; a busy writer stays explicit.
                    continue
                self._remember(key, projection)
                return projection
        raise ValueError('board knowledge changed during saved summary read; retry')

    def summary(self, run_id, *, method, as_of, board_id=None, page=1, page_size=40, query='', category='all', symbols=None):
        report = self._read(run_id, method, as_of)
        projection = self._board_projection(report)
        boards = projection['boards']
        identity = {**deepcopy(report['identity']), 'calculation_scope': CALCULATION_SCOPE,
            'memberships': {board['board_id']: {'snapshot_hash': board['membership']['snapshot_hash'],
                'projection_hash': board['membership']['projection_hash']} for board in boards}}
        identity['summary_sha256'] = stable_hash(identity)
        result = {'schema': SCHEMA, 'selection': {'run_id': run_id, 'method': method, 'as_of': as_of, 'status': 'available'},
            'identity': identity, 'method_context': deepcopy(report['method_context']),
            'saved_report_updated_at': report.get('updated_at'),
            'timing_semantics': 'saved_cutoff_reference_not_actual_first_availability',
            'population_summary': deepcopy(report['population_summary']), 'coverage': deepcopy(report['coverage']),
            'membership_semantics': {'mode': 'current_reference', 'historical_membership_verified': False,
                'training_eligible': False, 'cutoff_at': as_of+'T15:00:00+08:00',
                'availability': 'available' if boards else 'pending', 'overlap_is_non_additive': True},
            'ranking_policy': deepcopy(POLICY), 'limitations': list(LIMITATIONS)}
        if board_id is not None:
            board = next((b for b in boards if b['board_id'] == board_id), None)
            if board is None:
                raise KeyError(board_id)
            lookup = {row['symbol']: row for row in report['rows']}
            selected = [lookup.get(symbol) or {'symbol': symbol, 'name': '', 'method_view': {
                'status': 'not_in_scan', 'valid': False, 'direction': None, 'reason_codes': ['not_in_frozen_scan_population'],
                'forecast_valid': False, 'score_valid': False, 'method': report['analysis_method'], 'as_of': report['as_of']}}
                for symbol in projection['members'][board_id]]
            rows, pagination = _page(selected, page, page_size, query, category, symbols)
            return result | {'board': deepcopy(board), 'rows': rows, 'pagination': pagination}
        return result | {'boards': deepcopy(boards)}
