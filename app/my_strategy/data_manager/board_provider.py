"""Current THS index taxonomy obtained through the authorized Fuyao API.

This path never uses public-site token generation or another vendor's taxonomy.
Catalog and constituent endpoints return a whole list without pagination;
independent repeated reads prove the frozen identifier sets stayed stable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlencode

import requests

from my_strategy.core.paths import PROJECT_ROOT
from my_strategy.core.tz import local_now

SOURCE = 'fuyao:ths'
TAXONOMY = 'THS_INDEX'
FORMAT = 'ths_board_snapshot'
BASE_URL = 'https://fuyao.aicubes.cn'
DOC_URL = BASE_URL + '/docs/api-reference/a-share-index/'
CATALOG_PATH = '/api/a-share-index/catalog/ths-index-list'
MEMBERS_PATH = '/api/a-share-index/constituents/ths-stock-list'
KINDS = {'industry': ('I', 'industry'), 'concept': ('C', 'cn_concept')}


class BoardSourceError(ValueError): pass
class BoardAccessError(BoardSourceError): pass
class BoardBudgetExhausted(BoardSourceError): pass


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def load_board_api_key(env_file=None):
    """Read one credential from the environment or the ignored local env file."""
    if not os.environ.get('FUYAO_API_KEY', '').strip():
        path = Path(env_file or PROJECT_ROOT / '.env.board-kb.local')
        if path.is_file():
            for line in path.read_text(encoding='utf-8').splitlines():
                name, separator, value = line.partition('=')
                if separator and name.strip() == 'FUYAO_API_KEY':
                    os.environ.setdefault('FUYAO_API_KEY', value.strip().strip('"').strip("'"))
    key = os.environ.get('FUYAO_API_KEY', '').strip()
    if not key: raise BoardAccessError('FUYAO_API_KEY is not configured')
    return key


def board_identity(kind, source_code):
    if kind not in KINDS or not isinstance(source_code, str) or not re.fullmatch(r'\d{6}\.TI', source_code):
        raise BoardSourceError('invalid THS index kind or full source code')
    return 'THS_' + KINDS[kind][0] + '_' + source_code


def _symbol(row):
    value = str(row.get('thscode') or '').strip().upper()
    if not re.fullmatch(r'\d{6}\.(?:SH|SZ|BJ)', value):
        raise BoardSourceError('unsupported or missing constituent stock identity')
    digits, exchange = value.split('.')
    if not (exchange == 'SH' and digits.startswith('6') or exchange == 'SZ' and digits.startswith(('0', '3')) or
            exchange == 'BJ' and digits.startswith(('4', '8', '9'))):
        raise BoardSourceError('non-A-share or exchange-conflicting constituent cannot be omitted from a complete board')
    ticker = row.get('ticker')
    if ticker is not None and str(ticker).strip() != value.split('.')[0]:
        raise BoardSourceError('constituent ticker and full stock identity conflict')
    return value


def _source_members(payload):
    """Keep a proof of the full source set while explicitly projecting only NQ."""
    included, excluded = [], []
    for row in payload['data']['item']:
        code = str(row.get('thscode') or '').strip().upper()
        if re.fullmatch(r'\d{6}\.NQ', code):
            ticker = row.get('ticker')
            if ticker is not None and str(ticker).strip() != code.split('.')[0]:
                raise BoardSourceError('constituent ticker and full stock identity conflict')
            excluded.append(code)
        else:
            included.append(_symbol(row))
    full = included + excluded
    if len(set(full)) != len(full): raise BoardSourceError('duplicate constituent code in full response')
    return sorted(included), sorted(excluded), sorted(full)


class FuyaoBoardProvider:
    source = SOURCE
    taxonomy = TAXONOMY
    historical_membership_verified = False

    def __init__(self, *, session=None, request_budget=2000, min_interval=1.0, timeout=20, env_file=None):
        if type(request_budget) is not int or request_budget < 1:
            raise ValueError('request budget must be a positive integer')
        if isinstance(min_interval, bool) or not isinstance(min_interval, (int, float)) or not math.isfinite(min_interval) or min_interval < 0:
            raise ValueError('request interval must be nonnegative')
        self._api_key = load_board_api_key(env_file)
        self.session = session or requests.Session()
        self.request_budget, self.request_count = request_budget, 0
        self.min_interval, self.timeout = float(min_interval), float(timeout)
        self._last_request = None
        self.catalog_provenance = {}

    def close(self): self.session.close()

    def _safe(self, value):
        if isinstance(value, str): return value.replace(self._api_key, '[REDACTED]')
        if isinstance(value, list): return [self._safe(x) for x in value]
        if isinstance(value, dict):
            return {key: '[REDACTED]' if str(key).lower() in {'api_key', 'x-api-key', 'authorization'} else self._safe(item)
                for key, item in value.items()}
        return value

    def _get(self, path, params):
        if self.request_count >= self.request_budget:
            raise BoardBudgetExhausted('Fuyao request budget exhausted; resume with a new explicit budget')
        if self._last_request is not None:
            remaining = self.min_interval - (time.monotonic() - self._last_request)
            if remaining > 0: time.sleep(remaining)
        self.request_count += 1; self._last_request = time.monotonic()
        try:
            response = self.session.get(BASE_URL + path, params=params,
                headers={'X-api-key': self._api_key, 'Accept': 'application/json', 'User-Agent': 'KHQuant/0.1.6'},
                timeout=self.timeout, allow_redirects=False)
        except requests.RequestException as exc:
            raise BoardSourceError('Fuyao transport request failed: ' + type(exc).__name__) from None
        if response.status_code in (401, 403, 429):
            raise BoardAccessError('Fuyao access refused or rate limited: HTTP ' + str(response.status_code))
        if response.status_code != 200:
            raise BoardSourceError('Fuyao HTTP response invalid: ' + str(response.status_code))
        try: payload = self._safe(response.json())
        except ValueError: raise BoardSourceError('Fuyao response is not valid JSON') from None
        if not isinstance(payload, dict): raise BoardSourceError('Fuyao response envelope missing')
        code = payload.get('code')
        if code in (2001, 2003, 4001):
            raise BoardAccessError('Fuyao authentication, permission or rate limit failure: code ' + str(code))
        if type(code) is not int or code != 0:
            raise BoardSourceError('Fuyao business response failed: code ' + str(code))
        data = payload.get('data')
        if (not isinstance(data, dict) or not isinstance(data.get('item'), list) or
                not payload.get('request_id') or type(data.get('timestamp')) is not int or data['timestamp'] <= 0):
            raise BoardSourceError('Fuyao complete response items or envelope provenance missing')
        rows = data['item']
        if any(not isinstance(row, dict) for row in rows): raise BoardSourceError('Fuyao response row schema invalid')
        if 'total' in data and (type(data['total']) is not int or data['total'] != len(rows)):
            raise BoardSourceError('Fuyao declared total does not match the full response')
        if any(key in data for key in ('next', 'next_page', 'has_more', 'next_cursor')):
            if any(data.get(key) not in (None, False, '') for key in ('next', 'next_page', 'has_more', 'next_cursor')):
                raise BoardSourceError('Fuyao response is paginated or truncated contrary to the documented contract')
        return payload

    @staticmethod
    def _catalog_rows(payload, kind):
        boards = []
        for item in payload['data']['item']:
            source_code, name = str(item.get('thscode') or '').strip().upper(), str(item.get('name') or '').strip()
            if not name: raise BoardSourceError('THS catalog name missing')
            boards.append({'board_id': board_identity(kind, source_code), 'source_code': source_code,
                'name': name, 'kind': kind, 'source': SOURCE, 'taxonomy': TAXONOMY})
        if not boards: raise BoardSourceError('Fuyao complete THS directory is empty')
        by_id = {}
        for board in boards:
            identity = board['board_id']
            if identity in by_id and by_id[identity] != board:
                raise BoardSourceError('conflicting THS catalog identity or name')
            if identity in by_id: raise BoardSourceError('duplicate THS catalog identity')
            by_id[identity] = board
        return sorted(boards, key=lambda board: board['board_id'])

    def catalog(self, kinds=('industry', 'concept')):
        if not kinds or len(set(kinds)) != len(kinds) or any(kind not in KINDS for kind in kinds):
            raise ValueError('board kinds must be unique supported classifications')
        result = []; proof = {}
        for kind in kinds:
            params = {'tag': KINDS[kind][1]}
            first, second = self._get(CATALOG_PATH, params), self._get(CATALOG_PATH, params)
            boards, repeated = self._catalog_rows(first, kind), self._catalog_rows(second, kind)
            if boards != repeated: raise BoardSourceError('THS catalog changed during frozen directory collection')
            proof[kind] = {'full_response_contract': True, 'catalog_set_stable': True, 'board_count': len(boards),
                'catalog_set_hash': _hash(boards), 'first_response_sha256': _hash(first), 'repeated_response_sha256': _hash(second),
                'request_id': first['request_id'], 'source_timestamp': first['data']['timestamp'],
                'business_code': 0, 'documentation_url': DOC_URL, 'request_path': CATALOG_PATH, 'request_params': params,
                'source_response': first, 'repeated_source_response': second}
            result.extend({**board, 'catalog_provenance': proof[kind]} for board in boards)
        self.catalog_provenance = proof
        return sorted(result, key=lambda board: board['board_id'])

    def snapshot(self, board):
        if (not isinstance(board, dict) or board.get('board_id') != board_identity(board.get('kind'), board.get('source_code')) or
                board.get('source') != SOURCE or board.get('taxonomy') != TAXONOMY or not board.get('name')):
            raise BoardSourceError('board is not in the identified Fuyao THS index catalog')
        params = {'thscode': board['source_code']}
        first, second = self._get(MEMBERS_PATH, params), self._get(MEMBERS_PATH, params)
        initial, excluded, full = _source_members(first)
        repeated, repeated_excluded, repeated_full = _source_members(second)
        if initial != repeated or excluded != repeated_excluded or full != repeated_full:
            raise BoardSourceError('THS constituent set changed during frozen collection')
        observed = local_now().isoformat()
        proof = {'pagination_complete': True, 'full_response_contract': True, 'membership_set_stable': True,
            'unique_members': len(initial), 'membership_set_hash': _hash(initial),
            'repeated_membership_set_hash': _hash(repeated), 'first_member': initial[0] if initial else None, 'last_member': initial[-1] if initial else None,
            'first_response_sha256': _hash(first), 'repeated_response_sha256': _hash(second),
            'business_code': 0, 'request_path': MEMBERS_PATH, 'request_params': params,
            'request_id': first['request_id'], 'repeated_request_id': second['request_id'],
            'source_timestamp': first['data']['timestamp'], 'repeated_source_timestamp': second['data']['timestamp'],
            'documentation_url': DOC_URL, 'current_only': True, 'historical_membership_verified': False,
            'source_response': first, 'repeated_source_response': second,
            'catalog_provenance': board.get('catalog_provenance'),
            'member_list_state': 'populated_verified' if initial else 'empty_verified'}
        if 'total' in first['data']: proof['reported_count'] = first['data']['total']
        projection = {}
        if excluded:
            projection = {'membership_scope': 'a_share_projection_v1', 'source_member_count': len(full),
                'excluded_source_members': excluded}
            proof.update(full_source_membership_set_hash=_hash(full),
                repeated_full_source_membership_set_hash=_hash(repeated_full), full_source_membership_set_stable=True)
        return {**{key: board[key] for key in ('board_id', 'source_code', 'name', 'kind', 'source', 'taxonomy')}, 'format': FORMAT, 'observed_at': observed, 'available_at': observed,
            'effective_at': observed, 'complete': True, 'expected_count': len(initial), 'members': initial,
            'source_url': BASE_URL + MEMBERS_PATH + '?' + urlencode(params), 'provenance': proof,
            'payload_sha256': _hash(first), 'historical_membership_verified': False, 'current_only': True, **projection}
