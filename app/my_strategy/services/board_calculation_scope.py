"""Immutable SH/SZ A-share calculation projections of source board snapshots.

Source snapshots/proofs retain their original membership and hashes.  This
projection is a separate, versioned calculation identity, never a KB snapshot.
"""
from __future__ import annotations

import copy
import re

from my_strategy.adapters.czsc_adapter import normalize_symbol
from my_strategy.core.run_context import stable_hash

CALCULATION_SCOPE = 'sh_sz_a_v1'
PROJECTION_SCHEMA = 'board_calculation_membership_v1'


def _symbol(value):
    text = str(value).strip().upper()
    if re.fullmatch(r'\d{6}', text):
        return normalize_symbol(text)
    if not re.fullmatch(r'\d{6}\.[A-Z]{2,6}', text):
        raise ValueError('invalid source board member symbol')
    return text


def is_calculation_symbol(symbol):
    try:
        return re.fullmatch(r'(?:6\d{5}\.SH|[03]\d{5}\.SZ)', _symbol(symbol)) is not None
    except (ValueError, TypeError):
        return False


def _excluded_reason(symbol):
    if symbol.endswith('.BJ'):
        return 'beijing_excluded'
    if symbol.endswith('.NQ'):
        return 'neeq_excluded'
    return 'outside_sh_sz_a_scope'


def calculation_membership(snapshot):
    """Project a source snapshot without altering its membership or evidence.

    This pure membership transform is not source authentication. Production
    callers obtain snapshots from BoardKnowledgeStore.catalog, whose verified
    reader checks the original response/proof, indexed members and all SHAs.
    The saved projection records that verified source hash separately.
    """
    if not isinstance(snapshot, dict) or not snapshot.get('board_id') or not snapshot.get('snapshot_hash'):
        raise ValueError('board source snapshot identity incomplete')
    source = snapshot.get('members')
    if not isinstance(source, (list, tuple)):
        raise ValueError('board source members missing')
    members = sorted(_symbol(value) for value in source)
    if len(members) != len(set(members)):
        raise ValueError('duplicate board members')
    expected = snapshot.get('expected_count')
    if type(expected) is not int or expected < 0:
        raise ValueError('board expected member count is unknown')
    eligible = [symbol for symbol in members if is_calculation_symbol(symbol)]
    excluded = [symbol for symbol in members if not is_calculation_symbol(symbol)]
    complete = snapshot.get('complete') is True
    result = dict(schema=PROJECTION_SCHEMA, calculation_scope=CALCULATION_SCOPE,
        board_id=snapshot['board_id'], source_snapshot_hash=snapshot['snapshot_hash'],
        source_member_count=len(members), source_expected_count=expected, source_complete=complete,
        eligible_members=eligible, eligible_count=len(eligible), excluded_members=excluded,
        exclusion_reasons={symbol: _excluded_reason(symbol) for symbol in excluded},
        usable_membership=complete and expected == len(members) and bool(eligible))
    result['projection_hash'] = stable_hash(result)
    return result


def verify_calculation_membership(projection, snapshot=None):
    """Validate a saved projection, optionally binding it to its source proof."""
    if not isinstance(projection, dict):
        raise ValueError('board calculation membership missing')
    try:
        members = projection['eligible_members'] + projection['excluded_members']
        inferred = dict(board_id=projection['board_id'], snapshot_hash=projection['source_snapshot_hash'],
            members=members, expected_count=projection['source_expected_count'],
            complete=projection['source_complete'])
        expected = calculation_membership(inferred)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('board calculation projection malformed') from exc
    if stable_hash(projection) != stable_hash(expected) or (snapshot is not None and
            stable_hash(projection) != stable_hash(calculation_membership(snapshot))):
        raise ValueError('board calculation projection identity drift')
    return copy.deepcopy(projection)


def calculation_contract(memberships):
    """Identity shared by a market bundle and its native collection."""
    if not isinstance(memberships, dict):
        raise ValueError('board calculation membership map missing')
    verified = {}
    for board_id, projection in memberships.items():
        value = verify_calculation_membership(projection)
        if value['board_id'] != board_id:
            raise ValueError('board calculation membership map identity drift')
        verified[board_id] = value
    return dict(calculation_scope=CALCULATION_SCOPE,
        projection_hashes={key: verified[key]['projection_hash'] for key in sorted(verified)},
        required_symbols=sorted({symbol for value in verified.values() for symbol in value['eligible_members']}))
