import copy
import hashlib
import json

import pytest

from my_strategy.scripts.reaggregate_trend_validation import _read_verified,verify_saved_account
from my_strategy.services.czsc_trend_pipeline import completed_round_trips
from my_strategy.tests.test_czsc_trend_account import fixture,replay


def test_reaggregation_verifies_original_paid_cash_and_rejects_changed_ledger():
    frame,calendar,prepared=fixture();prepared['rows'][65].update(action='SELL',emitted=True,event_id='S')
    account=replay(frame,calendar,prepared)
    row={**account,'completed_round_trips':completed_round_trips(account['ledger'])}
    assert verify_saved_account(row,100000.,calendar[-1])['prior_metrics_reconciled']
    changed=copy.deepcopy(row);changed['ledger'][0]['cash_flow']-=1.
    with pytest.raises(ValueError,match='cash mismatch'):
        verify_saved_account(changed,100000.,calendar[-1])
    changed=copy.deepcopy(row);changed['metrics']['fees']+=1.
    with pytest.raises(ValueError,match='paid fees differ'):
        verify_saved_account(changed,100000.,calendar[-1])


def test_reused_asset_rejects_drift_against_prior_sha(tmp_path):
    path=tmp_path/'ledger.json';path.write_text(json.dumps([{'shares':100}]))
    original=hashlib.sha256(path.read_bytes()).hexdigest()
    value,asset=_read_verified(path,original)
    assert value==[{'shares':100}] and asset['sha256']==original
    path.write_text(json.dumps([{'shares':200}]))
    with pytest.raises(ValueError,match='SHA drift'):_read_verified(path,original)
