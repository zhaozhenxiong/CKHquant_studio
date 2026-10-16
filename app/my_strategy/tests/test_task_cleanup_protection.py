from copy import deepcopy
from my_strategy.scripts.verify_task_cleanup_protection import compare


def proof():
    return {'raw': {'database': {'sha256': 'raw'}, 'coverage': {'rows': 100}},
            'watchlist': {'count': 2, 'rows_sha256': 'watch'},
            'accounts': {'count': 2, 'state_sha256': 'account'},
            'research_objects': {'saved': {'sha256': 'ledger'}},
            'pointers': {'current_score': 'selected'}, 'pins': {'owner': True},
            'model_files': {'kronos': {'sha256': 'frozen'}}, 'errors': []}


def test_new_scan_and_prediction_objects_do_not_hide_preserved_baseline():
    before = proof(); after = deepcopy(before)
    after['research_objects']['new_integration_result'] = {'sha256': 'new'}
    assert compare(before, after)['passed']
    after['research_objects'].pop('saved')
    result = compare(before, after)
    assert not result['passed'] and result['differences'][0]['identity'] == 'saved'


def test_same_account_count_does_not_hide_modified_pointer_or_raw_quote_bytes():
    before = proof(); after = deepcopy(before)
    after['raw']['database']['sha256'] = 'changed'
    after['accounts']['state_sha256'] = 'different_account'
    after['pointers']['current_score'] = 'replacement'
    result = compare(before, after)
    assert {item['section'] for item in result['differences']} == {'raw', 'accounts', 'pointers'}
