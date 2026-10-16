"""Publication metadata reads omit bulk columns without skipping authentication."""
import json
import sqlite3

import pytest

from my_strategy.storage import board_fluent as storage
from my_strategy.storage.board_fluent import BoardFluentStore, canonical, digest
from my_strategy.tests.test_board_features import market
from my_strategy.tests.test_board_fluent import bundle, native_result


class OriginalProjection:
    """Run identical validation against the pre-change SELECT * result shape."""
    def __init__(self, connection):
        self.connection = connection

    def execute(self, sql, parameters):
        assert 'FROM feature_generations WHERE generation_id=?' in sql
        return self.connection.execute('SELECT * FROM feature_generations WHERE generation_id=?', parameters)


@pytest.fixture
def full_generation(market, tmp_path):
    value = bundle(market)
    store = BoardFluentStore(tmp_path/'fluent.db')
    generation = store.save_market(value)
    native, boards = native_result(market, value['boards'])
    store.attach_native(generation, native, boards)
    return store, generation, value


def test_full_market_and_native_metadata_equal_original_projection(full_generation):
    store, generation, _ = full_generation
    with store._read() as connection:
        before = store._generation(OriginalProjection(connection), generation)
        after = store._generation(connection, generation)
        assert before == after
        assert after['market_ready'] is after['full_ready'] is True
        assert after['native_identity'] is not None
        assert store._generation(connection, 'missing') is None
        assert store._generation(OriginalProjection(connection), 'missing') is None


def test_metadata_read_does_not_access_bulk_columns_but_complete_load_still_does(full_generation):
    store, generation, value = full_generation
    expected = store.get_generation(generation)
    with store._read() as connection:
        def prohibit_bulk(action, table, column, *_):
            if action == sqlite3.SQLITE_READ and table == 'feature_generations' and column in {'top_json', 'top_sha', 'bundle_sha'}:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        connection.set_authorizer(prohibit_bulk)
        assert store._generation(connection, generation) == expected
        with pytest.raises(sqlite3.DatabaseError, match='prohibited'):
            store._generation(OriginalProjection(connection), generation)
    assert store.load_market(value['cache_key']) == value
    with sqlite3.connect(store.path) as connection:
        connection.execute('DROP TRIGGER immutable_generation_inputs')
        connection.execute('UPDATE feature_generations SET top_json=? WHERE generation_id=?', ('{}', generation))
    # This matches the original metadata path: bundle authentication belongs to
    # load_market/verify_generation, which must continue rejecting that damage.
    assert store.get_generation(generation) == expected
    with pytest.raises(ValueError, match='SHA drift'):
        store.load_market(value['cache_key'])
    with pytest.raises(ValueError, match='SHA drift'):
        store.verify_generation(generation, level='full')


@pytest.mark.parametrize('corruption', ['public_json', 'public_sha', 'identity_json', 'identity_sha',
    'cache_key', 'contract_hash', 'price_date', 'publication_inputs', 'native_identity_json', 'native_identity_sha'])
def test_every_original_generation_identity_failure_is_preserved(full_generation, corruption):
    store, generation, _ = full_generation
    with sqlite3.connect(store.path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute('SELECT * FROM feature_generations WHERE generation_id=?', (generation,)).fetchone()
        connection.execute('DROP TRIGGER immutable_generation_inputs')
        if corruption == 'publication_inputs':
            public = json.loads(row['public_json'])
            public['publication_inputs'] = {'unexpected_source': 'self-consistently-resigned'}
            connection.execute('UPDATE feature_generations SET public_json=?,public_sha=? WHERE generation_id=?',
                (canonical(public), digest(public), generation))
        else:
            replacement = ('{}' if corruption.endswith('_json') else
                '1900-01-01' if corruption == 'price_date' else '0'*64)
            connection.execute('UPDATE feature_generations SET '+corruption+'=? WHERE generation_id=?', (replacement, generation))
    with store._read() as connection:
        with pytest.raises(ValueError) as original:
            store._generation(OriginalProjection(connection), generation)
        with pytest.raises(ValueError) as projected:
            store._generation(connection, generation)
    assert str(projected.value) == str(original.value)
