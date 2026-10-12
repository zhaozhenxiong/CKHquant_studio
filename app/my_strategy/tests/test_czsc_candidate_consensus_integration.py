"""Native v4 defaults and legacy family v3 defaults keep distinct input policies."""
import pytest

from my_strategy.tests.test_czsc_weighted_consensus_integration import api_client, SYMBOL


V3 = "czsc_agent_consensus_v3"
V4 = "czsc_agent_consensus_v4"


def test_new_structure_baseline_saves_v3_and_replay_keeps_its_full_config(monkeypatch):
    client, forwarded = api_client(monkeypatch)
    with client:
        response = client.post('/api/analysis', json={'symbol': SYMBOL, 'research': False})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data['request']['consensus_version'] == V3
        assert data['consensus_decisions']['version'] == V3
        assert data['request']['consensus_config']['evidence_valid_bars'] == 3
        assert data['request']['consensus_config'] == data['consensus_decisions']['config']
        again = client.post('/api/analysis', json={**data['request'], 'as_of': '2026-07-02'})
        assert again.status_code == 200, again.text
        assert again.json()['consensus_decisions']['config_hash'] == data['consensus_decisions']['config_hash']
        assert 'consensus_config' not in forwarded[0] and 'consensus_version' not in forwarded[0]


@pytest.mark.parametrize('version', ['czsc_agent_consensus_v1', 'czsc_agent_consensus_v2', V3])
def test_saved_older_versions_keep_their_original_contract(monkeypatch, version):
    client, _ = api_client(monkeypatch)
    with client:
        result = client.post('/api/analysis', json={'symbol': SYMBOL, 'research': False,
                                                  'consensus_version': version})
        assert result.status_code == 200, result.text
        body = result.json()
        assert body['consensus_decisions']['version'] == version
        if version == V3:
            assert body['request']['consensus_config']['evidence_valid_bars'] == 3
        else:
            assert 'evidence_valid_bars' not in body['request']['consensus_config']


@pytest.mark.parametrize('age', [0, 2, 4, 6, True, 1.5])
def test_api_rejects_unsupported_evidence_window_without_running_analysis(monkeypatch, age):
    client, forwarded = api_client(monkeypatch)
    with client:
        response = client.post('/api/analysis', json={'symbol': SYMBOL, 'research': False,
                                    'consensus_version': V3, 'consensus_config': {'evidence_valid_bars': age}})
        assert response.status_code == 422
        assert not forwarded


@pytest.mark.parametrize('family', ['ma_trend', 'dual', 'wyckoff'])
def test_non_native_family_defaults_to_frozen_v3_even_with_global_v4(monkeypatch, family):
    client, forwarded = api_client(monkeypatch)
    with client:
        response = client.post('/api/analysis', json={'symbol': SYMBOL, 'research': True, 'model_family': family})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data['request']['consensus_version'] == V3
        assert data['consensus_decisions']['version'] == V3
        assert data['request']['consensus_config'] == data['consensus_decisions']['config']
        assert forwarded and forwarded[0]['model_family'] == family


@pytest.mark.parametrize('spec', [{'research': False}, {'research': True, 'model_family': 'ma_trend'},
    {'research': True, 'model_family': 'dual'}, {'research': True, 'model_family': 'wyckoff'}])
def test_explicit_native_v4_rejects_non_native_feature_envelope_before_calculation(monkeypatch, spec):
    client, forwarded = api_client(monkeypatch)
    with client:
        response = client.post('/api/analysis', json={'symbol': SYMBOL, **spec, 'consensus_version': V4})
        assert response.status_code == 422
        assert not forwarded


def test_native_high_return_new_request_saves_actual_default_v4_and_full_config(monkeypatch):
    import copy
    from my_strategy.web_dashboard import high_return
    from my_strategy.core.run_context import stable_hash
    from my_strategy.tests.test_czsc_consensus_integration import fixture
    client, baseline_forwarded = api_client(monkeypatch)
    calls = []
    def native(**kwargs):
        calls.append(kwargs)
        result, _, _ = fixture()
        result = copy.deepcopy(result)
        result.update(data_version='native-fixture', model_family='high_return', chart_evidence={'rows': []},
            consensus_decisions={'version': kwargs['consensus_version'], 'config': kwargs['consensus_config'],
                'config_hash': stable_hash(kwargs['consensus_config'])})
        return result
    monkeypatch.setattr(high_return, 'analyze_high_return', native)
    with client:
        response = client.post('/api/analysis', json={'symbol': SYMBOL, 'research': True, 'model_family': 'high_return'})
        assert response.status_code == 200, response.text
        data = response.json()
        assert calls[0]['consensus_version'] == V4 and calls[0]['consensus_config']['version'] == V4
        assert data['request']['consensus_version'] == V4
        assert data['request']['consensus_config'] == calls[0]['consensus_config']
        assert not baseline_forwarded
        again = client.post('/api/analysis', json=data['request'])
        assert again.status_code == 200, again.text
        assert again.json()['consensus_decisions']['config_hash'] == data['consensus_decisions']['config_hash']
