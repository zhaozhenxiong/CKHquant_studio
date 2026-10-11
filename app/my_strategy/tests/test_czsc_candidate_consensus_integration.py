"""Default v3 and replayed v1/v2 requests bind separate policies end to end."""
import pytest

from my_strategy.tests.test_czsc_weighted_consensus_integration import api_client, SYMBOL


V3 = "czsc_agent_consensus_v3"


def test_new_analysis_saves_v3_and_replay_keeps_its_full_config(monkeypatch):
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


@pytest.mark.parametrize('version', ['czsc_agent_consensus_v1', 'czsc_agent_consensus_v2'])
def test_saved_older_versions_keep_their_original_contract(monkeypatch, version):
    client, _ = api_client(monkeypatch)
    with client:
        result = client.post('/api/analysis', json={'symbol': SYMBOL, 'research': False,
                                                  'consensus_version': version})
        assert result.status_code == 200, result.text
        body = result.json()
        assert body['consensus_decisions']['version'] == version
        assert 'evidence_valid_bars' not in body['request']['consensus_config']


@pytest.mark.parametrize('age', [0, 2, 4, 6, True, 1.5])
def test_api_rejects_unsupported_evidence_window_without_running_analysis(monkeypatch, age):
    client, forwarded = api_client(monkeypatch)
    with client:
        response = client.post('/api/analysis', json={'symbol': SYMBOL, 'research': False,
                                    'consensus_version': V3, 'consensus_config': {'evidence_valid_bars': age}})
        assert response.status_code == 422
        assert not forwarded
