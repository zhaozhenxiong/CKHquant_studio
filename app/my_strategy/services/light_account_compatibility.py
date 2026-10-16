"""One audited execution-only transition; no wildcard code compatibility.

The exact old/new dictionaries below are sealed after equivalence acceptance.
All model, policy, Broker, scorer and source dependencies remain bound to their
original hashes. A later edit of any included file requires a new review.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

LEGACY_CODE = {'agent_evidence.py': 'd575a8dee1db473b3003bba6f02b819a5edf82a88bb59ac34cf384b947c4a7e6',
 'czsc_market_features.py': '7a8322d6d6915f990449143d0c92fd3e152fcb967026b762a81ffb975481f837',
 'czsc_native_evidence.py': 'a4dc84348de0a10f668d53df0fdb4232f42454c7e7c15d5d6a4bfde98ddd0ebf',
 'execution/broker.py': '241ea5ea95d8ab7ce9b5b564ddd24bfcc9c8eac0c17619bf080e43d062b55344',
 'execution/cost_model.py': '7ec3cadf4ab198af77135aa6084e5524895a9163d7a465f377e7d2e37d750386',
 'execution/ledger.py': '970ba7a07e586535c99fbc9c2ecf28d1b898c369f98e8808fbb1e00f5d9e30c3',
 'execution/order.py': 'd515ed66e31a4b35fee1cbfc08da76104f61971296d9c8c636cbc8c38674225e',
 'kronos_prediction.py': '9558ffa1287fe5035792af7a5a39def0b61f291dc61cda9edcc0b97dffcf3d02',
 'kronos_tuning.py': 'ef103d9531d52d97ed9c50fca4b128302a4f29b51815c12116282d06c2a2da1f',
 'light_account_continuation.py': '67fce399eb1caa59f7a26d3d8fec6e8fdf40321134214557d1ec1c0c4e15e777',
 'light_daily_scheduler.py': '96c5ad17993fab6ce00a294ed8f2ef32c12b73d9f1d451928ab681acb5f4a539',
 'light_experiment_registry.py': '2862971e1d3c1bc362d6b99e2e4af98ce143d5d75dc6b327fadf7475f39ab5ea',
 'light_ma_features.py': '45809c37036922f39eb3879f4d96f6a4ef5970b087363f9b0a05974de30bdbb3',
 'light_qualification.py': '1cec8361853385e97e1a026cffbc86b286a5cc8314a02d629eef7226932957a7',
 'light_research.py': '9e3a8ea42936c44b95b2b0c64ecc1a344264308652a4df847cf1ee9569fc2cac',
 'light_score_training.py': 'd7b77962671e2d5784c8c52186c7dd0b3e4a791c96ef6067e409138d7a0220c2',
 'light_score_v2.py': '9280eb171f85bd888c13a41989c14fddecb4dedcdbd63b839c1cc4268d7da49e',
 'position_simulation.py': 'f568c3cd487cc81e06c5a8846bec75c60a3dd4ef164cd24c435cfd8bd2717232',
 'simple_score.py': '3fae33cb7ed49b353ed1333d8b40170b26417211ae6abab56dedea050cb300cd'}
ACCELERATED_CODE = {'agent_evidence.py': 'd575a8dee1db473b3003bba6f02b819a5edf82a88bb59ac34cf384b947c4a7e6',
 'czsc_market_features.py': '7a8322d6d6915f990449143d0c92fd3e152fcb967026b762a81ffb975481f837',
 'czsc_native_evidence.py': 'a4dc84348de0a10f668d53df0fdb4232f42454c7e7c15d5d6a4bfde98ddd0ebf',
 'execution/broker.py': '241ea5ea95d8ab7ce9b5b564ddd24bfcc9c8eac0c17619bf080e43d062b55344',
 'execution/cost_model.py': '7ec3cadf4ab198af77135aa6084e5524895a9163d7a465f377e7d2e37d750386',
 'execution/ledger.py': '970ba7a07e586535c99fbc9c2ecf28d1b898c369f98e8808fbb1e00f5d9e30c3',
 'execution/order.py': 'd515ed66e31a4b35fee1cbfc08da76104f61971296d9c8c636cbc8c38674225e',
 'kronos_prediction.py': '9558ffa1287fe5035792af7a5a39def0b61f291dc61cda9edcc0b97dffcf3d02',
 'kronos_tuning.py': 'ef103d9531d52d97ed9c50fca4b128302a4f29b51815c12116282d06c2a2da1f',
 'light_account_continuation.py': 'b28ce05f3e21dbfff7258aaf0167bf2857e91e352a949206d1fd0ff9867507f7',
 'light_daily_scheduler.py': '46b55c7d62ba797a84a383cd7f3a7467924ff171e60aeda6d6e146bcdd543724',
 'light_experiment_registry.py': '2862971e1d3c1bc362d6b99e2e4af98ce143d5d75dc6b327fadf7475f39ab5ea',
 'light_ma_features.py': '45809c37036922f39eb3879f4d96f6a4ef5970b087363f9b0a05974de30bdbb3',
 'light_prediction_inputs.py': '75544d5f340b228272677da5acfd5e010a2545c0cfaad6f5d9194f62c725198c',
 'light_qualification.py': '1cec8361853385e97e1a026cffbc86b286a5cc8314a02d629eef7226932957a7',
 'light_research.py': '5b7663fb1a10eaac4eb041447f0aa5d8c43a58e4f456ae14c36db7060ad364a9',
 'light_score_training.py': 'd7b77962671e2d5784c8c52186c7dd0b3e4a791c96ef6067e409138d7a0220c2',
 'light_score_v2.py': '9280eb171f85bd888c13a41989c14fddecb4dedcdbd63b839c1cc4268d7da49e',
 'position_simulation.py': 'f568c3cd487cc81e06c5a8846bec75c60a3dd4ef164cd24c435cfd8bd2717232',
 'simple_score.py': '3fae33cb7ed49b353ed1333d8b40170b26417211ae6abab56dedea050cb300cd'}


def account_code_compatible(frozen, current):
    return frozen == current or bool(LEGACY_CODE and ACCELERATED_CODE
        and frozen == LEGACY_CODE and current == ACCELERATED_CODE)


def execution_adapter_identity():
    return {'schema':'account_input_pipeline_v1', 'cpu_preparation_workers':2,
        'maximum_prepared_prefixes':8, 'lookahead_observed_dates_per_symbol':2,
        'checkpoint_interval_seconds':5., 'terminal_cancel_interruption_checkpoints':'always_complete',
        'model_inference':'serial_original_rng', 'broker':'serial_original_daily_scheduler',
        'compatibility_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
