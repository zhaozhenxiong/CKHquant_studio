"""Current board derivation; KB supplies mappings, Fluent alone receives outputs."""
from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path
import time

from my_strategy.adapters.czsc_adapter import _connection, raw_path
from my_strategy.core.paths import BASIC_KB_DB_PATH, DATA_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.services import board_features as mathematics
from my_strategy.services.board_calculation_scope import (CALCULATION_SCOPE, calculation_membership,
    calculation_contract, is_calculation_symbol)
from my_strategy.services.czsc_trend_support import VERIFIED_SOURCES, _calendar
from my_strategy.storage.board_fluent import BoardFluentStore, RANK_POLICY, digest
from my_strategy.storage.board_knowledge import BoardKnowledgeStore

FLUENT_PROFILE = {'version':'current_board_fluent_v2','mode':'current_reference','calculation_scope':CALCULATION_SCOPE,
                  'market_schema_hash':mathematics.BOARD_SCHEMA_HASH,'ranking_policy':RANK_POLICY,
                  'native_basis':'member_native','capital_basis':'turnover_amount_proxy'}


def _file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generation_contract():
    from my_strategy.services import board_native_summary
    return {'profile':FLUENT_PROFILE,'code_hashes':{Path(path).resolve().relative_to(Path(__file__).resolve().parent.parent).as_posix():_file_hash(path) for path in
        (inspect.getfile(BoardFluentStore),__file__)},
        'native_summary_schema_hash':board_native_summary.SUMMARY_SCHEMA_HASH,
        'native_summary_reader_sha256':_file_hash(inspect.getfile(board_native_summary)),
        'math_code_hashes':mathematics._code_hashes(BoardKnowledgeStore(readonly=True))}


def _witness(path):
    path=Path(path).resolve();result={'path':str(path)}
    for name,target in [('db',path),('wal',Path(str(path)+'-wal'))]:
        try:
            stat=target.stat();result[name]=None if name=='wal' and stat.st_size<=32 else {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        except FileNotFoundError:result[name]=None
    return result


def source_witness(*,raw_db_path=None,kb_db_path=None):
    return {'raw':_witness(raw_path(raw_db_path)),'knowledge':_witness(kb_db_path or BASIC_KB_DB_PATH)}


def latest_qualified_price_date(raw_db_path=None):
    with _connection(raw_db_path) as connection:
        rows=connection.execute('SELECT stock,MAX(date) FROM stock_daily_normalized WHERE has_trade_price=1 AND source IN ('+
            ','.join('?' for _ in VERIFIED_SOURCES)+') GROUP BY stock',sorted(VERIFIED_SOURCES)).fetchall()
        day=max((row[1] for row in rows if row[1] and is_calculation_symbol(row[0])),default=None)
    if not day:raise ValueError('no qualified local price date')
    return day


def freshness_status(*,fluent_db_path=None,raw_db_path=None,kb_db_path=None,source_root=None,native_config=None):
    """Cheap read-only witness check; content reuse is checked by the builder."""
    store=BoardFluentStore(fluent_db_path,readonly=True)
    market=store.current_generation('market');full=store.current_generation('full');reasons=[]
    if market is None:return {'available':False,'market':None,'full':None,'stale':True,'stale_reasons':['no_published_market'],
                              'needs_rebuild':True,'source_root':str(source_root) if source_root else None}
    actual=source_witness(raw_db_path=raw_db_path,kb_db_path=kb_db_path)
    frozen=market.get('publication_inputs',{}).get('source_witness',{})
    if actual.get('raw')!=frozen.get('raw'):reasons.append('raw_date_or_revision_changed')
    if actual.get('knowledge')!=frozen.get('knowledge'):reasons.append('membership_source_changed')
    if market.get('fluent_contract')!=generation_contract():reasons.append('schema_policy_or_generating_code_changed')
    # Native producers are validated by the collector, never by the staged module.
    native_stale=False
    market_native=market.get('native_identity') or {}
    if market_native.get('format')=='incremental_native_summary_collection_v1':
        from my_strategy.services.incremental_source import code_identity
        try:native_stale=market_native.get('producer')!=code_identity()
        except (OSError,ValueError,KeyError):native_stale=True
    elif full is not None and source_root is not None:
        recorded=full.get('native_identity') or {}
        generation=recorded.get('generation_contract') or {}
        try:
            if recorded.get('format')=='incremental_native_summary_collection_v1':
                from my_strategy.services.incremental_source import code_identity
                native_stale=recorded.get('producer')!=code_identity()
            else:
                from my_strategy.services.board_native_summary import native_generation_contract
                native_stale=(native_generation_contract(source_root=source_root,config=native_config or generation.get('config'))!=generation or
                              recorded.get('reader_code_sha256')!=_file_hash(inspect.getfile(native_generation_contract)))
        except (OSError,ValueError,KeyError):native_stale=True
        for relative,expected in generation.get('code_hashes',{}).items():
            path=Path(source_root)/'my_strategy'/relative
            if not path.is_file() or _file_hash(path)!=expected:native_stale=True
        cache_witness=full.get('publication_inputs',{}).get('native_cache_witness')
        if cache_witness is not None and _witness(cache_witness['path'])!=cache_witness:native_stale=True
    return {'available':True,'market':market,'full':full,'stale':bool(reasons),'stale_reasons':reasons,
            'native_stale':native_stale or bool(reasons),'needs_rebuild':bool(reasons),
            'source_witness':actual,'price_date':market['price_date'],'membership_date':market.get('membership_date'),
            'source_root':str(source_root) if source_root else None,'training_started':False}


def build_board_fluent(*,kb_db_path=None,raw_db_path=None,fluent_db_path=None,calendar_dates=None,
                      calendar_run_id=None,price_date=None,symbols=None,with_native=False,native_result=None,
                      native_cache_db_path=None,source_native_run_id=None,raw_source_run_id=None,
                      source_root=None,artifacts_root=None,native_config=None,sync_id=None,publish=True,migrate=True,progress=None,
                      check_cancel=None,component_store=None):
    """Build latest-only rows; never fetch remote data or prepare/train native."""
    if check_cancel:check_cancel()
    started=time.perf_counter();store=BoardFluentStore(fluent_db_path)
    day=latest_qualified_price_date(raw_db_path)
    if price_date is not None and price_date!=day:raise ValueError('board Fluent builds latest qualified price date only')
    if calendar_dates is None:calendar_dates,calendar_run_id=_calendar(calendar_run_id)
    if day not in calendar_dates:raise ValueError('latest qualified price date is outside verified calendar')
    calendar_prefix_hash=stable_hash(mathematics._calendar(calendar_dates,day))
    contract=generation_contract();witness=source_witness(raw_db_path=raw_db_path,kb_db_path=kb_db_path)
    reader=BoardKnowledgeStore(kb_db_path,readonly=True)
    original_catalog=reader.catalog
    snapshots=reader.catalog(current=True)
    snapshots=[value for value in snapshots if value['kind'] in {'industry','concept'}]
    if not snapshots:raise ValueError('no complete current board membership mappings')
    frozen_snapshots={value['board_id']:value['snapshot_hash'] for value in snapshots}
    memberships={value['board_id']:calculation_membership(value) for value in snapshots}
    scope_identity=calculation_contract(memberships)
    lease=None
    if sync_id is None:
        lease_identity={'source_witness':witness,'snapshot_hashes':frozen_snapshots,'calendar_dates':list(calendar_dates),
                        **scope_identity,
                        'contract':contract,'with_native':with_native,'source_native_run_id':source_native_run_id,
                        'raw_source_run_id':raw_source_run_id,'native_config':native_config,
                        'source_root':str(source_root) if source_root else None,
                        'symbols':sorted(set(symbols)) if symbols else None}
        lease=store.acquire_sync(digest(lease_identity),lease_identity,reuse_completed=False)
        sync_id=lease['sync_id']
        if not lease['owned']:
            return {**(lease.get('result') or {}),'sync_id':sync_id,'reused_sync':True,'status':lease['status'],
                    'training_started':False,'generation_id':(lease.get('result') or {}).get('generation_id'),
                    'cache_hit':True,'market_published':bool((lease.get('result') or {}).get('market_published')),
                    'full_published':bool((lease.get('result') or {}).get('full_published'))}

    def report(stage,current=0,total=1,**details):
        if lease is not None:store.heartbeat_sync(sync_id,stage=stage)
        if progress:progress({'event':'feature_sync','sync_id':sync_id,'stage':stage,'status':'running',
                              'current':current,'total':total,**details})

    cache_events=[];migration=None
    original_get=reader.get_feature

    def validate_sources():
        if check_cancel:check_cancel()
        actual=source_witness(raw_db_path=raw_db_path,kb_db_path=kb_db_path)
        if actual['raw']!=witness['raw']:raise ValueError('board source changed during preparation')
        if actual['knowledge']!=witness['knowledge']:
            # SQLite can checkpoint an already committed KB WAL while readers
            # run. Treat that as a witness change, then prove semantic equality.
            current={value['board_id']:value['snapshot_hash'] for value in original_catalog(current=True)
                     if value['kind'] in {'industry','concept'}}
            if current!=frozen_snapshots:raise ValueError('board membership source changed during preparation')
            witness['knowledge']=source_witness(raw_db_path=raw_db_path,kb_db_path=kb_db_path)['knowledge']

    def frozen_catalog(kind=None,as_of=None,current=False):
        if not current or as_of is not None:raise ValueError('Fluent membership is current-only')
        return copy.deepcopy([value for value in snapshots if kind is None or value['kind']==kind])

    def get_feature(kind,key):
        nonlocal migration
        if kind!='daily':raise ValueError('unsupported Fluent legacy cache kind')
        value=store.load_market(key,contract)
        if value is not None:cache_events.append('fluent');return value
        if migrate:
            value=original_get(kind,key)
            if value is not None:
                migration={'source_db':str(reader.path),'source_cache_key':key,'source_payload_sha256':digest(value),
                           'exact_original_bundle':True,'actual_content_identity_verified_by_original_math':True}
                cache_events.append('legacy');return value
        cache_events.append('miss');return None

    def put_feature(kind,key,value):
        if kind!='daily' or key!=value.get('cache_key') or stable_hash(value.get('cache_identity'))!=key:
            raise ValueError('Fluent mathematical cache identity drift')
        return value

    # Keep the exact original store type so the math's generation hashes remain
    # BoardKnowledgeStore hashes, not an adapter-class or staging fiction.
    reader.catalog=frozen_catalog;reader.get_feature=get_feature;reader.put_feature=put_feature
    try:
        report('board_market',0,len(snapshots))
        market=mathematics.build_board_features(reader,raw_db_path=raw_db_path,calendar_dates=calendar_dates,
                                              price_date=day,mode='current_reference',symbols=symbols,component_store=component_store)
        validate_sources()
        if generation_contract()!=contract:raise ValueError('Fluent generating code changed during preparation')
        if market.get('calculation_membership')!=memberships or market.get('calculation_scope')!=CALCULATION_SCOPE:
            raise ValueError('board mathematical calculation projection differs from frozen source')
        publication={'source_witness':witness,'snapshot_hashes':frozen_snapshots,**scope_identity}
        if check_cancel:check_cancel()
        existing_full=store.current_generation('full')
        full_market_compatible=(existing_full is not None and existing_full['cache_key']==market['cache_key'] and
            existing_full['fluent_contract']==contract and existing_full['publication_inputs'].get('source_witness')==witness)
        # A supplied local/native collection is already ready for attachment;
        # expose its market and summary together after the transaction verifies.
        generation_id=store.save_market(market,contract=contract,sync_id=sync_id,publish=publish and not full_market_compatible and native_result is None,
                                        migration=migration,publication_inputs=publication)
        if full_market_compatible and not with_native:generation_id=existing_full['generation_id']
        if migration is not None and store.load_market(market['cache_key'],contract)!=market:
            raise ValueError('legacy migration changed mathematical output')
        report('board_market',len(snapshots),len(snapshots),generation_id=generation_id)
        native_diagnostics=None
        outcome='completed'
        if with_native:
            from my_strategy.services.board_native_summary import (collect_native_summaries,aggregate_board_native,
                                                                  validate_native_reuse,native_generation_contract)
            requested_symbols=scope_identity['required_symbols']
            report('native_summary',0,len(requested_symbols))
            def native_progress(**values):
                stage=values.pop('stage','native_summary');current=values.pop('current',0);total=values.pop('total',1)
                report(stage,current,total,**values)
            reuse_full=False
            reusable_native=existing_full if existing_full is not None and existing_full['price_date']==day else store.ready_full_generation(day)
            if reusable_native is not None and native_result is None:
                old=reusable_native['native_identity'] or {}
                old_contract=old.get('generation_contract',{})
                explicit_matches=((source_native_run_id is None or source_native_run_id==old.get('source_native_run_id')) and
                                  (raw_source_run_id is None or raw_source_run_id==old.get('raw_source_run_id')))
                cache_matches=reusable_native['publication_inputs'].get('native_cache_witness')==_witness(
                    native_cache_db_path or DATA_ROOT/'processed/czsc/native_structure.db')
                config_matches=native_config is None or native_generation_contract(source_root=source_root,config=native_config)==old_contract
                calendar_matches=old.get('calendar_hash')==calendar_prefix_hash
                population_matches=(old.get('calculation_scope')==CALCULATION_SCOPE and
                    old.get('required_symbols')==requested_symbols)
                if explicit_matches and cache_matches and config_matches and calendar_matches and population_matches:
                    # Actual complete raw/liquidity/calendar/source proof is
                    # checked again, but no native history blob is decoded.
                    verified=validate_native_reuse(old,raw_db_path=raw_db_path,source_root=source_root,
                        artifacts_root=artifacts_root,progress=native_progress,check_cancel=check_cancel)
                    if verified['valid']:
                        store.verify_generation(reusable_native['generation_id'],level='full')
                        validate_sources()
                        indexed_market_compatible=(reusable_native['cache_key']==market['cache_key'] and
                            reusable_native['fluent_contract']==contract and reusable_native['publication_inputs'].get('source_witness')==witness)
                        if indexed_market_compatible:
                            generation_id=reusable_native['generation_id'];reuse_full=True
                            native_diagnostics={**verified,'indexed_full_reused':True,'native_blob_decode':False}
                            if check_cancel:check_cancel()
                            if publish:store.publish_full(generation_id)
                        else:
                            native_result=store.get_native_component(reusable_native['generation_id'])
                            if old.get('projection_hashes')!=scope_identity['projection_hashes']:
                                # Reuse verified per-stock facts under a new collection
                                # identity; the old board aggregate is never relabelled.
                                native_result['identity']={**native_result['identity'],**scope_identity,
                                    'source_membership_rebind':{'parent_identity_hash':digest(old),
                                        'parent_generation_id':reusable_native['generation_id']}}
                            native_result['diagnostics']={**native_result['diagnostics'],**verified,
                                'indexed_native_reused':True,'indexed_native_generation_id':reusable_native['generation_id'],
                                'native_blob_decode':False}
            if native_result is None and not reuse_full:
                if not requested_symbols:
                    native_result={'identity':{'price_date':day,**scope_identity},'complete':False,
                        'summaries':{},'input_refs':[],
                        'diagnostics':{'requested':0,'verified':0,'unavailable':0,'reason':'no_supported_members'}}
                else:
                    native_result=collect_native_summaries(price_date=day,
                    symbols=requested_symbols,
                    cache_db_path=native_cache_db_path,source_native_run_id=source_native_run_id,
                    raw_source_run_id=raw_source_run_id,source_root=source_root,artifacts_root=artifacts_root,
                    raw_db_path=raw_db_path,config=native_config,progress=native_progress,check_cancel=check_cancel,
                    calculation_memberships=memberships)
            validate_sources()
            if not reuse_full:native_diagnostics=native_result.get('diagnostics',{})
            incremental_partial=(not reuse_full and native_result['identity'].get('format')=='incremental_native_summary_collection_v1')
            if not reuse_full and requested_symbols and (native_result.get('complete') is True or incremental_partial):
                if any(native_result['identity'].get(key)!=value for key,value in scope_identity.items()):
                    raise ValueError('native collection calculation projection differs from market')
                if native_result['identity'].get('calendar_hash')!=calendar_prefix_hash:
                    raise ValueError('native component calendar prefix differs from current verified calendar')
                publication={**publication,'native_identity_hash':digest(native_result['identity'])}
                if not incremental_partial:
                    publication['native_cache_witness']=_witness(native_cache_db_path or DATA_ROOT/'processed/czsc/native_structure.db')
                generation_id=store.save_market(market,contract=contract,sync_id=sync_id,publish=False,
                                                migration=migration,publication_inputs=publication)
                processed=set(native_result.get('processed_symbols',[]))
                aggregates=[aggregate_board_native(value,({**native_result,'complete':all(symbol in processed for symbol in memberships[value['board_id']]['eligible_members'])}
                    if incremental_partial else native_result),min_members=market['policy']['min_members'],
                            min_coverage=market['policy']['min_coverage']) for value in snapshots]
                store.attach_native(generation_id,native_result,aggregates,publish=False,allow_partial=incremental_partial)
                if generation_contract()!=contract:raise ValueError('Fluent generating code changed during native preparation')
                if check_cancel:check_cancel()
                if native_result.get('complete') is True:
                    if publish:store.publish_full(generation_id)
                else:
                    outcome='partial'
                    if publish:store.publish(generation_id,'market')
            elif not reuse_full:
                outcome='partial'
                if check_cancel:check_cancel()
                if publish:store.publish(generation_id,'market')
            report('native_summary',len(snapshots),len(snapshots),generation_id=generation_id)
        generation=store.get_generation(generation_id)
        summary={'status':outcome,'sync_id':sync_id,'generation_id':generation_id,
                 'calculation_scope':CALCULATION_SCOPE,'projection_hashes':scope_identity['projection_hashes'],
                 'market_published':generation['market_published'],'full_published':generation['full_published'],
                 'cache_hit':cache_events[-1]!='miss','cache_source':cache_events[-1],'migration':migration is not None,
                 'price_date':day,'membership_date':market['membership_date'],'board_count':len(market['boards']),
                 'stock_count':len(market['stocks']),'market_basis_cache_key':market['cache_key'],
                 'raw_fingerprint':market['cache_identity']['price_content_sha256'],'native_diagnostics':native_diagnostics,
                 'elapsed_seconds':time.perf_counter()-started,'source_witness':witness,'training_started':False,
                 'mode':'current_reference','capital_basis':'turnover_amount_proxy','calendar_run_id':calendar_run_id}
        if lease is not None:store.finish_sync(sync_id,status=outcome,result=summary)
        return summary
    except BaseException as error:
        if lease is not None:store.finish_sync(sync_id,status='cancelled' if type(error).__name__=='Cancelled' else 'failed',error=str(error))
        raise
