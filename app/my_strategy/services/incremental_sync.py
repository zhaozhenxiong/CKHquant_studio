"""Explicit, bounded structural gap plans, independent of cancelled legacy jobs."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sqlite3
import time
import uuid

from my_strategy.adapters.czsc_adapter import normalize_symbol
from my_strategy.core.market_scope import require_supported_symbol
from my_strategy.core.run_context import stable_hash
from my_strategy.services.incremental_source import IncrementalSource, code_identity, _stop
from my_strategy.services.board_calculation_scope import CALCULATION_SCOPE,calculation_membership,is_calculation_symbol

SCOPES={'current_board','selected','watch','held','full'}
PLAN_TTL=7*86400


def _prune_plan_states(store):
    """Runtime summaries share their expiring plan's lifetime, never a run log."""
    with store._tx():
        rows=store.connection.execute("SELECT name FROM settings WHERE name LIKE 'incremental_plan:%'").fetchall()
        for row in rows:
            plan_id=row['name'].split(':',1)[1]
            key=store.cache_key({'plan_id':plan_id},'incremental_plans')
            if store._live_row(key,store.clock()) is None:
                store.connection.execute('DELETE FROM settings WHERE name=?',(row['name'],))


def _catalog(kb_db_path):
    from my_strategy.storage.board_knowledge import BoardKnowledgeStore
    return [{**row,'calculation_membership':calculation_membership(row)}
            for row in BoardKnowledgeStore(kb_db_path,readonly=True).catalog(current=True)]


def _members(row):
    return row['calculation_membership']['eligible_members']


def _scope_symbols(store,scope,symbols,board_id,catalog):
    if scope not in SCOPES:raise ValueError('invalid incremental scope')
    if scope=='selected':
        if not symbols or len(symbols)>50:raise ValueError('explicit selected pool requires 1..50 symbols')
        selected=sorted({require_supported_symbol(s) for s in symbols})
        if not all(is_calculation_symbol(s) for s in selected):raise ValueError('板块结构补齐仅支持沪深 A 股')
        return selected
    if symbols:raise ValueError('symbols are only accepted for selected scope')
    if scope=='current_board':
        board=next((row for row in catalog if row['board_id']==board_id),None)
        if not board:raise ValueError('selected board is missing from current verified catalog')
        return sorted(_members(board))
    if board_id:raise ValueError('board_id requires current_board scope')
    if scope=='full':return sorted({s for row in catalog for s in _members(row)})
    if scope=='watch':
        path=store.root.parent/'personal_portfolio.db'
        if not path.is_file():return []
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as connection:
            return sorted({normalize_symbol(row[0]) for row in connection.execute('SELECT symbol FROM watchlist')
                           if is_calculation_symbol(row[0])})
    found=set()
    source=IncrementalSource(store=store)
    for account in (source._setting('enabled_accounts') or {}).values():
        if not account.get('enabled'):continue
        lookup=source._peek(store.cache_key({'id':account['last_result_id']},'result_lookup'))
        if not lookup:raise ValueError('enabled account lookup is unavailable')
        result=source._peek(store.cache_key(lookup['identity'],'simulations'))
        if not result:raise ValueError('enabled account is unavailable')
        for comparison in result.get('comparisons',{}).values():
            daily=comparison.get('daily',[])
            for symbol,position in (daily[-1].get('positions',{}) if daily else {}).items():
                if position.get('shares',0)>0 or position.get('receivable_shares',0)>0:found.add(symbol)
    return sorted(s for s in found if is_calculation_symbol(s))


def _contract(store,raw_db_path,kb_db_path,scope,symbols,board_id,price_date):
    from my_strategy.services.board_fluent import source_witness,generation_contract,latest_qualified_price_date
    from my_strategy.services.czsc_trend_support import _calendar
    latest=latest_qualified_price_date(raw_db_path=raw_db_path)
    day=price_date or latest
    if day!=latest:raise ValueError('补齐只允许真实最新行情日，不以旧日替代')
    dates,_=_calendar(None);dates=[d for d in dates if d<=day]
    if not dates or dates[-1]!=day:raise ValueError('补齐目标日不在已核验日历')
    catalog=_catalog(kb_db_path)
    population=_scope_symbols(store,scope,symbols,board_id,catalog)
    if scope=='full' and not catalog:raise ValueError('full scope requires a verified current board catalog')
    contract={'scope':scope,'board_id':board_id,'symbols':population,'price_date':day,
        'calculation_scope':CALCULATION_SCOPE,
        'projection_hashes':{r['board_id']:r['calculation_membership']['projection_hash'] for r in catalog},
        'catalog':{r['board_id']:r['snapshot_hash'] for r in catalog},
        'calendar_hash':stable_hash(dates),'producer':code_identity(),'fluent':generation_contract(),
        # Catalog content hashes above prove membership. A readonly catalog
        # connection may coincide with a committed SQLite WAL checkpoint;
        # that physical KB timestamp change is not a new input.
        'witness':{'raw':source_witness(raw_db_path=raw_db_path,kb_db_path=kb_db_path)['raw']}}
    return contract,catalog,dates


def create_plan(*,store,raw_db_path=None,kb_db_path=None,fluent_db_path=None,scope='watch',symbols=None,
                board_id=None,price_date=None,max_symbols=20,budget_seconds=300):
    if type(max_symbols) is not int or not 1<=max_symbols<=100:raise ValueError('max_symbols must be 1..100')
    if type(budget_seconds) is not int or not 1<=budget_seconds<=1800:raise ValueError('budget_seconds must be 1..1800')
    contract,catalog,dates=_contract(store,raw_db_path,kb_db_path,scope,symbols,board_id,price_date)
    source=IncrementalSource(store=store,raw_db=raw_db_path);rows=[]
    for symbol in contract['symbols']:
        try:
            row=source.inspect(symbol,contract['price_date'],calendar_dates=dates)
            if row['observed_end']!=contract['price_date']:row.update(action='missing_price',reason='target_day_price_missing')
        except (ValueError,OSError,KeyError) as exc:
            row={'symbol':symbol,'action':'failed','reason':str(exc),'price_date':contract['price_date']}
        rows.append(row)
    counts={name:sum(row['action']==name for row in rows) for name in ('reuse','append','rebuild','missing_price','failed')}
    requested=set(contract['symbols']);global_population={s for row in catalog for s in _members(row)}
    now=store.clock();plan_id=uuid.uuid4().hex
    _prune_plan_states(store)
    plan={'plan_id':plan_id,'format':'incremental_gap_plan_v1','status':'planned','created_at':now,'expires_at':now+PLAN_TTL,
        'scope':scope,'board_id':board_id,'price_date':contract['price_date'],'requested':len(rows),'rows':rows,'counts':counts,
        'estimated_compute':counts['append']+counts['rebuild'],'estimated_deferred':max(0,counts['append']+counts['rebuild']-max_symbols),
        'max_symbols':max_symbols,'budget_seconds':budget_seconds,'contract':contract,
        'publication':{'full_scope':scope=='full','global_population':len(global_population),
            'affected_boards':sum(bool(requested.intersection(_members(r))) for r in catalog)},
        'training_started':False,'legacy_tasks_resumed':False}
    # This is a small expiring plan, not a numerical output or permanent run.
    store.put({'plan_id':plan_id},plan,namespace='incremental_plans',ttl_seconds=PLAN_TTL)
    return plan


def get_plan(*,store,plan_id):
    plan=store.get({'plan_id':plan_id},namespace='incremental_plans')
    if not plan or plan['expires_at']<=store.clock():raise KeyError('补齐计划不存在或已过期，请重新预览')
    state=store.get_setting('incremental_plan:'+plan_id,{})
    result=store.get_by_key(state['result_key']) if state.get('result_key') else None
    return {**plan,**state,**({'result':result} if result else {})}


def _unavailable(symbol,day,reason):
    value={'symbol':symbol,'price_date':day,'mode':'current_reference','training_eligible':False,
        'status':'unavailable','reason_codes':[reason],'structure':None,'wyckoff':None}
    value['row_hash']=stable_hash(value);return value


def execute_plan(*,store,plan_id,raw_db_path=None,kb_db_path=None,fluent_db_path=None,stop=None,progress=None):
    plan=get_plan(store=store,plan_id=plan_id)
    if plan['status']=='cancelled':raise ValueError('取消的计划保持终态，请新建补齐计划')
    if plan['status'] in {'running','failed'}:raise ValueError('该计划不能重复执行，请重新预览缺口')
    if plan['status'] in {'completed','partial'} and plan.get('result'):return {**plan['result'],'reused_plan':True}
    if plan['status']!='planned':raise ValueError('原计划结果已过期，请重新预览缺口')
    _stop(stop)
    contract,catalog,dates=_contract(store,raw_db_path,kb_db_path,plan['scope'],
        plan['contract']['symbols'] if plan['scope']=='selected' else None,plan.get('board_id'),plan['price_date'])
    if contract!=plan['contract']:
        changed=[key for key in contract if contract[key]!=plan['contract'].get(key)]
        raise ValueError('补齐计划输入已变化，请重新预览：'+','.join(changed))
    source=IncrementalSource(store=store,raw_db=raw_db_path)
    # Recheck each scoped input before computing, including external revisions.
    for row in plan['rows']:
        _stop(stop)
        if row.get('identity'):
            if source.inspect(row['symbol'],plan['price_date'],calendar_dates=dates)['identity']!=row['identity']:
                raise ValueError('补齐计划逐股输入已变化，请重新预览')
    store.set_setting('incremental_plan:'+plan_id,{'status':'running','started_at':store.clock()})
    deadline=time.monotonic()+plan['budget_seconds'];computed=0;rows=[];summaries={};generation=None;full=False
    cancelled=False;error=None
    try:
        for frozen in plan['rows']:
            _stop(stop);symbol=frozen['symbol'];action=frozen['action']
            if action=='missing_price':
                summaries[symbol]=_unavailable(symbol,plan['price_date'],'target_day_price_missing')
                row={'symbol':symbol,'status':'missing_price','observed_end':frozen['observed_end']}
            elif action=='failed':row={'symbol':symbol,'status':'failed','error':frozen['reason']}
            elif action!='reuse' and (computed>=plan['max_symbols'] or time.monotonic()>=deadline):
                row={'symbol':symbol,'status':'deferred','reason':'本批计算数量或时间预算已用完'}
            else:
                try:
                    _,frame,_,_=source.source(symbol,plan['price_date'],stop=stop,calendar_dates=dates)
                    stats=frame.attrs['incremental']
                    summaries[symbol]=source.summary(symbol,plan['price_date'],stop=stop,calendar_dates=dates)
                    status='reused' if stats['reused'] else 'appended' if stats['mode']=='native_state_append' else 'rebuilt'
                    if status!='reused':computed+=1
                    row={'symbol':symbol,'status':status,'incremental':stats}
                except InterruptedError:raise
                except (ValueError,KeyError,OSError) as exc:row={'symbol':symbol,'status':'failed','error':str(exc)}
            rows.append(row)
            if progress:progress({'stage':'incremental_source','completed':len(rows),'total':plan['requested'],
                'symbol':symbol,'counts':dict(Counter(r['status'] for r in rows))})
        _stop(stop)
        current,_,_=_contract(store,raw_db_path,kb_db_path,plan['scope'],
            plan['contract']['symbols'] if plan['scope']=='selected' else None,plan.get('board_id'),plan['price_date'])
        if current!=contract:raise ValueError('计算期间输入或生成码变化，未发布板块结果')
        if catalog and summaries:
            from my_strategy.services.board_fluent import build_board_fluent
            from my_strategy.storage.board_fluent import BoardFluentStore
            published_store=BoardFluentStore(fluent_db_path,readonly=True)
            old=published_store.current_generation('market')
            old_identity=(old or {}).get('native_identity') or {}
            inherited_complete=False
            if (old and old_identity.get('format')=='incremental_native_summary_collection_v1' and
                old_identity.get('calculation_scope')==CALCULATION_SCOPE and
                old_identity.get('producer')==contract['producer'] and old_identity.get('price_date')==plan['price_date'] and
                old_identity.get('calendar_hash')==contract['calendar_hash'] and
                old.get('publication_inputs',{}).get('source_witness',{}).get('raw')==contract['witness']['raw']):
                prior=published_store.get_native_component(old['generation_id'],allow_partial=True)
                inherited_complete=prior['complete']
                # Membership changes affect aggregation only; verified same-day
                # stock observations retain their exact numerical identity.
                for symbol in old_identity.get('completed_symbols',[]):
                    if is_calculation_symbol(symbol) and symbol not in summaries:summaries[symbol]=prior['summaries'][symbol]
            all_symbols=sorted({s for board in catalog for s in _members(board)})
            native_rows={s:summaries.get(s) or _unavailable(s,plan['price_date'],'outside_completed_plan_scope') for s in all_symbols}
            refs=[{'entity_id':s,'source_identity':r['identity'],'row_hash':r['row_hash']} for s,r in sorted(native_rows.items()) if r.get('identity')]
            complete=(plan['scope']=='full' or inherited_complete) and all(s in summaries for s in all_symbols)
            identity={'format':'incremental_native_summary_collection_v1','price_date':plan['price_date'],
                'calculation_scope':CALCULATION_SCOPE,'projection_hashes':contract['projection_hashes'],
                'mode':'current_reference','training_eligible':False,'required_symbols':all_symbols,'input_refs_hash':stable_hash(refs),
                'calendar_hash':stable_hash(dates),'producer':code_identity(),'scope':plan['scope'],
                'completed_symbols':sorted(summaries),'catalog_hash':stable_hash(contract['catalog'])}
            native={'identity':identity,'summaries':native_rows,'input_refs':refs,'complete':complete,
                'processed_symbols':sorted(summaries),'diagnostics':{'scope':plan['scope'],'requested':len(all_symbols),
                    'verified':sum(r['status']=='available' for r in native_rows.values()),'processed':len(summaries),
                    'unavailable':sum(r['status']!='available' for r in native_rows.values()),'local_plan_complete_is_not_full':not complete}}
            published=build_board_fluent(kb_db_path=kb_db_path,raw_db_path=raw_db_path,fluent_db_path=fluent_db_path,
                calendar_dates=dates,price_date=plan['price_date'],with_native=True,native_result=native,
                check_cancel=lambda:_stop(stop),migrate=False,component_store=store)
            generation=published['generation_id'];full=published['full_published']
    except InterruptedError:
        cancelled=True
    except Exception as exc:error=str(exc)
    covered={row['symbol'] for row in rows}
    rows.extend({'symbol':r['symbol'],'status':'deferred','reason':'cancelled' if cancelled else 'execution_stopped'}
                for r in plan['rows'] if r['symbol'] not in covered)
    counts={key:sum(row['status']==key for row in rows) for key in ('reused','appended','rebuilt','missing_price','failed','deferred')}
    status='cancelled' if cancelled else 'failed' if error else 'partial' if counts['failed'] or counts['deferred'] else 'completed'
    result={'plan_id':plan_id,'scope':plan['scope'],'price_date':plan['price_date'],'status':status,'error':error,
        'requested':plan['requested'],'processed':sum(counts[k] for k in ('reused','appended','rebuilt','missing_price')),
        'rows':rows,'counts':counts,'generation_id':generation,'full_published':full,
        'training_started':False,'cancelled_tasks_resumed':False,'native_workers':1}
    ref=store.put({'plan_id':plan_id,'outcome':stable_hash(result)},result,namespace='incremental_plan_results')
    store.set_setting('incremental_plan:'+plan_id,{'status':status,'finished_at':store.clock(),'result_key':ref['cache_key']})
    return result
