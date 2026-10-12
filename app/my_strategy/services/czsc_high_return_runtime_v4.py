"""Causal display/runtime adapter for the native v4 research population."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from collections import Counter
import pandas as pd
from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.services.czsc_high_return_pipeline_v4 import (VERSION,KIND,ENVELOPE,research_config,forward_ma,model_identity,_predict,_bundle_for,_file_hash,RISK_POLICY)
from my_strategy.services.czsc_high_return_models_v4 import HeadSession
from my_strategy.services.czsc_native_evidence_v4 import build_native_evidence_v4
from my_strategy.services.czsc_signal_points_v4 import build_signal_points,make_point_decisions
from my_strategy.services.czsc_high_return_features_v4 import entry_feature_row,event_prefix_rows
from my_strategy.storage.czsc_results import ResultStore
from my_strategy.core.run_context import stable_hash


def read_run(model_run_id=None):
    if model_run_id is None:
        runs=ResultStore().list(KIND,limit=200)
        model_run_id=next((r['run_id'] for r in runs if ResultStore().get(r['run_id']).get('config',{}).get('risk_policy')==RISK_POLICY),None)
    if model_run_id is None:return None,None
    root=(ARTIFACT_RUNS_ROOT/model_run_id).resolve()
    if not root.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()):raise ValueError('native model outside run root')
    report=json.loads((root/'reports/native_v4.json').read_text());freeze=report['protocol_freeze']
    if report['strategy_version']!=VERSION or freeze['freeze_hash']!=stable_hash({k:v for k,v in freeze.items() if k!='freeze_hash'}):
        raise ValueError('native v4 freeze contract mismatch')
    if freeze['config_hash']!=stable_hash(report['config']) or freeze['ma_bundles_hash']!=stable_hash(report['ma_bundles']) or freeze['l2_bundles_hash']!=stable_hash(report['model_bundles']):
        raise ValueError('native catalog changed after freeze')
    if any(_file_hash(path)!=sha for path,sha in freeze['code_hashes'].items()):raise ValueError('native model generating code changed after freeze')
    return root,report


def runtime_overlay(features,calendar,*,model_run_id=None,model_fold=None,model_mode='historical',as_of=None,device='cpu'):
    from my_strategy.services.czsc_research_ml import _device
    if model_mode not in {'historical','production'}:raise ValueError('native v4 requires historical or production mode')
    actual_device=str(_device(device or 'cpu'))
    frame=features.copy();frame['date']=pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
    cutoff=str(as_of or frame.iloc[-1].date)[:10];frame=frame[frame.date<=cutoff].reset_index(drop=True)
    if frame.empty or frame.symbol.nunique()!=1 or list(frame.date)!=sorted(set(frame.date)) or not set(frame.date).issubset(calendar):
        raise ValueError('native input requires a unique verified single-stock prefix')
    root,report=read_run(model_run_id);cfg=report['config'] if report else research_config()
    pack=build_native_evidence_v4(frame,cfg,market_dates=calendar);frame=pack['frame'];points=build_signal_points(frame,cfg,native_pack=pack)
    if report and (report['source_snapshot']['native_config_hash']!=pack['config_hash'] or report['source_snapshot']['native_schema_hash']!=pack['schema_hash']):raise ValueError('native event configuration mismatch')
    session=HeadSession();ma_bundles=report['ma_bundles'] if report else []
    ma_routes=forward_ma(frame,points,root or ARTIFACT_RUNS_ROOT,ma_bundles,cfg,session,model_mode)
    points=build_signal_points(frame,cfg,native_pack=pack,ma_routes=ma_routes)
    decisions=make_point_decisions(frame,points,config=cfg,include_unselected=True)
    candidates={c['date']:c for c in points['candidates']};events_by_day={};native={v['date']:v for v in pack['evaluations']}
    for e in points['events']:events_by_day.setdefault(e['date'],[]).append(e)
    prefixes=event_prefix_rows(frame,points['events'],calendar);following=dict(zip(calendar[:-1],calendar[1:]));rows=[];bundles=report['model_bundles'] if report else []
    if model_fold:
        bundles=[b for b in bundles if b['name']==model_fold]
        if not bundles or bundles[0]['available_at']>=cutoff:raise ValueError('native fixed checkpoint unavailable at cutoff')
    reason_counts=Counter();initial=float(cfg['initial_cash'])
    for i,(feature,decision) in enumerate(zip(frame.to_dict('records'),decisions)):
        day=decision['date'];stamp=day+'T15:00:00+08:00';c=candidates.get(day);plan=decision.get('entry_plan') or {}
        risk=bool(c and c['candidate_eligible'] and plan.get('entry_allowed'))
        bundle=_bundle_for(day,bundles,head='entry',model_root=root,usage_mode=model_mode);p=manifest=None
        unqualified_raw_buy=any(e['side']=='BUY' and e.get('confirmed',True) and not e.get('input_eligible') for e in events_by_day.get(day,[]))
        status=('input_ineligible' if unqualified_raw_buy else 'no_candidate') if not c else 'candidate_risk_ineligible' if not risk else 'model_unavailable' if not bundle else 'scored'
        if status=='scored':
            state={'date':day,'close':float(feature['close']),'cash':initial,'equity':initial}
            values=entry_feature_row(feature,state,c,plan,initial_cash=initial,event_prefix=prefixes[i]);values['input_eligible']=True
            p,manifest=_predict(session,values,root,bundle,'entry',mode=model_mode,policy=points['policy'],source_snapshot_hash=cfg.get('source_snapshot_hash'))
            if p is None:status='input_ineligible'
        identity=model_identity(manifest,bundle,root,day,points['policy'])
        ma=copy.deepcopy(ma_routes[day]);probability=ma['probability'];ma_state={k:feature.get(k) for k in
            ('close','close_above_ma20','ma20_slope5','ma_alignment','ret_5','ma5_ma10_gap')}
        long_buy=bool(feature.get('model_input_eligible') and feature['close_above_ma20']>0 and feature['ma20_slope5']>0 and (feature['ma_alignment']>0 or feature['ret_5']>0))
        long_sell=bool(feature.get('model_input_eligible') and feature['close_above_ma20']<=0 and feature['ma20_slope5']<0)
        ma_state.update(ma_buy=long_buy,ma_sell=long_sell,trend_long_state='bullish' if long_buy else 'bearish' if long_sell else 'neutral',
            tactical_weakening=bool(feature.get('ma5_ma10_gap',0)<0 and feature.get('ret_5',0)<0))
        ma.update(status='scored' if probability is not None else 'model_unavailable',kind='model_probability',
            direction_state={'source_id':'ma:'+day,'date':day,'available_at':stamp,'input_eligible':bool(feature.get('model_input_eligible')),
                'buy_valid':long_buy,'sell_valid':long_sell,'current_state':ma_state},
            score_date=day,target='actual_complete_fee_net_roundtrip_positive')
        hr={'kind':'model_probability','status':status,'probability':p,'threshold':bundle['policy']['entry_threshold'] if bundle else cfg['default_entry_threshold'],
            'target':'actual_complete_fee_net_roundtrip_high_return_and_initial_risk','model_identity':identity,
            'score_date':day,'in_scope':risk,'reason_codes':[] if status=='scored' else list(dict.fromkeys([status,*plan.get('reason_codes',[]),*(r for e in events_by_day.get(day,[]) for r in e.get('reason_codes',[]))]))}
        evaluations={'ma':ma,'structure':native[day]['structure'],'wyckoff':native[day]['wyckoff'],'high_return':hr}
        factual=[]
        for original in events_by_day.get(day,[]):
            event=copy.deepcopy(original);source=event['source'];side=event['side'];valid=False
            if event.get('direction_eligible') and event.get('confirmed',True):
                if source=='ma':
                    if event.get('auxiliary'):
                        slot='bs2' if 'bs2' in event['event_type'] else 'bs3'
                        expected='二买' if slot=='bs2' and side=='BUY' else '二卖' if slot=='bs2' else '三买' if side=='BUY' else '三卖'
                        confirmed=feature.get('confirmed_native_'+slot+'_at')
                        valid=bool(feature.get('confirmed_native_'+slot)==expected and confirmed and
                            str(confirmed)[:10]==str(event.get('first_confirmed_at'))[:10] and
                            str(feature.get('point_anchor'))[:10]==str(event.get('anchor_at'))[:10])
                    else:valid=long_buy if side=='BUY' else long_sell
                    valid=bool(valid and feature.get('model_input_eligible'))
                else:
                    snapshot=native[day][source]
                    for supported in snapshot.get('support_facts',[]):
                        if supported.get('original_event_id',supported['event_id'])!=event['original_event_id']:continue
                        boundary=supported.get('invalidation_price')
                        valid=bool(snapshot.get('input_eligible') and boundary is not None and
                            (float(feature['close'])>=float(boundary) if side=='BUY' else float(feature['close'])<=float(boundary)))
                        break
            event.update(candidate_state_valid=valid,candidate_state_date=day,candidate_state_available_at=stamp)
            factual.append(event)
        bundle_evidence={'version':'czsc_candidate_evidence_v4','facts':factual,
            'evaluations':evaluations,'provenance':points['policy'],'risks':copy.deepcopy(plan),
            'risk_details':[{'event_id':e['event_id'],'source':e['source'],'side':e['side'],'eligible':e['candidate_eligible'],'candidate_eligible':e['candidate_eligible'],
                'stop_price':e.get('stop_price'),'stop_distance':(float(feature['close'])-e['stop_price'])/float(feature['close']) if e.get('stop_price') else None,
                'reason_codes':e['reason_codes'],'risk_reason_codes':[r for r in e['reason_codes'] if 'risk' in r or 'stop' in r]} for e in events_by_day.get(day,[])],
            'candidate_envelope':ma['candidate_envelope']}
        reason_counts[status]+=1
        row={**decision,'native_evaluation':native[day],'candidate_evidence':bundle_evidence,'agents':evaluations,
            'model_resolution':{**identity,'reason_codes':hr['reason_codes'],'probability':p,'threshold':hr['threshold'],'publication_allowed':False,
                'historical_reconstruction':model_mode=='historical','model_family':'high_return_native_v4'},
            'ma_route':ma,'high_return_probability':p,'entry_probability':p,'probability_target':hr['target'],
            'candidate_sources':c['sources'] if c else [],'risk_eligible':risk,'source_candidate_eligible':bool(c and c['candidate_eligible']),
            'raw_risk_plan':plan or None,'reference_close':float(feature['close']),'reference_price_date':day,'next_market_session':following.get(day),
            'signal_points':events_by_day.get(day,[]),'exit_probability':None,
            'exit_model':{'probability':None,'holding_contract_compatible':False,'reason':'actual_holding_required'},
            'action':'WAIT','position_intent':'WAIT','publication_allowed':False}
        rows.append(row)
    return {'rows':rows,'latest_decision':rows[-1],'signal_points':points['events'],'candidates':points['candidates'],
        'ma_routes':list(ma_routes.values()),'native_pack':pack,'features':frame,'config':cfg,'candidate_evidence_policy':points['policy'],
        'wyckoff_events':pack['wyckoff_events'],'model_run_id':root.name if root else None,
        'model_gate':{'passed':False,'reasons':['independent_windows_not_mature','partial_frozen_cohort_development_only']},
        'publication_allowed':False,'strategy_version':VERSION,'compute_info':{'selected_device':actual_device,'actual_device':'cpu',
            'model_family':'high_return_native_v4','model_inference_rows':session.diagnostics['rows'],
            'model_inference_batches':session.diagnostics['batches'],'model_loads':session.diagnostics['model_loads'],
            'model_inference_device':'cpu','inference':dict(session.diagnostics),'source_reason_counts':dict(reason_counts)}}
