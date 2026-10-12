"""One native fact/evaluation view shared by chart, popup and history."""
from __future__ import annotations
import copy
from collections import Counter
from my_strategy.services.czsc_agent_consensus import build_consensus

def decorate_analysis(result,*,raw,features,overlay,consensus_config=None):
    out=copy.deepcopy(result);symbol=result['symbol'];display=result['frequencies']['日线']['bars']
    visible={str(b.get('date') or b.get('time'))[:10] for b in display};prefix=copy.deepcopy(overlay['rows'])
    observed={str(r['date'])[:10]:r for r in raw.to_dict('records')};basis=result.get('price_basis','unadjusted')
    for row in prefix:
        day=row['date'];bar=observed[day]
        row.update(symbol=symbol,reference_close=float(bar['close']),reference_price_date=day,
            reference_price_source='signal_day_close',reference_price_basis=basis,
            ohlc={key:float(bar[key]) for key in ('open','high','low','close')},events=row['candidate_evidence']['facts'])
        for e in row['events']:
            e.update(evidence_date=day,reference_price=float(bar['close']),reference_price_date=day,
                reference_price_source='signal_day_close',reference_price_basis=basis)
        for source in ('ma','structure','wyckoff'):
            agent=row['agents'][source];facts=[e for e in row['events'] if e['source']==source]
            agent.update(buy_count=sum(e['side']=='BUY' and e.get('direction_eligible') for e in facts),
                sell_count=sum(e['side']=='SELL' and e.get('direction_eligible') for e in facts),events=facts,
                risk_buy_count=sum(e['side']=='BUY' and e.get('candidate_eligible') for e in facts),
                risk_sell_count=sum(e['side']=='SELL' and e.get('candidate_eligible') for e in facts))
        w=row['agents']['wyckoff'];state=w['current_state']
        w.update(current_demand=state['buy_score'],current_supply=state['sell_score'],
            event_demand=state['event_demand_support'],event_supply=state['event_supply_support'],
            buy_score=state['buy_score'],sell_score=state['sell_score'],kind='rule_evidence',target='current causal price/volume rule strength, not profit probability')
        hr=row['agents']['high_return'];p=hr['probability']
        hr['evaluation_status']='risk_blocked' if hr['status']=='candidate_risk_ineligible' else 'scored_passed' if p is not None and p>=hr['threshold'] else 'scored_rejected' if p is not None else hr['status']
    consensus=build_consensus(prefix,symbol=symbol,model_family='high_return',config=consensus_config,version='czsc_agent_consensus_v4')
    final_rows={r['date']:r for r in consensus['rows']}
    for row in prefix:
        final=final_rows[row['date']]['agents']['high_return'];status=final['status']
        row['agents']['high_return']['evaluation_status']={'risk_blocked_unscored':'risk_blocked','passed':'scored_passed','rejected':'scored_rejected'}.get(status,status)
        row['agents']['high_return']['reason_codes']=final['reason_codes']
    full_consensus=copy.deepcopy(consensus);hidden=[r for r in consensus['rows'] if visible and r['date']<min(visible)]
    consensus.update(state_before_display=copy.deepcopy(hidden[-1]['state_after'] if hidden else consensus['initial_state']),
        prefix_start=prefix[0]['date'] if prefix else None,prefix_bars=len(prefix),
        rows=[r for r in consensus['rows'] if r['date'] in visible],events=[e for e in consensus['events'] if e['date'] in visible])
    wfull=overlay['wyckoff_events'];wvisible=[e for e in wfull if str(e['available_at'])[:10] in visible]
    for e in wvisible:
        day=str(e['available_at'])[:10];e.update(reference_price=float(observed[day]['close']),reference_price_date=day,reference_price_source='signal_day_close',reference_price_basis=basis)
    rows=[r for r in prefix if r['date'] in visible]
    # The browser consumes facts plus the assessed daily state. Avoid sending
    # three copies of the entire cumulative invalidation journal per row.
    # Snapshot IDs and causal invalidations today remain inspectable; the
    # full native journal is regenerated from the unchanged source prefix.
    for row in rows:
        row.pop('native_evaluation',None)
        row['candidate_evidence'].pop('evaluations',None)
        for source in ('structure','wyckoff'):
            agent=row['agents'][source]
            agent['invalidated_event_count']=len(agent.get('invalidated_event_ids',[]))
            agent['invalidations_today']=copy.deepcopy(agent.get('invalidations_at_day',[]))
            agent.pop('invalidated_event_ids',None)
    out['chart_evidence']={'version':'czsc_chart_evidence_v4','rows':rows,'candidate_assessment_version':'czsc_candidate_evidence_v4',
        'statistics':{'visible_window':{'bars':len(rows),'source_events':sum(len(r['events']) for r in rows),'wyckoff_events':len(wvisible),'final_points':len(consensus['events'])},
            'full_prefix':{'bars':len(prefix),'source_events':len(overlay['signal_points']),'wyckoff_events':len(wfull),'final_points':len(full_consensus['events'])}}}
    out['consensus_decisions']=consensus;out['wyckoff_events']=wvisible
    out['events']=[{**e, 'time':str(e['available_at'])[:10],
        'action':e['side'].lower(), 'signal':e['event_type'],
        'reason':'独立来源的原始方向候选；最终 B/S 由多 Agent 仲裁，实际成交以 Broker 账本为准'}
        for r in rows for e in r['events']]
    return out
