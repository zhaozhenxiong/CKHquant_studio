/* Group dated evidence without changing its source identity or visibility. */
((root) => {
  'use strict';
  const day = (value) => String(value || '').slice(0,10);
  function periodKey(value, frequency) {
    const text = day(value);
    if (frequency === '日线') return text;
    if (frequency === '月线') return text.slice(0,7);
    const date = new Date(`${text}T00:00:00Z`);
    date.setUTCDate(date.getUTCDate() + 4 - (date.getUTCDay() || 7));
    const year = date.getUTCFullYear();
    const week = Math.ceil(((date - Date.UTC(year,0,1)) / 86400000 + 1) / 7);
    return `${year}-${String(week).padStart(2,'0')}`;
  }
  const groupId = (symbol, frequency, time, kind, action) =>
    ['khq-marker',symbol,frequency,time,kind,action].map((value) => encodeURIComponent(value || '')).join(':');
  function build({bars = [], dailyBars = [], frequency, events = [], trades = [], symbol,
                  start = '', asOf, up, down, includeExcluded = false}) {
    const mappings = [], grouped = new Map(), seenEvents = new Set();
    const byPeriod = new Map(bars.filter((bar) => bar.is_closed !== false)
      .map((bar) => [periodKey(bar.time,frequency),day(bar.time)]));
    const observedDaily = dailyBars.map((bar) => day(bar.time)).filter(Boolean).sort();
    let intentCount = 0, tradeCount = 0, deferredCount = 0, ignoredCount = 0;
    const add = (row, execution) => {
      if (row.symbol && symbol && row.symbol !== symbol) return;
      const original = day(execution ? row.date : row.evidence_date || row.available_at || row.signal_at || row.time || row.date);
      if (!original || original < start || (asOf && original > asOf)) return;
      const action = row.action || row.side || '';
      const buying = /buy|open|买|开/i.test(action);
      const selling = /sell|close|卖|平/i.test(action);
      if (!buying && !selling) return;
      const rowSymbol = row.symbol || symbol || '';
      // Only identical source event identities are duplicates. Distinct events
      // from one agent, and every actual fill, remain separate evidence rows.
      if (!execution && row.event_id) {
        const identity = `${rowSymbol}:${row.event_id}`;
        if (seenEvents.has(identity)) return;
        seenEvents.add(identity);
      }
      const exclusionReasons = execution ? [] : [
        ...(row.evidence_visible === false ? ['evidence_not_visible'] : []),
        ...(row.candidate_eligible === false ? ['candidate_ineligible'] : []),
      ];
      const excluded = exclusionReasons.length > 0;
      const displayed = !excluded || includeExcluded;
      if (excluded) ignoredCount++;
      const key = periodKey(original,frequency), time = byPeriod.get(key);
      const closedAt = frequency === '日线' ? time : observedDaily.find((date) => periodKey(date,frequency) > key);
      if (!time || time < original || !closedAt || (asOf && closedAt > asOf)) {
        if (displayed) deferredCount++;
        return;
      }
      const kind = execution ? 'execution' : 'intention', side = buying ? 'BUY' : 'SELL';
      const id = groupId(rowSymbol,frequency,time,kind,side);
      const source = row.source || (execution ? 'broker' : null);
      const mapping = {...row, row, group_id:id, kind, symbol:rowSymbol, action:side, side,
        source, event_id:row.event_id || null, event_type:row.event_type || row.signal || null,
        original_date:original, event_date:day(row.date || row.time || row.signal_at || original),
        bar_date:time, closed_at:closedAt, excluded, displayed, exclusion_reasons:exclusionReasons};
      mappings.push(mapping);
      if (!grouped.has(id)) grouped.set(id, {id, group_id:id, symbol:rowSymbol, frequency,
        time, bar_date:time, closed_at:closedAt, kind, action:side, side, entries:[], events:[]});
      const group = grouped.get(id);
      group.entries.push(mapping);
      group.events.push(row);
      if (displayed) execution ? tradeCount++ : intentCount++;
    };
    events.forEach((event) => add(event,false));
    // A chart never borrows an unrelated symbol's fill or anonymous ledger row.
    trades.filter((trade) => trade.symbol && (!symbol || trade.symbol === symbol)).forEach((trade) => add(trade,true));
    const groups = [...grouped.values()].sort((a,b) => a.time.localeCompare(b.time)
      || (a.action === b.action ? 0 : a.action === 'BUY' ? -1 : 1)
      || (a.kind === b.kind ? 0 : a.kind === 'intention' ? -1 : 1));
    const markers = [];
    for (const group of groups) {
      group.entries.sort((a,b) => a.original_date.localeCompare(b.original_date));
      group.events = group.entries.map((entry) => entry.row);
      const visible = group.entries.filter((entry) => entry.displayed);
      group.sources = [...new Set(visible.map((entry) => entry.source).filter(Boolean))].sort();
      group.source_count = group.sources.length;
      group.event_count = group.kind === 'intention' ? visible.length : 0;
      group.trade_count = group.kind === 'execution' ? visible.length : 0;
      group.total_count = group.entries.length;
      group.excluded_count = group.entries.filter((entry) => entry.excluded).length;
      group.visible = visible.length > 0;
      if (!group.visible) continue;
      const buying = group.action === 'BUY', execution = group.kind === 'execution';
      const letter = buying ? 'B' : 'S';
      markers.push({id:group.id, time:group.time, position:buying ? 'belowBar' : 'aboveBar',
        color:buying ? up : down,
        shape:execution ? buying ? 'circle' : 'square' : buying ? 'arrowUp' : 'arrowDown',
        text:execution ? `实${letter}` : letter,
        size:execution ? 1.2 : 1});
    }
    return {markers,mappings,groups,intentCount,tradeCount,deferredCount,ignoredCount};
  }
  function describe(mappings, time, frequency) {
    if (!time) return '点击最终 B/S 或对应 K 线，查看多 Agent 判断；悬浮只切换指标与筹码。';
    const items = mappings.filter((item) => item.bar_date === time && item.displayed !== false);
    if (!items.length) return `${time} · 该K线无买卖标记`;
    return items.map((item) => `${item.action === 'BUY' ? '买入' : '卖出'}${item.kind === 'execution' ? '成交' : item.source === 'consensus' ? '最终点' : '意图'} ${item.original_date}${frequency === '日线' ? '' : ` · ${frequency}闭合 ${item.closed_at}`}`).join('；');
  }
  const api = {periodKey,build,describe};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.KHQuantMarkers = api;
})(globalThis);
