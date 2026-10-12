/* Full causal price/volume events, including pending tests, on every display period. */
((root) => {
  'use strict';
  const day = (value) => String(value || '').slice(0,10);
  const names = new Set(['Spring','Test','SOS','LPS','Upthrust','SOW','LPSY']);
  const periodKey = (value, frequency) => {
    if (root.KHQuantMarkers?.periodKey) return root.KHQuantMarkers.periodKey(value,frequency);
    const text = day(value);
    if (frequency === '日线') return text;
    if (frequency === '月线') return text.slice(0,7);
    const date = new Date(`${text}T00:00:00Z`);
    date.setUTCDate(date.getUTCDate() + 4 - (date.getUTCDay() || 7));
    const year = date.getUTCFullYear();
    return `${year}-${String(Math.ceil(((date - Date.UTC(year,0,1)) / 86400000 + 1) / 7)).padStart(2,'0')}`;
  };
  function buildDetailed({events = [], bars = [], frequency = '日线', asOf = '', start = '', symbol = ''}) {
    const periods = new Map(bars.map((bar) => [periodKey(bar.time,frequency),bar]));
    const grouped = new Map(), seen = new Set();
    let eventCount = 0, formingProjectedCount = 0;
    for (const event of events) {
      const name = event.event || event.event_type;
      const observedAt = event.origin_observed_at || event.observed_at;
      const firstAvailableAt = event.first_available_at || event.available_at;
      const observed = day(observedAt), available = day(firstAvailableAt);
      const observedTime = Date.parse(observedAt), availableTime = Date.parse(firstAvailableAt);
      if (!names.has(name) || !observed || !available || observed > available || event.evidence_visible === false
          || Number.isFinite(observedTime) && Number.isFinite(availableTime) && observedTime > availableTime
          || (symbol && event.symbol && event.symbol !== symbol) || (start && available < start)
          || (asOf && available > day(asOf))) continue;
      const bar = periods.get(periodKey(available,frequency)), time = day(bar?.time);
      if (!bar || time < available || (asOf && time > day(asOf))) continue;
      const identity = `${event.symbol || symbol}:${event.event_id || [name,observedAt,firstAvailableAt,event.direction].join(':')}`;
      if (seen.has(identity)) continue;
      seen.add(identity);
      const direction = String(event.direction || event.side || '').toUpperCase();
      const supply = direction === 'SELL' || direction === 'SUPPLY' || ['Upthrust','SOW','LPSY'].includes(name);
      const confirmed = event.confirmed === true || event.is_confirmed === true
        || event.confirmed == null && event.is_confirmed == null && !/pending|await|test_required/i.test(event.status || '') && name !== 'SOS';
      const side = supply ? 'SELL' : 'BUY', status = confirmed ? 'confirmed' : 'pending';
      const id = ['khq-wyckoff',event.symbol || symbol,frequency,periodKey(available,frequency),name,side,status].map(encodeURIComponent).join(':');
      if (!grouped.has(id)) grouped.set(id,{id,group_id:id,time,bar_date:time,kind:'wyckoff',action:side,side,event_type:name,
        confirmed,status,sources:['wyckoff'],visible:true,daily_confirmed:frequency !== '日线',
        period_status:bar.is_closed === false ? 'forming' : 'closed',closed_at:bar.is_closed === false ? null : time,entries:[]});
      const group = grouped.get(id);
      const row = {...event,source:'wyckoff',event:name,event_type:name,evidence_date:available,first_available_at:firstAvailableAt,
        origin_observed_at:observedAt,original_event_date:observed,confirmed};
      group.entries.push({row,event_id:event.event_id || null,event_date:available,original_date:observed,
        first_available_at:firstAvailableAt,bar_date:time,source:'wyckoff',displayed:true,excluded:false});
      eventCount++;
      if (frequency !== '日线' && bar.is_closed === false) formingProjectedCount++;
    }
    const groups = [...grouped.values()].sort((a,b) => a.time.localeCompare(b.time) || a.id.localeCompare(b.id));
    for (const group of groups) {
      group.entries.sort((a,b) => a.event_date.localeCompare(b.event_date) || String(a.event_id).localeCompare(String(b.event_id)));
      group.event_count = group.entries.length;
      group.events = group.entries.map((entry) => entry.row);
      group.first_available_at = group.entries.map((entry) => entry.first_available_at).sort()[0];
    }
    const markers = groups.map((group) => ({id:group.id,time:group.time,position:group.action === 'SELL' ? 'aboveBar' : 'belowBar',
      color:group.action === 'SELL' ? '#3b82f6' : '#d97706',shape:'square',size:.55,
      text:`W ${group.event_type}${group.confirmed ? '确认' : '待测试'}${group.event_count > 1 ? ` ×${group.event_count}` : ''}`}));
    return {markers,groups,eventCount,formingProjectedCount};
  }
  const build = (options) => buildDetailed(options).markers;
  const api = {build,buildDetailed};
  root.KHQuantWyckoffMarkers = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
