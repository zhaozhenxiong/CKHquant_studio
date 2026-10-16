/* Display measured execution facts separately from requested devices and cached provenance. */
(() => {
  'use strict';
  const object = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  const count = value => Number.isSafeInteger(value) && value >= 0 ? value : null;
  const number = value => value.toLocaleString('zh-CN', {maximumFractionDigits: 1});
  const deviceName = value => value === 'mps' ? 'MPS' : /^cuda(?::\d+)?$/.test(value) ? value.toUpperCase() : value === 'cpu' ? 'CPU' : null;
  function caption(task) {
    const detail = object(task?.progress_detail), info = object(task?.compute_info);
    const running = ['queued', 'running', 'cancelling'].includes(task?.status);
    const execution = object(running ? detail.prediction_execution || info.prediction_execution : info.prediction_execution || detail.prediction_execution);
    const parts = [], workers = count(running ? detail.cpu_workers ?? info.cpu_workers : info.cpu_workers ?? detail.cpu_workers);
    if (workers && (info.method || task?.spec?.analysis_method || detail.current_phase)) parts.push(`CPU 结构准备 ${workers} 进程`);
    const phase = detail.current_phase || info.current_phase;
    if (running) {
      if (phase === 'source_preparation') parts.push('当前等待结构准备');
      if (phase === 'model_inference') parts.push('当前预测与评价');
    }
    const fresh = count(execution.fresh_calls), reused = count(execution.reused_calls);
    if (fresh !== null) parts.push(`本批新预测请求 ${number(fresh)} 次`);
    if (reused !== null) parts.push(`预测缓存 ${number(reused)} 次`);
    const verified = count(execution.execution_verified_calls);
    const devices = Object.entries(object(execution.actual_devices));
    // Never infer execution from the selected device or a cached result's old resource.
    const validDevices = verified > 0 && devices.length && devices.every(([key, value]) => deviceName(key) && count(value) !== null && value > 0)
      && devices.reduce((sum, [, value]) => sum + value, 0) === verified
      && fresh !== null && verified <= fresh;
    if (validDevices) parts.push(`本批已核验 ${devices.map(([key, value]) => `${deviceName(key)} ${number(value)} 次`).join(' / ')}`);
    else if (fresh > 0) parts.push('本批推理设备待核验');
    const cached = Object.entries(object(execution.cached_provenance_devices)).filter(([key, value]) => deviceName(key) && count(value) !== null && value > 0);
    if (reused > 0 && cached.length) parts.push(`缓存原计算：${cached.map(([key]) => deviceName(key)).join(' / ')}`);
    const finalTiming = info.phase_wall_seconds || object(info.timing).phase_wall_seconds;
    const timing = object(running ? detail.phase_wall_seconds || finalTiming : finalTiming || detail.phase_wall_seconds);
    for (const [key, label] of [['waiting_preparation', '结构等待'], ['evaluation', '预测与评价']]) {
      const seconds = timing[key];
      if (typeof seconds === 'number' && Number.isFinite(seconds) && seconds >= 0) parts.push(`${label} ${number(seconds)} 秒`);
    }
    if (!Object.keys(execution).length && (info.method || task?.spec?.analysis_method)) {
      const requested = deviceName(task?.spec?.device);
      if (requested) parts.push(`请求 ${requested} · 本批设备记录尚未返回`);
    }
    return parts.join(' · ');
  }
  function resourceCaption(result) {
    const resource = object(result?.resource), execution = object(resource.execution);
    const actual = deviceName(resource.actual_device), parts = [];
    if (result?.reused === true) parts.push(`预测缓存${actual ? ` · 原计算 ${actual}` : ''}`);
    else {
      const groups = ['model_parameter_devices', 'tokenizer_parameter_devices', 'input_tensor_devices'];
      const verified = execution.schema === 'kronos_execution_v1' && execution.fresh_inference === true
        && execution.tensor_devices_verified === true && execution.synchronization_completed === true
        && execution.inference_calls === 1 && execution.actual_device === resource.actual_device
        && groups.every(key => Array.isArray(execution[key]) && execution[key].length > 0 && execution[key].every(device => device === resource.actual_device));
      parts.push(verified && actual ? `本次已核验 ${actual} 推理` : '本次推理设备待核验');
    }
    const timing = object(resource.timings);
    for (const [key, label] of [['model_load_seconds', result?.reused ? '原模型加载' : '模型加载'], ['inference_seconds', result?.reused ? '原推理' : '推理']]) {
      const seconds = timing[key];
      if (typeof seconds === 'number' && Number.isFinite(seconds) && seconds >= 0) parts.push(`${label} ${number(seconds)} 秒`);
    }
    if (typeof resource.elapsed_seconds === 'number' && Number.isFinite(resource.elapsed_seconds) && resource.elapsed_seconds >= 0) parts.push(`${result?.reused ? '原计算' : '合计'} ${number(resource.elapsed_seconds)} 秒`);
    return parts.join(' · ');
  }
  globalThis.KHQuantTaskComputeView = {caption, resourceCaption};
})();
