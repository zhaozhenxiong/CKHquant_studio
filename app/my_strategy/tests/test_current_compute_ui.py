"""Current actual compute and device selection UI contracts.

Preserves the shipped JavaScript case archived from research_progress; device
selection and actual work must remain separate from claimed GPU computation.
"""
from pathlib import Path
import shutil
import subprocess
import pytest


def test_shipped_ui_uses_actual_ml_work_and_preserves_explicit_selection():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped JavaScript checks")
    source_path = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const elements=new Map();
function element(id) {
  if (!elements.has(id)) {
    const item={value:'',textContent:'',innerHTML:'',hidden:false,disabled:false};
    Object.defineProperty(item,'options',{get:()=>[...item.innerHTML.matchAll(/<option value="([^"]+)"/g)].map((match)=>({value:match[1]}))});
    elements.set(id,item);
  }
  return elements.get(id);
}
global.document={getElementById:element};
const source=fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);',
  'globalThis.progressTest={state,renderComputeResult,renderScanComputePlan,taskDetail,taskActions,refreshCompute};');
vm.runInThisContext(source);
const {state,renderComputeResult,renderScanComputePlan,taskDetail,taskActions,refreshCompute}=progressTest;
state.scan={};
renderComputeResult('scan',{research:true,selected_device:'cuda:1',cuda_work:true,model_inference_rows:64,
  model_inference_batches:2,cpu_workers:4,cache_hits:63});
let text=element('scan-compute-result').textContent;
assert(text.includes('CUDA 模型推理') && text.includes('64 行') && text.includes('2 个推理批次'));
assert(text.includes('CPU 准备并发上限 4') && text.includes('特征缓存命中 63'));
assert(!text.includes('个准备进程'));
renderComputeResult('scan',{research:true,selected_device:'cuda:1',cuda_work:false,model_inference_rows:0});
text=element('scan-compute-result').textContent;
assert(text.includes('未执行 ML 推理（0 行）') && !text.includes('本次 CUDA'));
renderComputeResult('scan',{research:true,selected_device:'cpu',cuda_work:false,model_inference_rows:4});
assert(element('scan-compute-result').textContent.includes('CPU 模型推理'));
renderComputeResult('scan',{research:true,selected_device:'mps',mps_work:true,model_inference_rows:4});
assert(element('scan-compute-result').textContent.includes('MPS 模型推理'));
renderComputeResult('scan',{research:true,selected_device:'mps',mps_work:false,model_inference_rows:0});
assert(!element('scan-compute-result').textContent.includes('本次 MPS'));
renderComputeResult('scan',{selected_device:'mps',actual_device:'cpu',backend:'numpy',bars:4,mps_work:false});
assert(element('scan-compute-result').textContent.includes('CPU 信号计算'));
assert(element('scan-compute-result').textContent.includes('CPU float64 精度'));
renderComputeResult('scan',{research:true,training:true,selected_device:'mps',mps_work:true,model_training_models:2,model_training_batches:10});
assert(element('scan-compute-result').textContent.includes('MPS 模型训练'));
assert(element('scan-compute-result').textContent.includes('10 个优化批次'));
renderComputeResult('scan',{research:true,selected_device:'cuda:1'});
assert(element('scan-compute-result').textContent.includes('未保存本次 ML 推理统计'));
element('scan-mode').value='research'; element('scan-model').value=''; element('scan-device').value='cuda:1';
element('scan-model-policy').value='auto';
renderScanComputePlan();
assert(element('scan-compute-plan').textContent.includes('按所选日期匹配模型并批量推理'));
assert(element('scan-compute-plan').textContent.includes('缺少合格入场头时明确阻断'));
assert(element('scan-compute-plan').textContent.includes('设备 cuda:1'));
assert(!element('scan-compute-plan').textContent.includes('不执行 GPU ML 推理'));
assert.equal(element('scan-model').value,'');
element('scan-model-policy').value='pinned'; element('scan-model').value='explicit-model'; renderScanComputePlan();
assert.equal(element('scan-model').value,'explicit-model');
assert(element('scan-compute-plan').textContent.includes('CPU 准备结构与量价特征'));
assert(element('scan-compute-plan').textContent.includes('真实推理行数、批次和 GPU 工作以本次结果为准'));
element('scan-mode').value='structure'; renderScanComputePlan();
assert(element('scan-compute-plan').textContent.includes('批量信号运算'));
const running={kind:'scan',job_id:'fixture',status:'running',spec:{research:true,device:'cuda:1'},
  progress_detail:{stage:'CPU并行研究特征准备',current:32,total:64,device:'cpu'}};
text=taskDetail(running);
assert(text.includes('当前阶段 CPU') && text.includes('尚非 GPU 执行证明'));
assert(!text.includes('已执行 CUDA'));
assert(taskActions(running).includes('data-cancel="fixture"'));
running.status='succeeded'; running.compute_info={research:true,selected_device:'cuda:1',cuda_work:true,
  model_inference_rows:64,model_inference_batches:2};
text=taskDetail(running);
assert(text.includes('CUDA 模型推理') && !text.includes('当前阶段'));
global.sessionStorage={getItem:()=>null};
global.fetch=async()=>({ok:true,status:200,text:async()=>JSON.stringify({available:true,requested_device:'mps',selected_device:'mps',
  devices:[{type:'mps',device:'mps',index:null,name:'Apple MPS',total_memory_bytes:null}]})});
refreshCompute('data','mps').then(()=>{
  for (const prefix of ['analysis','scan','backtest','data']) {
    assert.equal(element(`${prefix}-device`).value,'mps');
    assert(element(`${prefix}-device`).innerHTML.includes('value="mps"'));
    assert(!element(`${prefix}-device`).innerHTML.includes('value="cuda:0"'));
  }
  assert(element('compute-devices').textContent.includes('float32 模型训练与推理'));
}).catch((error)=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run([node, "-e", script, str(source_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
