"""Measure enhancer + frozen DINO costs, explicitly recording shared-GPU conditions."""
import fcntl,json,os,subprocess,time
import numpy as np
import torch
from frozen_factory import ROOT,atomic,state_sha

def profile_efficiency(model,mode,number):
 with (ROOT/'efficiency.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  path=ROOT/'EFFICIENCY.json';report=json.loads(path.read_text()) if path.exists() else dict(methods={})
  if mode in report['methods']:return
  before=state_sha(model.state_dict());model.eval();dummy=torch.zeros(1,3,256,256,device='cuda')
  with torch.no_grad():
   for _ in range(10):model(dummy)
   torch.cuda.synchronize()
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:model(dummy)
   torch.cuda.synchronize()
   counted=float(sum(event.flops for event in prof.key_averages()))
   assert counted>0
   times=[]
   for _ in range(1000):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();model(dummy);end.record();end.synchronize();times.append(start.elapsed_time(end))
  trainable=sum(p.numel() for p in model.parameters() if p.requires_grad);restoration=sum(p.numel() for p in model.parameters());teacher=0
  if mode=='dino_cross_stage':
   from dino_stage_interaction import _TEACHERS
   assert _TEACHERS and all(not p.requires_grad for t in _TEACHERS.values() for p in t.parameters())
   teacher=sum(p.numel() for t in _TEACHERS.values() for p in t.parameters())
   assert teacher>10000000
  assert state_sha(model.state_dict())==before
  process_usage=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
  row=dict(at=time.time(),mode=mode,weights_source_initialization=number,trainable_parameters=trainable,restoration_parameters=restoration,frozen_teacher_parameters=teacher,
   total_parameters_M=(restoration+teacher)/1e6,FLOPs_G=counted/1e9,runtime_ms=float(np.mean(times)),runtime_median_ms=float(np.median(times)),runtime_sd_ms=float(np.std(times)),
   input_shape=[1,3,256,256],precision='float32, current accepted TF32 flags',iterations=1000,warmup=10,physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),gpu_name=torch.cuda.get_device_name(0),
   FLOPs_convention='PyTorch1.13 profiler supported conv/matmul forward FLOPs, multiply-add=2. Includes actual frozen DINO invocation and its matrix multiplications. Some elementwise/norm/interpolation operations are not counted. Not identical to paper THOP MAC-as-FLOP convention.',
   runtime_convention='CUDA-event elapsed inference, enhancer and frozen DINO preprocessing/forward included; disk I/O and external IQA metrics excluded.',
   shared_GPU_process_inventory=process_usage,shared_gpu_caveat='Measured on a shared RTX4090 during research. Not a speed claim against paper RTX3090 runtimes.',model_state_unchanged=True)
  report['methods'][mode]=row;atomic(path,report)
  print('EFFICIENCY_MEASURED',mode,row['total_parameters_M'],row['FLOPs_G'],row['runtime_ms'],flush=True)
