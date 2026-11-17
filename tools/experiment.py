"""UFO200epoch delayed DropPath/EMA runs with raw/EMA validation and durable recovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import traceback
import torch
from frozen_factory import ROOT, SOURCE, PLAN, original, seed, create_model, state_sha, sha, atomic, snapshot
from dp_ema import masks
from restoration_losses import RestorationLoss
from delayed_ema import DelayedEMA

RESOURCE_POLICY=json.loads((ROOT/'SCHEDULER_POLICY.json').read_text())


_OPERATORS_INSTALLED=False

def configure():
    global _OPERATORS_INSTALLED
    seed()
    if not _OPERATORS_INSTALLED:
        import strict_math
        strict_math.install();_OPERATORS_INSTALLED=True


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def key(dataset, mode, number):
    return f'{dataset}/{number}/{mode}'


def run_path(dataset, mode, number, preflight=False):
    return ROOT / ('preflight' if preflight else 'runs') / dataset / f'seed{number}' / mode


def save_torch(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def copy_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def compare_epoch_inputs(groups):
    assert groups and all(len(v)==len(groups[0]) for v in groups)
    reference=[(v['input_sha'],v['label_sha'],v['mask_sha']) for v in groups[0]]
    assert all([(v['input_sha'],v['label_sha'],v['mask_sha']) for v in rows]==reference for rows in groups[1:]), 'Cross-run inputs/GT/DropPath masks differ'
    return reference


def barrier(dataset,mode,number,epoch):
    # Resource-constrained runs may advance independently. Compare each epoch
    # once all six committed chunks exist; this never changes tensor updates.
    import fcntl
    folder=ROOT/'cohorts'/dataset/f'seed{number}';folder.mkdir(parents=True,exist_ok=True)
    atomic(folder/f'{number}_{mode}.json',dict(epoch=epoch,at=time.time()))
    paths=[run_path(dataset,m,number)/'trace_epochs'/f'epoch{epoch:03d}.json' for m in PLAN['methods']]
    if not all(p.exists() for p in paths):return
    with (folder/'deferred_audit.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        groups=[json.loads(p.read_text())['steps'] for p in paths]
        compared=compare_epoch_inputs(groups)
        assert len(compared)==PLAN['steps_per_epoch'][dataset]
        atomic(folder/f'epoch{epoch:03d}_inputs_audit.json',dict(epoch=epoch,runs=len(groups),steps_per_run=len(compared),all_inputs_GT_and_masks_exact=True,deferred_nonblocking=True))


def validate(model, dataset, collect=False):
    from utils.dataset import get_loader
    from utils.metrics import getPSNR, getSSIM
    import numpy as np
    was_training = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y, names in get_loader(str(SOURCE / 'UnderWaterDataset' / dataset / 'val'), 24, 256,
                train=False, resize=True, num_workers=4, shuffle=False, pin_memory=True):
            output = model(x.cuda())
            assert torch.isfinite(output).all()
            prediction = output.clamp(0, 1).cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
            target = y.numpy().astype(np.float32).transpose(0, 2, 3, 1)
            for i, name in enumerate(names):
                rows.append(dict(name=name, psnr=float(getPSNR(prediction[i:i+1], target[i:i+1], 1.)),
                    ssim=float(getSSIM(prediction[i:i+1], target[i:i+1], 1.))))
    model.train(was_training)
    assert len(rows) == PLAN['split_counts'][dataset]['val']
    return dict(psnr=float(np.mean([v['psnr'] for v in rows])), ssim=float(np.mean([v['ssim'] for v in rows])))


def train(dataset, mode, number, preflight=False, resume=False):
    configure()
    from utils.dataset import get_loader
    dest = run_path(dataset, mode, number, preflight)
    assert not (preflight and resume)
    if not resume:
        dest.mkdir(parents=True, exist_ok=False)
    chunks = dest / 'trace_epochs'
    chunks.mkdir(exist_ok=resume)
    temporary = tempfile.TemporaryDirectory(prefix='dp_ema_trainer_')
    recipe = dict(PLAN['recipe'], loss=PLAN['losses'][mode]['pixel'], dataset=dataset, variant='none' if mode == 'baseline' else 'M2v2R',
        rho=1. if mode == 'baseline' else .25, save_path=temporary.name, run_tag='droppath_ema')
    constructor = original.BaselineUNet
    original.BaselineUNet = lambda **kwargs: create_model(mode, number)
    trainer = original.Trainer(argparse.Namespace(**recipe))
    original.BaselineUNet = constructor
    model = trainer.model
    loss_config=PLAN['losses'][mode]
    trainer.loss_fn=RestorationLoss(loss_config['pixel'],loss_config['ssim_weight'])
    ema = DelayedEMA(model, PLAN['ema']['start_optimizer_step_by_dataset'][dataset], PLAN['ema']['decay_constant'])
    seed()
    initial = state_sha(model.state_dict())
    expected = json.loads((ROOT / 'INITIALIZATIONS.json').read_text())['models'][f'{number}/{mode}']
    assert initial == expected['full_sha'] and state_sha(ema.model.state_dict()) == initial
    source_hashes = snapshot()
    helpers = {n: sha(ROOT / 'tools' / n) for n in ['experiment.py', 'frozen_factory.py', 'strict_math.py', 'base_equivalent_operators.py', 'delayed_ema.py','restoration_losses.py']}
    metadata = dict(execution_revision='four_cumulative_stages_v1', dataset=dataset, mode=mode, initialization_seed=number, training_seed=7,
        loss_config=loss_config, model_mode=mode, epochs=PLAN['epochs'], recipe={k:v for k,v in recipe.items() if k not in ['save_path', 'run_tag']},
        initial_full_sha=initial, common_backbone_sha=expected['common_backbone_sha'],
        trainable_parameters=expected['trainable_parameters'], drop_path=PLAN['dropout'], ema=PLAN['ema'],
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES']), torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), backend=PLAN['backend'],
        source_sha256=source_hashes, tool_sha256=helpers, preflight_only=preflight,
        teacher_weights_sha256=json.loads((ROOT/'TEACHER_PROVENANCE.json').read_text())['sha256'],
        primary_checkpoint='EMA validation-best', restoration_pretrained=False,
        actual_resource_engine=Path(__file__).name,resource_engine_sha256=sha(Path(__file__)),
        physical_gpu_name=torch.cuda.get_device_name(0),compute_capability=list(torch.cuda.get_device_capability(0)),
        resource_policy_sha256=sha(ROOT/'SCHEDULER_POLICY.json'))
    attempts=dest/'execution_attempts';attempts.mkdir(exist_ok=True)
    atomic(attempts/f'{time.time_ns()}_pid{os.getpid()}.json',metadata)
    train_gen = torch.Generator().manual_seed(1007)
    loader = get_loader(str(SOURCE / 'UnderWaterDataset' / dataset / 'train'), 24, 256,
        train=True, resize=True, num_workers=8, shuffle=True, pin_memory=True,
        generator=train_gen, worker_init_fn=original._worker_init)
    per_epoch = PLAN['steps_per_epoch'][dataset]
    assert len(loader) == per_epoch
    reference_path = ROOT / 'references' / f'{dataset}_seed7_TRACE.json'
    data_reference = json.loads(reference_path.read_text())['steps'] if reference_path.exists() else []
    assert not data_reference or len(data_reference) == per_epoch * PLAN['epochs']
    control_path=ROOT/'references/historical'/dataset/f'seed{number}/isp_control/TRACE.json'
    old_control=json.loads(control_path.read_text())['steps'] if mode=='isp_control' and control_path.exists() else None
    rows, records, start_epoch = [], [], 0
    best_ema = best_raw = -float('inf')
    best_ema_state = best_raw_state = None
    best_ema_record = best_raw_record = None
    if resume:
        recovery = json.loads((ROOT / 'RECOVERIES.json').read_text())[key(dataset, mode, number)]
        checkpoint_path = ROOT / recovery['checkpoint_relative']
        assert sha(checkpoint_path) == recovery['checkpoint_sha256']
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        previous = json.loads((dest / 'run_config.json').read_text())
        assert previous['tool_sha256'] == helpers and previous['source_sha256'] == source_hashes
        assert metadata['gpu'] in RESOURCE_POLICY['authorized_gpus']
        assert previous['backend']==metadata['backend']
        assert torch.cuda.get_device_name(0)==RESOURCE_POLICY['hardware_requirement']['name']
        assert tuple(torch.cuda.get_device_capability(0))==(8,9)
        model.load_state_dict(checkpoint['model'], strict=True)
        ema.load_state_dict(checkpoint['ema'])
        trainer.optim.load_state_dict(checkpoint['optim'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = int(checkpoint['epoch'])
        for name, signature in sorted(checkpoint['trace_chunks_sha256'].items()):
            assert sha(chunks / name) == signature
            rows.extend(json.loads((chunks / name).read_text())['steps'])
        assert len(rows) == start_epoch * per_epoch == ema.num_updates
        records = checkpoint['records']
        best_ema, best_raw = checkpoint['best_ema'], checkpoint['best_raw']
        best_ema_state, best_raw_state = checkpoint['best_ema_state'], checkpoint['best_raw_state']
        best_ema_record, best_raw_record = checkpoint['best_ema_record'], checkpoint['best_raw_record']
        if best_ema_state is not None:
            save_torch(best_ema_state, dest / 'best_model.pth')
            atomic(dest / 'best_ema.json', best_ema_record)
        save_torch(best_raw_state, dest / 'best_raw_model.pth')
        atomic(dest / 'best_raw.json', best_raw_record)
        atomic(dest / 'records.json', records)
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        random.setstate(checkpoint['python_rng'])
        import numpy as np
        np.random.set_state(checkpoint['numpy_rng'])
        train_gen.set_state(checkpoint['train_generator_rng'])
        atomic(dest / f'RESUME_ATTEMPT_{recovery["attempt"]}.json', dict(at=time.time(),
            restored_raw_ema_optimizer_scheduler_all_rng=True, from_epoch=start_epoch,
            prefix_steps=len(rows), checkpoint_sha256=recovery['checkpoint_sha256'],
            previous_gpu=previous['gpu'],actual_resume_gpu=metadata['gpu'],resource_engine_sha256=sha(Path(__file__))))
        barrier(dataset, mode, number, start_epoch)
    else:
        atomic(dest / 'run_config.json', metadata)
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    error = None
    try:
        for epoch in range(start_epoch + 1, 2 if preflight else PLAN['epochs']+1):
            model.train()
            epoch_loss = gradient_norm_sum = 0.
            part_sums=dict(pixel=0.,ssim_loss=0.,weighted_ssim=0.);ssim_count=0
            rate = trainer.optim.param_groups[0]['lr']
            for x, y, names in loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                prediction = model(x)
                assert torch.isfinite(prediction).all()
                loss = trainer.loss_fn(prediction, y)
                assert torch.isfinite(loss)
                trainer.optim.zero_grad(set_to_none=True)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in parameters)
                gradient = torch.cat([p.grad.flatten() for _, p in parameters])
                row = dict(step=len(rows) + 1, input_sha=tensor_sha(x), label_sha=tensor_sha(y),
                    prediction_sha=tensor_sha(prediction), gradient_sha=tensor_sha(gradient),
                    mask_sha=tensor_sha(masks(model)), loss=float(loss))
                gradient_norm_sum += float(gradient.norm())
                trainer.optim.step()
                decay = ema.update(model)
                row['updated_sha'] = tensor_sha(torch.cat([p.flatten() for _, p in parameters]))
                row['ema_sha'] = tensor_sha(torch.cat([p.flatten() for p in ema.model.parameters()])) if ema.initialized else None
                if len(rows)<len(data_reference):
                    reference = data_reference[len(rows)]
                    assert row['input_sha'] == reference['input_sha'] and row['label_sha'] == reference['label_sha'] and row['mask_sha'] == reference['mask_sha'], 'Historical seed7 input/augmentation changed'
                parts=trainer.loss_fn.component_values()
                row['loss_components']=parts
                for name,value in parts.items():
                    if value is not None:part_sums[name]+=value
                ssim_count+=int(parts['ssim_loss'] is not None)
                if old_control is not None:
                    previous=old_control[len(rows)]
                    assert all(row[name]==previous[name] for name in ['step','input_sha','label_sha','prediction_sha','gradient_sha','mask_sha','loss','updated_sha','ema_sha']), 'Fixed Charb+.1SSIM ISP differs from historical200epoch trajectory'
                rows.append(row)
                epoch_loss += row['loss']
            assert len(rows) == epoch * per_epoch and ema.num_updates == len(rows)
            if epoch == 1 and not preflight:
                reference = json.loads((run_path(dataset, mode, number, True) / 'TRACE.json').read_text())['steps']
                assert rows == reference, 'DropPath/raw/EMA trajectory differs from own preflight'
            atomic(chunks / f'epoch{epoch:03d}.json', dict(steps=rows[-per_epoch:]))
            raw_metrics = validate(model, dataset)
            ema_metrics = validate(ema.model, dataset) if ema.initialized else None
            record = dict(epoch=epoch, train_loss=epoch_loss/per_epoch, gradient_norm=gradient_norm_sum/per_epoch,
                learning_rate=rate, ema_decay=decay, raw=raw_metrics, ema=ema_metrics,
                loss_components=dict(pixel=part_sums['pixel']/per_epoch,ssim_loss=part_sums['ssim_loss']/ssim_count if ssim_count else None,weighted_ssim=part_sums['weighted_ssim']/per_epoch))
            records.append(record)
            atomic(dest / 'records.json', records)
            if ema_metrics is not None and ema_metrics['psnr'] > best_ema:
                best_ema = ema_metrics['psnr']
                best_ema_state = copy_state(ema.model)
                best_ema_record = dict(epoch=epoch, **ema_metrics)
                save_torch(best_ema_state, dest / 'best_model.pth')
                atomic(dest / 'best_ema.json', best_ema_record)
            if raw_metrics['psnr'] > best_raw:
                best_raw = raw_metrics['psnr']
                best_raw_state = copy_state(model)
                best_raw_record = dict(epoch=epoch, **raw_metrics)
                save_torch(best_raw_state, dest / 'best_raw_model.pth')
                atomic(dest / 'best_raw.json', best_raw_record)
            trainer.scheduler.step()
            import numpy as np
            full = dict(model=model.state_dict(), ema=ema.state_dict(), optim=trainer.optim.state_dict(),
                scheduler=trainer.scheduler.state_dict(), epoch=epoch, args=metadata, records=records,
                best_ema=best_ema, best_raw=best_raw, best_ema_state=best_ema_state, best_raw_state=best_raw_state,
                best_ema_record=best_ema_record, best_raw_record=best_raw_record,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(), numpy_rng=np.random.get_state(), train_generator_rng=train_gen.get_state(),
                trace_steps=len(rows), trace_chunks_sha256={f'epoch{e:03d}.json':sha(chunks/f'epoch{e:03d}.json') for e in range(1,epoch+1)})
            save_torch(full, dest / 'last_full.pth')
            if not preflight and epoch in PLAN['milestone_epochs']:
                assert ema.initialized
                stage=dest/'milestones';stage.mkdir(exist_ok=True)
                save_torch(copy_state(model),stage/f'epoch{epoch:03d}_raw_model.pth')
                save_torch(copy_state(ema.model),stage/f'epoch{epoch:03d}_ema_model.pth')
            atomic(dest / 'PROGRESS.json', dict(at=time.time(), epoch=epoch, steps=len(rows),
                seconds=time.time()-started, resumed_from_epoch=start_epoch, raw_validation=raw_metrics, ema_validation=ema_metrics))
            print(dataset, mode, number, 'epoch', epoch, 'loss', record['train_loss'],
                'raw_val', raw_metrics['psnr'], 'ema_val', ema_metrics['psnr'] if ema_metrics else None, flush=True)
            if not preflight:
                barrier(dataset, mode, number, epoch)
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        atomic(dest / 'TRACE.json', dict(steps=rows, error=error, initial_full_sha=initial,
            peak_reserved=torch.cuda.max_memory_reserved(), seconds=time.time()-started))
    assert len(rows) == per_epoch * (1 if preflight else PLAN['epochs'])
    assert snapshot() == source_hashes and helpers == {n:sha(ROOT/'tools'/n) for n in helpers}
    assert all(int(s['step']) == len(rows) for s in trainer.optim.state.values())
    assert ema.num_updates == len(rows) and trainer.scheduler.T_max == PLAN['epochs']
    if not preflight:
        save_torch(copy_state(model), dest/'last_raw_model.pth')
        save_torch(copy_state(ema.model), dest/'last_ema_model.pth')
    atomic(dest / 'completed.json', dict(at=time.time(), epoch=1 if preflight else PLAN['epochs'],
        steps=len(rows), ema_updates=ema.num_updates, preflight_only=preflight,
        first_epoch_matches_preflight=not preflight, source_and_tools_unchanged=True,
        historical_prefix_steps_checked=min(len(rows),len(data_reference)),
        all_available_historical_prefix_inputs_match=True,
        cross_run_audit='deferred: consult cohort audit files and FINAL_AUDIT',
        ema_initialized=ema.initialized,ema_averaging_updates=ema.averaging_updates,ema_start_step=ema.start_step,
        peak_reserved=torch.cuda.max_memory_reserved(), resumed_from_epoch=start_epoch,
        best_ema_sha256=sha(dest/'best_model.pth') if ema.initialized else None, best_raw_sha256=sha(dest/'best_raw_model.pth'),
        last_full_sha256=sha(dest/'last_full.pth')))
    temporary.cleanup()


def evaluate(dataset, mode, number, split, weights, configured=False):
    if not configured:
        configure()
    import numpy as np
    from utils.dataset import get_loader
    from utils.metrics import getPSNR, getSSIM
    dest = run_path(dataset, mode, number)
    assert (dest/'completed.json').exists()
    if split=='test':
        selected=json.loads((ROOT/'METHOD_SELECTION.json').read_text())
        assert selected['locked_before_new_Test'] and mode in selected['test_modes']
    checkpoint = dest / ('best_model.pth' if weights == 'ema' else 'best_raw_model.pth')
    signature = sha(checkpoint)
    locked = json.loads((ROOT/'WEIGHTS_LOCK.json').read_text())[key(dataset,mode,number)][weights]
    assert locked == signature
    output = dest / f'{weights}_{split}_per_image.json'
    assert not output.exists()
    model = create_model(mode, number, 'cuda').eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu'),strict=True)
    rows = []
    input_hash, label_hash = hashlib.sha256(), hashlib.sha256()
    with torch.no_grad():
        for x,y,names in get_loader(str(SOURCE/'UnderWaterDataset'/dataset/split),24,256,
                train=False,resize=True,num_workers=4,shuffle=False,pin_memory=True):
            input_hash.update(x.numpy().tobytes());label_hash.update(y.numpy().tobytes())
            raw = model(x.cuda());assert torch.isfinite(raw).all()
            prediction=raw.clamp(0,1).cpu().numpy().astype(np.float32).transpose(0,2,3,1)
            target=y.numpy().astype(np.float32).transpose(0,2,3,1)
            for i,name in enumerate(names):
                rows.append(dict(name=name,psnr=float(getPSNR(prediction[i:i+1],target[i:i+1],1.)),
                    ssim=float(getSSIM(prediction[i:i+1],target[i:i+1],1.))))
    assert len(rows)==PLAN['split_counts'][dataset][split] and sha(checkpoint)==signature
    atomic(output,dict(dataset=dataset,mode=mode,initialization_seed=number,weights=weights,split=split,
        n=len(rows),psnr=float(np.mean([v['psnr'] for v in rows])),ssim=float(np.mean([v['ssim'] for v in rows])),
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),checkpoint_sha256=signature,input_sha256=input_hash.hexdigest(),target_sha256=label_hash.hexdigest(),images=rows))
    print('EVALUATED',dataset,mode,number,weights,split,flush=True)


def evaluate_completed(dataset,mode,number):
    # All four architectures are fixed before this new batch. Lock each full-Val
    # best before its Test; completed runs need not wait for other datasets.
    from select_method import select
    select()
    configure()
    for weight in ['raw','ema']:
        for split in ['val','test']:
            if not (run_path(dataset,mode,number)/f'{weight}_{split}_per_image.json').exists():
                evaluate(dataset,mode,number,split,weight,configured=True)
    atomic(run_path(dataset,mode,number)/'EVALUATED.json',dict(at=time.time(),splits=4,Test_authorized=True,immediate_after_training=True))
    try:
        import subprocess
        subprocess.run(['/data1/anaconda3/bin/python3',str(ROOT/'tools/export_excel.py')],check=True)
    except Exception as exc:
        atomic(ROOT/'EXCEL_REFRESH_PENDING.json',dict(at=time.time(),reason=repr(exc)))


if __name__=='__main__':
    activation=json.loads((ROOT/'ACTIVATION.json').read_text())
    assert activation['authorized_after_complete_first_batch_review']
    assert (Path(PLAN['first_batch_root'])/'COMPLETE.json').exists()
    owner=json.loads(Path('/data1/wza/wwe/gpu24_dispatch_20261104/CURRENT_WWE_OWNER.json').read_text())
    assert owner['root']==PLAN['coordinator_root'], 'No independent admission outside the global three-card controller'

    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=PLAN['datasets']+PLAN['external_datasets'])
    parser.add_argument('--datasets',nargs='+',choices=PLAN['datasets']+PLAN['external_datasets'])
    parser.add_argument('--mode',choices=PLAN['methods'],required=True)
    parser.add_argument('--init-seed',type=int,choices=[7,43],required=True)
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--pipeline',action='store_true')
    parser.add_argument('--evaluate',choices=['val','test','all','val_only','external']);parser.add_argument('--weights',choices=['raw','ema'],default='ema')
    args=parser.parse_args();assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    assert int(os.environ['CUDA_VISIBLE_DEVICES']) in RESOURCE_POLICY['authorized_gpus']
    assert torch.cuda.get_device_name(0)==RESOURCE_POLICY['hardware_requirement']['name']
    assert tuple(torch.cuda.get_device_capability(0))==(8,9)
    if args.evaluate=='external':
        from external_evaluate import evaluate_external
        evaluate_external(args.dataset,args.mode,args.init_seed)
    elif args.pipeline:
        assert args.datasets and not args.evaluate
        (ROOT/'workers').mkdir(exist_ok=True)
        status=ROOT/'workers'/f'seed{args.init_seed}_{args.mode}.json'
        for dataset in args.datasets:
            destination=run_path(dataset,args.mode,args.init_seed)
            if (destination/'completed.json').exists():
                evaluate_completed(dataset,args.mode,args.init_seed);continue
            atomic(status,dict(at=time.time(),pid=os.getpid(),dataset=dataset,mode=args.mode,init_seed=args.init_seed,phase='preflight'))
            if not (run_path(dataset,args.mode,args.init_seed,True)/'completed.json').exists():train(dataset,args.mode,args.init_seed,preflight=True)
            atomic(status,dict(at=time.time(),pid=os.getpid(),dataset=dataset,mode=args.mode,init_seed=args.init_seed,phase='training'))
            train(dataset,args.mode,args.init_seed,resume=destination.exists())
            atomic(status,dict(at=time.time(),pid=os.getpid(),dataset=dataset,mode=args.mode,init_seed=args.init_seed,phase='evaluation'))
            evaluate_completed(dataset,args.mode,args.init_seed)
        atomic(status,dict(at=time.time(),pid=os.getpid(),mode=args.mode,init_seed=args.init_seed,phase='complete',datasets=args.datasets))
    elif args.evaluate in ['all','val_only']:
        assert args.dataset
        try:
            configure()
            for weight in ['ema','raw']:
                for split in (['val','test'] if args.evaluate=='all' else ['val']):
                    if not (run_path(args.dataset,args.mode,args.init_seed)/f'{weight}_{split}_per_image.json').exists():
                        evaluate(args.dataset,args.mode,args.init_seed,split,weight,configured=True)
            atomic(run_path(args.dataset,args.mode,args.init_seed)/'EVALUATED.json',dict(at=time.time(),splits=4 if args.evaluate=='all' else 2,Test_authorized=args.evaluate=='all'))
        except BaseException as exc:
            atomic(run_path(args.dataset,args.mode,args.init_seed)/'EVALUATION_ERROR.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            raise
    elif args.evaluate:evaluate(args.dataset,args.mode,args.init_seed,args.evaluate,args.weights)
    else:
        assert args.dataset
        train(args.dataset,args.mode,args.init_seed,args.preflight,args.resume)
