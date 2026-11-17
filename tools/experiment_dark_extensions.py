"""Dark-only image-extension compatibility around the frozen experiment engine."""
import argparse
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install():
    """Include the original split's .JPEG/.jpeg files without changing pixels."""
    source=ROOT/'source'
    sys.path.insert(0,str(source))
    import utils.dataset as dataset
    def image_file(name):
        return Path(name).suffix.lower() in {'.png','.jpg','.jpeg','.bmp'}
    dataset.is_image_file=image_file
    return dataset


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset');parser.add_argument('--datasets',nargs='+')
    parser.add_argument('--mode');parser.add_argument('--init-seed',type=int)
    args,_=parser.parse_known_args()
    assert set(args.datasets or [args.dataset])=={'EUVP-Dark'},'Use this compatibility entry only for EUVP-Dark'
    lock=json.loads((ROOT/'DATA_COMPATIBILITY_LOCK.json').read_text())
    for relative,digest in lock['files'].items():assert sha(ROOT/relative)==digest,relative
    dataset=install()
    manifest=json.loads((ROOT/'DATA_MANIFEST.json').read_text())
    counts={}
    for split,expected in [('train',4440),('val',555),('test',555)]:
        folder=ROOT/'source/UnderWaterDataset/EUVP-Dark'/split
        data=dataset.DatasetFromFolder(str(folder),256,train=split=='train',resize=True)
        inputs=[Path(p).name for p in data.data_filenames];labels=[Path(p).name for p in data.label_filenames]
        assert inputs==labels==manifest['splits']['EUVP-Dark'][split]
        assert len(data)==expected
        counts[split]=expected
    receipt=dict(at=time.time(),version=lock['version'],dataset='EUVP-Dark',mode=args.mode,init_seed=args.init_seed,
                 pid=os.getpid(),argv=list(sys.argv),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 image_counts=counts,train_steps_per_epoch=185,original_engine_sha256=sha(ROOT/'tools/experiment.py'),
                 data_compatibility_lock_sha256=sha(ROOT/'DATA_COMPATIBILITY_LOCK.json'),
                 original_data_manifest_sha256=sha(ROOT/'DATA_MANIFEST.json'),
                 source_pixels_filenames_and_splits_unchanged=True,
                 compatibility='Include .JPEG/.jpeg and other case variants in the already declared Dark split; all transforms/loader/RNG/loss/model/EMA remain the frozen engine.')
    directory=ROOT/'data_compatibility_workers/EUVP-Dark'/f'seed{args.init_seed}'/str(args.mode)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/f'{time.time_ns()}_pid{os.getpid()}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    runpy.run_path(str(ROOT/'tools/experiment.py'),run_name='__main__')


if __name__=='__main__':main()
