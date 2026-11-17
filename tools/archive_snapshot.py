"""Resumable, independent-inode snapshot that excludes active observers."""
from pathlib import Path
import hashlib,json,os,shutil,time

def sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
 return digest.hexdigest()

def excluded(relative):
 parts=relative.parts;name=relative.name
 return ('__pycache__' in parts or name.endswith(('.lock','.tmp','.tmp.xlsx','.copytmp')) or name.startswith('LATEST_') or
  relative.as_posix() in ['logs/progress_observer.log','logs/export_live.log','logs/export_initial.log','logs/export_initial_rerun.log'])

def snapshot(source,destination):
 source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
 manifest_path=destination/'MANIFEST.json'
 if manifest_path.exists():
  manifest=json.loads(manifest_path.read_text())
  for relative,item in manifest['files'].items():
   path=destination/relative;assert path.exists() and sha(path)==item['sha256'],relative
   current=source/relative
   if current.exists():assert sha(current)==item['sha256'] and not path.samefile(current),relative
  return manifest['files']
 files={}
 for current in sorted(source.rglob('*')):
  relative=current.relative_to(source)
  if excluded(relative):continue
  target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
  if current.is_symlink():
   link=os.readlink(current)
   if target.is_symlink():assert os.readlink(target)==link
   elif not target.exists():target.symlink_to(link,target_is_directory=current.is_dir())
   else:raise AssertionError(f'Unexpected non-symlink {target}')
  elif current.is_dir():target.mkdir(exist_ok=True)
  elif current.is_file():
   signature=sha(current)
   if not target.exists() or sha(target)!=signature:
    temp=target.with_name(target.name+'.copytmp')
    if temp.exists():temp.chmod(temp.stat().st_mode|0o200)
    shutil.copy2(current,temp);assert sha(temp)==signature;temp.replace(target)
   assert not current.samefile(target) and sha(target)==signature,relative
   files[str(relative)]=dict(sha256=signature,size_bytes=current.stat().st_size)
 temporary=manifest_path.with_suffix('.json.tmp');temporary.write_text(json.dumps(dict(at=time.time(),files=files,live_observer_artifacts_excluded=True,partial_copy_resumable=True),indent=2)+'\n');temporary.replace(manifest_path)
 return files
