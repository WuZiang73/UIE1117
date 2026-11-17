"""Recover the exact frozen teacher from Meta's original public checkpoint."""
from pathlib import Path
import hashlib
import urllib.request

ROOT=Path(__file__).resolve().parent
URL='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
SHA='0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    target=ROOT/'teachers/vitb14.pth';target.parent.mkdir(exist_ok=True)
    if target.exists():
        assert sha(target)==SHA,'Existing teacher differs; preserve it and diagnose'
        print('TEACHER_ALREADY_VERIFIED');return
    part=target.with_suffix('.pth.part')
    with urllib.request.urlopen(URL,timeout=60) as response,part.open('xb') as stream:
        for block in iter(lambda:response.read(8*1024**2),b''):stream.write(block)
    assert sha(part)==SHA,'Downloaded teacher differs; partial file retained'
    part.rename(target)
    print('TEACHER_DOWNLOADED_SHA256_VERIFIED')

if __name__=='__main__':main()
