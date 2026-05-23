import argparse
from pathlib import Path
import fsspec

def download(s3path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if s3path.startswith('s3://'):
        fs = fsspec.filesystem('s3', anon=True)
        fname = s3path.split('/')[-1]
        with fs.open(s3path, 'rb') as r, open(outdir/fname, 'wb') as f:
            f.write(r.read())
        print('downloaded', s3path, '->', outdir/fname)
    else:
        # fallback to http
        import requests
        r = requests.get(s3path)
        r.raise_for_status()
        fname = s3path.split('/')[-1]
        with open(outdir/fname, 'wb') as f:
            f.write(r.content)
        print('downloaded', s3path, '->', outdir/fname)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('url')
    p.add_argument('--outdir', default='data')
    args = p.parse_args()
    download(args.url, args.outdir)
