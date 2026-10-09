"""Download every row in the official NGSIM table with resumable, ordered pages."""
import concurrent.futures,csv,hashlib,json,time,io,shutil,threading
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
API='https://data.transportation.gov/resource/8ect-6jqj'
def main():
    folder=ROOT/'datasets/raw/ngsim/full';folder.mkdir(parents=True,exist_ok=True)
    pages=folder/'pages';pages.mkdir(exist_ok=True)
    response=requests.get(API+'.json',params={'$select':'location,count(*) as n','$group':'location'},timeout=120);response.raise_for_status()
    expected={r['location']:int(r['n']) for r in response.json()}
    (folder/'inventory.json').write_text(json.dumps(dict(url=response.url,counts=expected),indent=2),encoding='utf-8')
    cancelled=threading.Event()
    size=100000
    def fetch(job):
        if cancelled.is_set():raise RuntimeError('Download cancelled after network failure; rerun to resume')
        loc,offset,n=job;path=pages/f'{loc}_{offset:08d}.csv';meta=path.with_suffix('.json')
        if path.exists() and meta.exists():
            m=json.loads(meta.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest()==m['sha256'] and m['rows']==min(size,n-offset):return m
        params={'$where':f"location='{loc}'",'$order':':id','$limit':size,'$offset':offset}
        for attempt in range(3):
            try:
                r=requests.get(API+'.csv',params=params,timeout=(15,60));r.raise_for_status()
                data=r.content;count=sum(1 for _ in csv.reader(io.StringIO(data.decode('utf-8-sig'))))-1
                if count!=min(size,n-offset):raise ValueError(f'Wrong page size {count}')
                path.with_suffix('.partial').write_bytes(data);path.with_suffix('.partial').replace(path)
                m=dict(path=path.name,url=r.url,rows=count,sha256=hashlib.sha256(data).hexdigest())
                meta.write_text(json.dumps(m,indent=2),encoding='utf-8');print(f'{loc} {offset}: {count} rows',flush=True);return m
            except Exception as exc:
                print(f'retry {loc}:{offset}: {exc}',flush=True)
                if attempt==2:cancelled.set();raise
                time.sleep(min(2**attempt,30))
    jobs=[(loc,o,n) for loc,n in sorted(expected.items()) for o in range(0,n,size)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(fetch,jobs))
    path=folder/'trajectories.csv';tmp=path.with_suffix('.partial')
    with tmp.open('wb') as out:
        for i,m in enumerate(records):
            with (pages/m['path']).open('rb') as f:
                header=f.readline()
                if i==0:out.write(header)
                shutil.copyfileobj(f,out)
    tmp.replace(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    manifest=dict(url=API+'.csv',count_query=response.url,retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),rows=sum(m['rows'] for m in records),counts=expected,sha256=h.hexdigest(),bytes=path.stat().st_size,pages=records,scope='Complete official trajectory table; excludes videos and GIS attachments')
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('COMPLETE',manifest['rows'],flush=True)
if __name__=='__main__':main()
