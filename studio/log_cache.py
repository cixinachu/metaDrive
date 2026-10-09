"""Incremental JSONL reads for dashboard polling; raw downloads remain complete."""
from collections import OrderedDict
import json
from threading import RLock


class JsonlCache:
    def __init__(self, max_files=16):
        self.max_files=max_files
        self.entries=OrderedDict()
        self.lock=RLock()

    def read(self,path):
        with self.lock:
            try:stat=path.stat()
            except FileNotFoundError:
                self.entries.pop(path,None)
                return []
            entry=self.entries.get(path)
            if entry is None or stat.st_ino!=entry['inode'] or stat.st_size<entry['offset'] or (stat.st_size==entry['offset'] and stat.st_mtime_ns!=entry['mtime']):
                entry=dict(inode=stat.st_ino,offset=0,mtime=0,pending=b'',rows=[])
                self.entries[path]=entry
            if stat.st_size>entry['offset']:
                with path.open('rb') as f:
                    f.seek(entry['offset'])
                    chunk=f.read(stat.st_size-entry['offset'])
                    entry['offset']=f.tell()
                lines=(entry['pending']+chunk).split(b'\n')
                entry['pending']=lines.pop()
                for line in lines:
                    try:entry['rows'].append(json.loads(line))
                    except (ValueError,UnicodeDecodeError):pass
            entry['mtime']=stat.st_mtime_ns
            self.entries.move_to_end(path)
            while len(self.entries)>self.max_files:self.entries.popitem(last=False)
            return entry['rows']


def tail_text(path,limit=6000):
    try:
        with path.open('rb') as f:
            f.seek(0,2)
            f.seek(max(0,f.tell()-limit))
            return f.read().decode('utf-8',errors='replace')
    except FileNotFoundError:return ''
