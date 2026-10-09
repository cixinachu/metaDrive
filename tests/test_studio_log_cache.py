import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from studio.log_cache import JsonlCache,tail_text

class LogCacheTests(unittest.TestCase):
    def test_append_partial_truncate_and_no_reread(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'metrics.jsonl';c=JsonlCache()
            p.write_bytes(b'{"step":1}\n{"step":')
            self.assertEqual(c.read(p),[{'step':1}])
            with patch.object(Path,'open',side_effect=AssertionError('Unchanged log reread')):
                self.assertEqual(c.read(p),[{'step':1}])
            with p.open('ab') as f:f.write(b'2}\n')
            self.assertEqual(c.read(p),[{'step':1},{'step':2}])
            p.write_bytes(b'{"step":9}\n')
            self.assertEqual(c.read(p),[{'step':9}])
            p.unlink();self.assertEqual(c.read(p),[])

    def test_bounded_cache_and_tail(self):
        with tempfile.TemporaryDirectory() as d:
            c=JsonlCache(max_files=2)
            for i in range(3):
                p=Path(d)/str(i);p.write_text('{"a":1}\n');c.read(p)
            self.assertEqual(len(c.entries),2)
            p.write_text('x'*10000+'last line')
            self.assertEqual(tail_text(p,9),'last line')

if __name__=='__main__':unittest.main()
