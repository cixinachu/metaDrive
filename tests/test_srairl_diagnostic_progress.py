import io
import threading

from scripts.diagnose_srairl_road_geometry import DiagnosticProgress, collection_progress


def test_heartbeat_is_visible_during_blocking_work_and_stops(monkeypatch, capsys):
    emitted=threading.Event()
    class StatusStream(io.StringIO):
        def write(self,text):
            result=super().write(text)
            if '等待中' in text:emitted.set()
            return result
    stream=StatusStream()
    monkeypatch.setattr('sys.stderr',stream)
    with DiagnosticProgress(.01) as progress:
        progress.update('初始化地图',force=True)
        assert emitted.wait(2), 'no heartbeat while the main thread was waiting'
    assert not progress.thread.is_alive()
    assert '初始化地图' in stream.getvalue()
    assert '等待中' in stream.getvalue()
    assert capsys.readouterr().out==''  # stdout must remain valid report JSON


def test_expert_callback_reports_real_episode_and_step_counters(capsys):
    progress=DiagnosticProgress(.001)
    callback=collection_progress(progress,'merge',20.,10)
    assert callback(dict(episodes=0,step=0))
    progress.last_print=0
    assert callback(dict(episodes=0,episode_step=42))
    assert callback(dict(episodes=1,episode_steps=180,collision=False,out_of_road=False))
    captured=capsys.readouterr()
    assert captured.out==''
    assert '回合 1/10' in captured.err
    assert '仿真步 42' in captured.err
    assert '步数 180' in captured.err
    assert '碰撞=False 越界=False' in captured.err
