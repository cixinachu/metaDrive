"""Run with the llm_drive Python; mocks API traffic and creates no real jobs."""
import base64
import argparse
import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from playwright.sync_api import sync_playwright
from studio.config import DEFAULTS,SCENARIOS,ALGORITHM_NAMES
from env.tasks.srairl_interaction import SCENARIO_SPEED_CONFIG,ROAD_GEOMETRY_CONFIG


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT/'studio/static'),**kwargs)
    def log_message(self,*args):pass
    def do_GET(self):
        if self.path.split('?')[0]=='/run':self.path='/run.html'
        try:super().do_GET()
        except (BrokenPipeError,ConnectionResetError):pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.output:args.output.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    catalog=dict(defaults=DEFAULTS,scenarios=[dict(id=k,name=v[0]) for k,v in SCENARIOS.items()],
        algorithms=ALGORITHM_NAMES,algorithm_metadata={},srairl_scenario_speeds=SCENARIO_SPEED_CONFIG,
        srairl_road_geometry=ROAD_GEOMETRY_CONFIG,srairl_expert_datasets=[],models=[],
        srairl_models=[],reward_guide_models=[],srairl_pipeline_bundles=[])
    settings=dict(operation='expert_collect',scenario='curve',scenarios=['curve'],traffic_density=.05,
        lane_num=3,horizon=800,demo_episodes=25,seed=20000,srairl_view='topdown')
    job=dict(id='audit_train',request=dict(task='srairl',kind='train',settings=settings),
        status=dict(state='completed',step=25,total=25,episodes=25),
        summary=dict(episodes=25,accepted_episodes=23,rejected_episodes=2),log='')
    preview=dict(id='audit_preview',request=dict(task='srairl',kind='preview',settings={}),status=dict(state='completed'))
    posted=[];errors=[]
    def intercept(route):
        path=route.request.url.split('/api/',1)[-1]
        if path=='catalog':data=catalog
        elif path=='jobs' and route.request.method=='GET':data=dict(active=None,jobs=[preview,job])
        elif path=='jobs' and route.request.method=='POST':
            body=route.request.post_data_json;posted.append(body)
            data=dict(id='audit_preview' if body['kind']=='preview' else 'audit_new')
        elif '/frame' in path:
            route.fulfill(status=200,content_type='image/png',body=base64.b64decode(
                'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='));return
        elif '/metrics?' in path:data=dict(offset=0,rows=[])
        elif path.endswith('/videos'):data=dict(videos=[])
        else:data=preview if 'audit_preview' in path else job
        route.fulfill(status=200,content_type='application/json',body=json.dumps(data))
    try:
        with sync_playwright() as p:
            executable=Path.home()/'.cache/ms-playwright/chromium_headless_shell-1161/chrome-linux/headless_shell'
            browser=p.chromium.launch(headless=True,executable_path=str(executable) if executable.exists() else None,
                args=['--no-sandbox'])
            context=browser.new_context(viewport=dict(width=1440,height=1000))
            context.route('**/api/**',intercept)
            page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
            base=f'http://127.0.0.1:{server.server_port}'
            page.goto(base);page.wait_for_function("document.getElementById('history').options.length===2")
            page.select_option('#methodSelect','srairl');page.select_option('#srTaskSelect','srairl_expert')
            assert page.locator('#srairl_traffic_density').is_visible()
            for name in ('safeRlRewardModeField','safeRlCostModeField','safeRlSeedField','safeRlDeviceField',
                    'safeRlRecordingField','srairlEpochsField','srairlRankField','safeRlAlgorithmSection'):
                assert not page.locator('#'+name).is_visible(),name
            assert page.input_value('#srairl_collection_mode')=='single'
            page.select_option('#srairl_scenario','curve');page.fill('#srairl_traffic_density','0.05')
            page.fill('#srairl_demo_episodes','10')
            assert '仅采集弯道' in page.locator('#srairlCollectionSummary').inner_text()
            page.select_option('#srairl_collection_mode','all')
            assert '总计 30' in page.locator('#srairlCollectionSummary').inner_text()
            page.select_option('#history','audit_train')
            page.wait_for_function("document.getElementById('srairl_demo_episodes').value==='25'")
            for name,value in dict(srairl_scenario='curve',srairl_traffic_density='0.05',
                    srairl_collection_mode='single',srairl_seed='20000',srairl_horizon='800').items():
                assert page.input_value('#'+name)==value,name
            page.evaluate('window.scrollTo(0,500)');page.wait_for_timeout(100)
            article=page.locator('article').bounding_box();viewport=page.locator('.viewport').bounding_box()
            assert 0<=article['y']<=25,article
            assert viewport['width']>=article['width']-55,(viewport,article)
            assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
            if args.output:page.screenshot(path=str(args.output/'desktop.png'),full_page=True)
            page.set_viewport_size(dict(width=390,height=844))
            assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
            assert page.locator('article').evaluate("el=>getComputedStyle(el).position")=='static'
            if args.output:page.screenshot(path=str(args.output/'mobile.png'),full_page=True)
            page.select_option('#methodSelect','safe_rl')
            assert page.locator('#reward_mode').is_visible() and page.locator('#device').is_visible()
            # Browser submission must send scope/density and use SR geometry for previews.
            page.select_option('#methodSelect','srairl');page.select_option('#srTaskSelect','srairl_expert')
            page.wait_for_timeout(800);page.wait_for_function('!scenePreviewBusy')
            for scope in ('single','all'):
                page.select_option('#srairl_collection_mode',scope)
                page.evaluate("activeBusy=false;start('train')")
                page.wait_for_timeout(200)
                request=[r for r in posted if r['kind']=='train'][-1]
                assert request['settings']['srairl_traffic_density']==.05
                assert request['settings']['srairl_collect_scenarios']==(['curve'] if scope=='single' else ['straight','curve','merge'])
            previews=[r for r in posted if r['kind']=='preview' and r['task']=='srairl']
            assert previews and previews[-1]['settings']['srairl_seed']==20000
            monitor=context.new_page();monitor.goto(base+'/run?job=audit_train')
            monitor.wait_for_function("document.getElementById('runTitle').textContent.includes('专家示范')")
            assert monitor.locator('article').evaluate("el=>getComputedStyle(el).position")!='sticky'
            assert '合格 23' in monitor.locator('#runMessage').inner_text()
            if args.output:monitor.screenshot(path=str(args.output/'run-monitor.png'),full_page=True)
            assert not errors,errors
            print('PASS: Chromium workflow, submission, SR preview, history restore, desktop/mobile layout, and run monitor')
            browser.close()
    finally:server.shutdown()


if __name__=='__main__':main()
