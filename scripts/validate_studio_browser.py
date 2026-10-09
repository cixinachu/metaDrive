"""Optional Playwright UI check; requires the local server and Chromium shell."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1600,'height':1100})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8765')
    page.wait_for_selector('[data-scenario="intersection"]')
    page.locator('[data-scenario="intersection"]').click()
    page.wait_for_function("document.querySelector('#pageTitle').textContent.includes('场景预览') && document.querySelector('#state').textContent==='已完成'",timeout=60000)
    page.wait_for_function("document.querySelector('#frame').naturalWidth===800")
    page.screenshot(path=str(ROOT/'outputs/studio_preview_ui.png'))
    page.locator('[data-algorithm="wcsac_iqn"]').click()
    page.evaluate("document.querySelectorAll('details').forEach(x=>x.open=true)")
    for k,v in dict(timesteps=96,horizon=40,num_scenarios=2,learning_starts=16,batch_size=16,
                    hidden_size=32,buffer_size=500,log_freq=10,n_quantiles=8,checkpoint_freq=48).items():
        page.locator('#'+k).fill(str(v))
    page.locator('#device').select_option('cpu')
    page.locator('#start').click()
    page.wait_for_function("document.querySelector('#pageTitle').textContent.includes('训练 · WCSAC-IQN') && document.querySelector('#state').textContent==='已完成'",timeout=90000)
    page.wait_for_selector('.chartHead a')
    with page.expect_download() as download:
        page.locator('.chartHead a').first.click()
    download.value.save_as(str(ROOT/'outputs/studio_browser_curve.csv'))
    assert (ROOT/'outputs/studio_browser_curve.csv').stat().st_size>50
    page.locator('[data-mode="eval"]').click()
    page.locator('#episodes').fill('2');page.locator('#num_scenarios').fill('2')
    page.locator('#horizon').fill('40')
    assert 'wcsac_iqn' in page.locator('#model_id option:checked').inner_text()
    page.locator('#start').click()
    page.wait_for_function("document.querySelector('#pageTitle').textContent.includes('评估 · WCSAC-IQN') && document.querySelector('#state').textContent==='已完成'",timeout=90000)
    page.wait_for_selector('#resultSection:not([hidden])')
    page.screenshot(path=str(ROOT/'outputs/studio_evaluation_ui.png'))
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path=str(ROOT/'outputs/studio_mobile_ui.png'))
    assert not errors,errors
    (ROOT/'outputs/studio_browser_validation.json').write_text(json.dumps(dict(page_errors=errors,
        real_preview=True,ui_started_training=True,ui_started_evaluation=True,chart_download=True,
        desktop_and_mobile_screenshots=True),indent=2))
    print('Browser preview → training → model selection → evaluation → download all passed.')
    browser.close()
