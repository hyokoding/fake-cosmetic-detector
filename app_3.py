"""
가품 탐지 Streamlit 앱 (멀티 플랫폼)
========================================
지원 플랫폼: 11번가 (requests) / SSG (playwright) / 롯데온 (selenium)
"""

import time, json, re, warnings
import numpy as np
import pandas as pd
import cv2, requests, joblib
import streamlit as st

warnings.filterwarnings('ignore')

MODEL_PATH      = 'model_xgb_pseudo.pkl'
NORM_STATS_PATH = 'norm_stats.json'
OFFICIAL_PATH   = 'official_image_quality.csv'
BRAND_ENCODER   = {'Estee_Lauder': 0, 'Jo_Malone': 1, 'Kiehls': 2}
IMG_SIZE        = (256, 256)

FEATURES = ['price_rate','store_like_count_norm','brand_name',
            'delivery_overseas','store_rank',
            'lap_main','sale_brisque','laplacian_ratio','brisque_diff',
            'resolution','lap_change_pct',
            'platform_11번가','platform_Gmarket','platform_lotteon',
            'platform_smartstore','platform_ssg']

OFFICIAL_PRICES = {
    ('Jo_Malone',30):85000, ('Jo_Malone',100):195000, ('Jo_Malone',150):260000,
    ('Kiehls',28):29000, ('Kiehls',50):44000, ('Kiehls',125):72000, ('Kiehls',150):84000,
    ('Estee_Lauder',7):38000, ('Estee_Lauder',15):62000,
    ('Estee_Lauder',30):98000, ('Estee_Lauder',50):145000,
}

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_norm_stats():
    with open(NORM_STATS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def load_official_quality():
    df = pd.read_csv(OFFICIAL_PATH, encoding='utf-8-sig')
    return {(row['brand'], int(row['volume'])): {
        'official_laplacian': row['official_laplacian'],
        'official_brisque':   row['official_brisque'],
    } for _, row in df.iterrows()}

# ── 이미지 처리 ──────────────────────────────────────────────
def download_image(url, referer='https://www.11st.co.kr/', max_retries=2):
    if not isinstance(url, str) or not url.strip(): return None
    headers = {'Accept':'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
               'Accept-Language':'ko-KR,ko;q=0.9', 'Referer':referer}
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except:
            if attempt < max_retries: time.sleep(1)
    return None

def calc_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def calc_brisque(img):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    resized = cv2.resize(gray, IMG_SIZE)
    mu      = cv2.GaussianBlur(resized, (7,7), 1.166)
    mu_sq   = cv2.GaussianBlur(resized**2, (7,7), 1.166)
    sigma   = np.sqrt(np.abs(mu_sq - mu**2))
    mscn    = (resized - mu) / (sigma + 1.0)
    flat    = mscn.flatten()
    skew    = float(np.mean(flat**3) / (np.std(flat)**3 + 1e-6))
    kurt    = float(np.mean(flat**4) / (np.std(flat)**4 + 1e-6))
    return round(abs(skew)*10 + abs(kurt-3)*5, 4)

def est_lap_600(lap, w, h, t=600):
    if any(pd.isna(v) or v==0 for v in [lap,w,h]): return np.nan
    sw, sh = t/w, t/h
    return lap*(sw**2)*(sh**2) if (sw<=1 and sh<=1) else lap*(sw*sh)*0.6

# ── 11번가 크롤러 ─────────────────────────────────────────────
def crawl_11st(url):
    try:
        headers = {
            'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                         'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language':'ko-KR,ko;q=0.9',
            'Referer':'https://www.11st.co.kr/',
        }
        resp = requests.get(url, headers=headers, timeout=15)
        html = resp.text
        name_m  = re.search(r'"name"\s*:\s*"([^"]+)"', html)
        price_m = re.search(r'"price"\s*:\s*(\d+)', html)
        img_m   = re.search(r'"image"\s*:\s*"([^"]+)"', html)
        return {
            'product_name':      name_m.group(1) if name_m else '',
            'price':             int(price_m.group(1)) if price_m else None,
            'image_url':         img_m.group(1) if img_m else None,
            'delivery_overseas': 1 if '해외배송' in html else 0,
            'platform':          '11번가',
        }
    except Exception:
        return None

# ── SSG 크롤러 (playwright) ───────────────────────────────────
def crawl_ssg(url):
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--no-sandbox','--disable-dev-shm-usage','--disable-gpu'])
            ctx  = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36',
                locale='ko-KR')
            page = ctx.new_page()
            page.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
            page.goto(url, wait_until='networkidle', timeout=40000)
            page.wait_for_timeout(5000)
            # 추가
            page.evaluate("window.scrollTo(0, 300)")
            page.wait_for_timeout(1000)
            
            # 스크롤해서 이미지 lazy load 트리거
            page.evaluate("window.scrollTo(0, 500)")
            page.wait_for_timeout(1000)
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(500)

            try:    name = page.locator('span.cdtl_info_tit_txt').first.inner_text().strip()
            except: name = ''

            # 변경
            # 변경
            prices = page.evaluate(r"""() =>
                Array.from(document.querySelectorAll('[class*="price"]'))
                .map(e => e.innerText.trim())
                .filter(t => t.match(/^\d[\d,]+$/))
            """)
            price = int(re.sub(r'[^\d]', '', prices[0])) if prices else None

            # 변경
            # 변경
            img_url = page.evaluate(r"""() => {
                const n = src => (!src ? '' : src.startsWith('//') ? 'https:'+src : src);
                document.querySelectorAll('img[data-src],img[data-lazy-src]').forEach(img => {
                    const lazy = img.dataset.src || img.dataset.lazySrc || '';
                    if(lazy) img.src = lazy;
                });
                for(const img of document.querySelectorAll('img')){
                    const s = n(img.src || '');
                    if(s && s.includes('sitem.ssgcdn')) return s;
                }
                return '';
            }""")

            del_text = page.evaluate(r"""() => {
                const el=document.querySelector('.cdtl_delivery_info,[class*="delivery_wrap"]');
                return el?el.innerText:'';
            }""")
            browser.close()

        return {
            'product_name':      name,
            'price':             price,
            'image_url':         img_url,
            'delivery_overseas': 1 if '해외' in del_text else 0,
            'platform':          'ssg',
        }
    except Exception as e:
        st.warning(f"SSG 크롤링 오류: {e}")
        return None

# ── 롯데온 크롤러 (selenium) ──────────────────────────────────
def crawl_lotteon(url):
    try:
        import ast
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        opts = Options()
        opts.add_argument('--headless')
        opts.add_argument('--no-sandbox')
        opts.add_argument('--disable-dev-shm-usage')
        opts.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36')

        driver = webdriver.Chrome(options=opts)
        driver.get(url)
        wait = WebDriverWait(driver, 10)

        try:
            name = wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, '.pd-widget1__product-name'))).text.strip()
        except: name = ''

        # 가격 (변경)
        try:
            price_text = driver.execute_script("""
                const els = Array.from(document.querySelectorAll('[class*="price"]'));
                const el = els.find(e => e.innerText.includes('판매가'));
                if(!el) return '';
                const m = el.innerText.match(/판매가[\s\\n]+([\d,]+)/);
                return m ? m[1] : '';
            """)
            price = int(re.sub(r'[^\d]', '', price_text)) if price_text else None
        except: price = None

        # 이미지 (변경)
        try:
            img_url = driver.execute_script("""
                return Array.from(document.querySelectorAll('img'))
                    .map(i => i.src || i.dataset?.src || '')
                    .find(s => s.includes('contents.lotteon.com/itemimage')) || '';
            """)
        except: img_url = None

        try:
            del_text = driver.find_element(
                By.XPATH,
                '//*[@id="stickyOptionParent"]/div/div/div/div[1]/div/div/div[1]/dl[1]/dd/p[1]'
            ).text
        except: del_text = ''

        # 판매자 랭크 (metaData)
        store_rank = np.nan
        try:
            meta = driver.find_element(By.ID, 'metaData').get_attribute('value')
            try:    meta_json = json.loads(meta)
            except: meta_json = ast.literal_eval(meta)
            grade = meta_json.get('product',{}).get('basicInfo',{}).get('sellerGrade','')
            raw_rank = {'SUPER':3,'PREMIUM':2,'STANDARD':1,'BASIC':0}.get(
    str(grade).upper(), np.nan)
            store_rank = raw_rank / 3 if not pd.isna(raw_rank) else np.nan
        except: pass

        driver.quit()

        return {
            'product_name':      name,
            'price':             price,
            'image_url':         img_url,
            'delivery_overseas': 1 if '해외' in del_text else 0,
            'store_rank':        store_rank,
            'platform':          'lotteon',
        }
    except Exception as e:
        st.warning(f"롯데온 크롤링 오류: {e}")
        return None

# ── 가격 비율 ─────────────────────────────────────────────────
def calc_price_rate(price, brand, volume):
    key = (brand, int(volume) if volume else 0)
    official = OFFICIAL_PRICES.get(key)
    return round(price / official, 4) if (official and price) else np.nan

# ── 피처 빌드 ─────────────────────────────────────────────────
def build_features(info, norm_stats, official_quality):
    f        = {}
    platform = info.get('platform', '11번가')
    brand    = info.get('brand_name', 'unknown')

    f['price_rate'] = float(np.clip(info.get('price_rate', np.nan), 0, 5))

    like_cnt = info.get('store_like_count', 0)
    if platform == 'Gmarket':
        f['store_like_count_norm'] = np.nan
    else:
        stats   = norm_stats.get(platform, {'min':0,'max':1})
        log_val = np.log1p(float(like_cnt) if like_cnt else 0)
        mn, mx  = stats['min'], stats['max']
        f['store_like_count_norm'] = float(
            np.clip((log_val-mn)/(mx-mn) if mx>mn else 0, 0, 1))

    f['brand_name']        = float(BRAND_ENCODER.get(brand, -1))
    f['delivery_overseas'] = float(info.get('delivery_overseas', 0))
    f['store_rank']        = info.get('store_rank', np.nan)

    referer_map = {'11번가':'https://www.11st.co.kr/',
                   'ssg':'https://www.ssg.com/',
                   'lotteon':'https://www.lotteon.com/',
                   'smartstore':'https://smartstore.naver.com/'}
    img = download_image(info.get('image_url'),
                         referer=referer_map.get(platform,'https://www.11st.co.kr/'))

    if img is not None:
        lap  = calc_laplacian(img)
        bris = calc_brisque(img)
        w, h = img.shape[1], img.shape[0]
        lap_main = est_lap_600(lap,w,h) if platform=='smartstore' else lap
        f['lap_main']     = lap_main
        f['sale_brisque'] = bris
        f['resolution']   = float(w*h)
        key = (brand, int(info.get('volume',0)) if info.get('volume') else 0)
        if key in official_quality:
            oq = official_quality[key]
            f['laplacian_ratio'] = lap_main/(oq['official_laplacian']+1e-6)
            f['brisque_diff']    = bris - oq['official_brisque']
        else:
            f['laplacian_ratio'] = np.nan
            f['brisque_diff']    = np.nan
    else:
        for k in ['lap_main','sale_brisque','resolution','laplacian_ratio','brisque_diff']:
            f[k] = np.nan

    f['lap_change_pct'] = np.nan
    for p in ['11번가','Gmarket','lotteon','smartstore','ssg']:
        f[f'platform_{p}'] = 1.0 if platform==p else 0.0

    return pd.DataFrame([f])[FEATURES]

# ════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════
st.set_page_config(page_title="가품 탐지기", page_icon="🔍", layout="centered")
st.title("🔍 화장품 가품 탐지기")
st.caption("상품 URL을 입력하면 가품 여부를 자동으로 분석합니다.")

try:
    model      = load_model()
    norm_stats = load_norm_stats()
    official_q = load_official_quality()
except Exception as e:
    st.error(f"모델 파일 로드 실패: {e}")
    st.stop()

with st.form("input_form"):
    platform_ui = st.selectbox("플랫폼 선택", ["11번가", "SSG", "롯데온"])
    url = st.text_input("상품 URL", placeholder="해당 플랫폼 상품 URL을 붙여넣으세요")
    col1, col2 = st.columns(2)
    with col1:
        brand  = st.selectbox("브랜드", ["Jo_Malone", "Kiehls", "Estee_Lauder"])
        volume = st.number_input("용량 (ml)", min_value=0, value=50, step=1)
    with col2:
        store_like = st.number_input("판매자 찜 수", min_value=0, value=0, step=1)
        if platform_ui == "11번가":
            rank_label = st.selectbox("판매자 랭크 (11번가, 숫자 클수록 높음)",
                ["없음/모름", "0", "1", "2", "3", "4", "5"])
            store_rank = np.nan if rank_label == "없음/모름" else int(rank_label) / 5
        elif platform_ui == "SSG":
            rank_label = st.selectbox("판매자 랭크 (SSG, 숫자 클수록 높음)",
                ["없음/모름", "0", "1", "2", "3", "4"])
            store_rank = np.nan if rank_label == "없음/모름" else int(rank_label) / 4
        else:
            st.info("롯데온 판매자 랭크는 페이지에서 자동 추출됩니다.")
            store_rank = np.nan
    # ← 여기! col2 밖, form 안 맨 끝에 한 번만
    submitted = st.form_submit_button("🔍 분석 시작", use_container_width=True)

    with st.spinner(f"{platform_ui} 상품 정보 크롤링 중..."):
        product = {'11번가': crawl_11st,
                   'SSG':    crawl_ssg,
                   '롯데온': crawl_lotteon}[platform_ui](url)

    if product is None:
        st.error("상품 정보를 가져오지 못했어요. URL을 확인해주세요.")
        st.stop()


    platform_key = {'11번가':'11번가', 'SSG':'ssg', '롯데온':'lotteon'}[platform_ui]
    price_rate   = calc_price_rate(product.get('price'), brand, volume)

    st.subheader("📦 상품 정보")
    c1, c2 = st.columns([1, 2])
    with c1:
        if product.get('image_url'):
            st.image(product['image_url'], width=200)
    with c2:
        st.write(f"**상품명:** {product.get('product_name','-')}")
        if product.get('price'):
            st.write(f"**가격:** {product['price']:,}원")
        st.write(f"**공식가 대비:** {price_rate:.1%}" if not pd.isna(price_rate) else "**공식가 대비:** 계산 불가")
        st.write(f"**해외배송:** {'예' if product.get('delivery_overseas') else '아니오'}")

    with st.spinner("이미지 분석 중..."):
        info = {
            'price_rate':        price_rate,
            'store_like_count':  store_like,
            'brand_name':        brand,
            'delivery_overseas': product.get('delivery_overseas', 0),
            'store_rank': product.get('store_rank', store_rank if not pd.isna(store_rank) else np.nan),
            'image_url':         product.get('image_url'),
            'platform':          platform_key,
            'volume':            volume,
        }
        X = build_features(info, norm_stats, official_q)

    prob = float(model.predict_proba(X)[0][1])

    st.divider()
    st.subheader("🎯 분석 결과")

    if prob >= 0.6:
        st.error(f"🔴 **가품 의심** (가품 확률: {prob:.1%})")
    elif prob >= 0.4:
        st.warning(f"🟡 **주의 필요** (가품 확률: {prob:.1%})")
    else:
        st.success(f"🟢 **정품 가능성 높음** (가품 확률: {prob:.1%})")

    st.progress(prob)
    st.caption("🔴 60% 이상: 가품 의심  |  🟡 40~60%: 주의 필요  |  🟢 40% 미만: 정품 가능성 높음")

    # SHAP
    with st.expander("📊 판정 근거 (SHAP 분석)", expanded=True):
        try:
            import shap, matplotlib.pyplot as plt, matplotlib
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.rcParams['font.family'] = ['AppleGothic', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            matplotlib.rcParams['mathtext.fontset'] = 'cm'

            explainer   = shap.TreeExplainer(model)
            shap_values = explainer(X.astype(float))

            labels = {
                'price_rate':'공식가 대비 가격비율',
                'store_like_count_norm':'판매자 찜 수 (정규화)',
                'brand_name':'브랜드', 'delivery_overseas':'해외배송 여부',
                'store_rank':'판매자 랭크', 'lap_main':'이미지 선명도',
                'sale_brisque':'이미지 노이즈', 'laplacian_ratio':'공식이미지 대비 선명도',
                'brisque_diff':'공식이미지 대비 노이즈차', 'resolution':'이미지 해상도',
                'lap_change_pct':'선명도 변화율', 'platform_11번가':'플랫폼_11번가',
                'platform_Gmarket':'플랫폼_Gmarket', 'platform_lotteon':'플랫폼_롯데온',
                'platform_smartstore':'플랫폼_스마트스토어', 'platform_ssg':'플랫폼_SSG',
            }

            shap_df = pd.DataFrame({
                '피처':   [labels.get(f,f) for f in FEATURES],
                '피처값': [f"{v:.4f}" if pd.notna(v) else "NaN" for v in X.iloc[0].values],
                'SHAP값': shap_values.values[0],
            })
            shap_df['영향'] = shap_df['SHAP값'].apply(
                lambda x: '🔴 가품 방향' if x>0.001 else ('🟢 정품 방향' if x<-0.001 else '➖ 중립'))
            shap_df = shap_df.sort_values('SHAP값', key=abs, ascending=False)
            shap_df['SHAP값'] = shap_df['SHAP값'].apply(lambda x: f"{float(x):+.4f}")
            st.dataframe(shap_df.reset_index(drop=True), use_container_width=True)

            st.write("**SHAP 워터폴 차트**")
            fig, _ = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(shap_values[0], max_display=12, show=False)
            st.pyplot(fig)
            plt.close()

        except ImportError:
            st.info("SHAP 설치 필요: pip install shap")
        except Exception as e:
            st.error(f"SHAP 분석 오류: {e}")

    with st.expander("📋 피처 원본값"):
        display_df = X.T.rename(columns={0:'값'})
        display_df['값'] = display_df['값'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
        st.dataframe(display_df, use_container_width=True)
