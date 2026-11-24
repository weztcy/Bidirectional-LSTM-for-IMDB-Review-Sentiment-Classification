import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const PORT = 9222;
const APP_URL = "http://localhost:8501";
const OUT_DIR = path.resolve("screenshots");

if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

function delay(ms) {
  return new Promise(res => setTimeout(res, ms));
}

class CDPClient {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.id = 1;
    this.callbacks = new Map();

    this.ready = new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = reject;
    });

    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id && this.callbacks.has(msg.id)) {
        const { resolve, reject } = this.callbacks.get(msg.id);
        this.callbacks.delete(msg.id);
        if (msg.error) {
          reject(msg.error);
        } else {
          resolve(msg.result);
        }
      }
    };
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const msgId = this.id++;
      this.callbacks.set(msgId, { resolve, reject });
      this.ws.send(JSON.stringify({ id: msgId, method, params }));
    });
  }

  async eval(expression) {
    const res = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true
    });
    if (res.exceptionDetails) {
      console.error("Eval Exception:", res.exceptionDetails);
      throw new Error(res.exceptionDetails.text || 'Eval error');
    }
    return res.result ? res.result.value : undefined;
  }

  async uncapPageAndMeasure() {
    // Uncap Streamlit fixed height overflow containers so full document expands naturally
    return await this.eval(`
      (() => {
        // Inject uncap styles if not present
        let style = document.getElementById('full-page-uncap-style');
        if (!style) {
          style = document.createElement('style');
          style.id = 'full-page-uncap-style';
          style.innerHTML = \`
            html, body, .stApp, [data-testid="stAppViewContainer"], section.main, [data-testid="stMain"] {
              height: auto !important;
              min-height: 100% !important;
              overflow: visible !important;
              overflow-y: visible !important;
            }
            [data-testid="stSidebar"] {
              height: 100% !important;
              min-height: 100% !important;
            }
          \`;
          document.head.appendChild(style);
        }

        // Measure true bottom boundary across all elements
        let maxBottom = 0;
        document.querySelectorAll('*').forEach(el => {
          const rect = el.getBoundingClientRect();
          const b = rect.bottom + window.scrollY;
          if (b > maxBottom) maxBottom = b;
        });

        const mainBlock = document.querySelector('[data-testid="stMainBlockContainer"]') || document.querySelector('.block-container');
        const sidebar = document.querySelector('[data-testid="stSidebarUserContent"]') || document.querySelector('[data-testid="stSidebar"]');

        const mainH = mainBlock ? mainBlock.scrollHeight : 0;
        const sidebarH = sidebar ? sidebar.scrollHeight : 0;

        const effectiveHeight = Math.max(
          Math.ceil(maxBottom) + 60,
          mainH + 120,
          sidebarH + 80,
          1080
        );

        return {
          width: 1440,
          height: effectiveHeight
        };
      })()
    `);
  }

  async captureFullPage(filepath) {
    const dim = await this.uncapPageAndMeasure();
    console.log(`Setting viewport for full page capture: ${dim.width}x${dim.height}`);

    await this.send('Emulation.setDeviceMetricsOverride', {
      width: dim.width,
      height: dim.height,
      deviceScaleFactor: 2,
      mobile: false
    });

    await delay(800);

    const screenshot = await this.send('Page.captureScreenshot', {
      format: 'png',
      captureBeyondViewport: false,
      clip: {
        x: 0,
        y: 0,
        width: dim.width,
        height: dim.height,
        scale: 1
      }
    });

    // Reset viewport back to standard
    await this.send('Emulation.setDeviceMetricsOverride', {
      width: 1440,
      height: 900,
      deviceScaleFactor: 2,
      mobile: false
    });

    const buffer = Buffer.from(screenshot.data, 'base64');
    fs.writeFileSync(filepath, buffer);
    console.log(`Saved screenshot: ${filepath} (${buffer.length} bytes, ${dim.width}x${dim.height})`);
  }

  close() {
    this.ws.close();
  }
}

async function main() {
  console.log("Launching Microsoft Edge in headless mode...");
  const edgeProc = spawn(EDGE_PATH, [
    `--remote-debugging-port=${PORT}`,
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--window-size=1440,900',
    'about:blank'
  ]);

  await delay(2000);

  console.log("Opening new tab at", APP_URL);
  const target = await new Promise((resolve, reject) => {
    const req = http.request({
      hostname: '127.0.0.1',
      port: PORT,
      path: `/json/new?${APP_URL}`,
      method: 'PUT'
    }, (res) => {
      let d = '';
      res.on('data', chunk => d += chunk);
      res.on('end', () => resolve(JSON.parse(d)));
    });
    req.on('error', reject);
    req.end();
  });

  console.log("Target connected:", target.id);
  const client = new CDPClient(target.webSocketDebuggerUrl);
  await client.ready;

  await client.send('Page.enable');
  await client.send('DOM.enable');
  await client.send('Runtime.enable');

  await client.send('Emulation.setDeviceMetricsOverride', {
    width: 1440,
    height: 900,
    deviceScaleFactor: 2,
    mobile: false
  });

  console.log("Waiting for Streamlit app to load...");
  for (let i = 0; i < 30; i++) {
    await delay(1000);
    const tabs = await client.eval(`
      (() => {
        const els = Array.from(document.querySelectorAll('[data-testid="stTab"], [role="tab"]'));
        return els.map(e => e.textContent.trim()).filter(Boolean);
      })()
    `).catch(() => []);

    if (tabs && tabs.length >= 3) {
      console.log("Found tabs:", tabs);
      break;
    }
  }

  await delay(2500);

  // --- SCREENSHOT 1: Tab 1 - Live Prediction (Positive Sample) ---
  console.log("\n[1/4] Capturing FULL PAGE Tab 1: Live Prediction (Positive Sample)...");
  let hasCard = await client.eval(`Boolean(document.querySelector('.sentiment-card'))`);
  if (!hasCard) {
    console.log("Clicking Analyze Sentiment...");
    await client.eval(`
      (() => {
        const btn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.includes('Analyze Sentiment'));
        if (btn) btn.click();
      })()
    `);
    for (let i = 0; i < 10; i++) {
      await delay(1000);
      hasCard = await client.eval(`Boolean(document.querySelector('.sentiment-card'))`);
      if (hasCard) break;
    }
  }
  await delay(1500);
  await client.captureFullPage(path.join(OUT_DIR, "01_fullpage_live_prediction_positive.png"));

  // --- SCREENSHOT 2: Tab 1 - Live Prediction (Negative Sample) ---
  console.log("\n[2/4] Capturing FULL PAGE Tab 1: Live Prediction (Negative Sample)...");
  await client.eval(`
    (() => {
      const textarea = document.querySelector('textarea');
      if (textarea) {
        const negativeReview = "What an awful excuse for a movie. The acting was stiff and completely devoid of emotion, the plot holes were big enough to drive a truck through, and the pacing dragged relentlessly. Don't waste your money.";
        const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
        nativeSetter.call(textarea, negativeReview);
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
        textarea.dispatchEvent(new Event('change', { bubbles: true }));
      }
      setTimeout(() => {
        const btn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.includes('Analyze Sentiment'));
        if (btn) btn.click();
      }, 500);
    })()
  `);

  console.log("Waiting for negative sentiment analysis result...");
  for (let i = 0; i < 12; i++) {
    await delay(1000);
    const isNeg = await client.eval(`
      (() => {
        const card = document.querySelector('.sentiment-card');
        return card && card.textContent.includes('NEGATIVE');
      })()
    `);
    if (isNeg) {
      console.log("Negative sentiment card rendered!");
      break;
    }
  }
  await delay(1500);
  await client.captureFullPage(path.join(OUT_DIR, "02_fullpage_live_prediction_negative.png"));

  // --- SCREENSHOT 3: Tab 2 - Model Evaluation & Metrics ---
  console.log("\n[3/4] Switching to Tab 2: Model Evaluation & Metrics...");
  const tab2Text = await client.eval(`
    (() => {
      const tabs = Array.from(document.querySelectorAll('[data-testid="stTab"], [role="tab"]'));
      if (tabs.length >= 2) {
        tabs[1].click();
        return tabs[1].textContent.trim();
      }
      return null;
    })()
  `);
  console.log("Clicked Tab 2:", tab2Text);

  // Wait for Tab 2 content (metrics values, table, figures)
  for (let i = 0; i < 15; i++) {
    await delay(1000);
    const hasTab2Content = await client.eval(`
      (() => {
        const text = document.body.innerText;
        const hasImgs = document.querySelectorAll('img').length >= 1;
        return text.includes('Holdout Test Set Performance') && hasImgs;
      })()
    `);
    if (hasTab2Content) {
      console.log("Tab 2 metrics and figures loaded!");
      break;
    }
  }
  await delay(2500);
  await client.captureFullPage(path.join(OUT_DIR, "03_fullpage_model_evaluation.png"));

  // --- SCREENSHOT 4: Tab 3 - Methodology & Architecture ---
  console.log("\n[4/4] Switching to Tab 3: Methodology & Architecture...");
  const tab3Text = await client.eval(`
    (() => {
      const tabs = Array.from(document.querySelectorAll('[data-testid="stTab"], [role="tab"]'));
      if (tabs.length >= 3) {
        tabs[2].click();
        return tabs[2].textContent.trim();
      }
      return null;
    })()
  `);
  console.log("Clicked Tab 3:", tab3Text);

  // Wait for Tab 3 content (Methodology text and loss chart)
  for (let i = 0; i < 15; i++) {
    await delay(1000);
    const hasTab3Content = await client.eval(`
      (() => {
        const text = document.body.innerText;
        const hasImgs = document.querySelectorAll('img').length >= 1;
        return text.includes('Architectural Design & Pipeline') && hasImgs;
      })()
    `);
    if (hasTab3Content) {
      console.log("Tab 3 methodology and loss figure loaded!");
      break;
    }
  }
  await delay(2500);
  await client.captureFullPage(path.join(OUT_DIR, "04_fullpage_methodology_architecture.png"));

  console.log("\nAll full-page screenshots successfully captured!");

  client.close();
  edgeProc.kill();
  process.exit(0);
}

main().catch((err) => {
  console.error("Error during full page capture:", err);
  process.exit(1);
});
