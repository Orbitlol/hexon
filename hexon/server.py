"""
Hexon Interactive Mobile Model Inspector, Playground & Export Download Server.

Serves an interactive UI on 0.0.0.0:8000 where users can:
- Test live inference on the exported GGUF v3 models (`Q4_0`, `Q8_0`, `F16`)
- Inspect GGUF v3 metadata and tensor quantization blocks
- Download the exported `.gguf` and `.pt` files directly
"""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict
from urllib.parse import unquote

from .gguf_runtime import GGUFReader
from .inference import HexonMobileEngine
from .rag import HexonChromaRAG


EXPORTS_DIR = Path("exports")
ENGINES: Dict[str, HexonMobileEngine] = {}
RAG_ENGINE: HexonChromaRAG | None = None


def get_rag() -> HexonChromaRAG:
    global RAG_ENGINE
    if RAG_ENGINE is None:
        RAG_ENGINE = HexonChromaRAG(persist_dir="data/chromadb")
    return RAG_ENGINE


def get_engine(quant: str) -> HexonMobileEngine:
    quant = quant.lower()
    if quant == "135m":
        if "135m" not in ENGINES:
            gguf_path = EXPORTS_DIR / "hexon-135m-q4_0.gguf"
            ENGINES["135m"] = HexonMobileEngine.from_file(gguf_path, rag=get_rag())
        return ENGINES["135m"]
    if quant not in ("q4_0", "q8_0", "f16"):
        quant = "q8_0"
    if quant not in ENGINES:
        gguf_path = EXPORTS_DIR / f"hexon-nano-{quant}.gguf"
        ENGINES[quant] = HexonMobileEngine.from_file(gguf_path, rag=get_rag())
    return ENGINES[quant]


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Hexon — Mobile AI Model (PyTorch + GGUF v3)</title>
  <style>
    :root {
      --bg: #0b0f17;
      --panel: #131b2e;
      --border: #23304d;
      --accent: #38bdf8;
      --accent2: #818cf8;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --ok: #34d399;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
      line-height: 1.5;
    }
    .container { max-width: 1080px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }
    .badge {
      background: rgba(52, 211, 153, 0.14);
      color: var(--ok);
      border: 1px solid rgba(52, 211, 153, 0.4);
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 600;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 20px;
    }
    @media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
    }
    h2 { font-size: 18px; margin-bottom: 14px; color: var(--accent); }
    .chat-box {
      background: #080c14;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      height: 300px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 12px;
    }
    .msg { padding: 10px 14px; border-radius: 10px; max-width: 88%; font-size: 14px; }
    .msg.user { background: #1e293b; align-self: flex-end; }
    .msg.bot { background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); align-self: flex-start; }
    .meta { font-size: 11px; color: var(--muted); margin-top: 4px; }
    .controls { display: flex; gap: 8px; margin-bottom: 10px; }
    select, input, button {
      background: #0f172a;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 14px;
    }
    input { flex: 1; }
    button {
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      color: #0b0f17;
      font-weight: 700;
      cursor: pointer;
      border: none;
    }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
    .chip {
      font-size: 12px;
      padding: 5px 10px;
      border-radius: 999px;
      background: #1e293b;
      color: var(--text);
      cursor: pointer;
      border: 1px solid var(--border);
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
    th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); font-weight: 600; }
    a.dl {
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }
    a.dl:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Hexon Mobile AI Model</h1>
        <p style="color: var(--muted); font-size: 14px;">
          PyTorch GQA Transformer + Spec-Compliant GGUF v3 &amp; TorchScript Mobile Exports
        </p>
      </div>
      <span class="badge">● Mobile Inference Locked (No On-Device Training)</span>
    </header>

    <div class="grid">
      <div class="card">
        <h2>Live On-Device GGUF + ChromaDB RAG Playground</h2>
        <div class="chips">
          <span class="chip" onclick="ask('Who are you and what is your name?')">Who are you?</span>
          <span class="chip" onclick="ask('What processor, GPU, and battery does the Poco M3 have?')">Poco M3 specs (RAG)</span>
          <span class="chip" onclick="ask('How does ChromaDB perform fast vector similarity search?')">ChromaDB HNSW (RAG)</span>
          <span class="chip" onclick="ask('Write a Python function to compute the nth Fibonacci number iteratively.')">Python Fibonacci</span>
          <span class="chip" onclick="ask('Explain why ocean tides happen on Earth.')">Why ocean tides happen</span>
        </div>
        <div class="chat-box" id="chat"></div>
        <div class="controls">
          <select id="quant">
            <option value="135m" selected>Hexon-135M GGUF (93.8 MiB • 134.5M Params)</option>
            <option value="q8_0">Hexon-Nano Q8_0 (3.04 MB • 2.56M Params)</option>
            <option value="q4_0">Hexon-Nano Q4_0 (1.63 MB • 2.56M Params)</option>
            <option value="f16">Hexon-Nano F16 (5.67 MB • 2.56M Params)</option>
          </select>
          <label style="display:flex;align-items:center;gap:6px;font-size:13px;background:#0f172a;padding:6px 10px;border-radius:8px;border:1px solid var(--border);">
            <input type="checkbox" id="use_rag" checked style="flex:0;" /> ChromaDB RAG
          </label>
        </div>
        <div class="controls">
          <input id="prompt" type="text" placeholder="Ask Hexon any question..." value="What processor, GPU, and battery does the Poco M3 have?" />
          <button onclick="sendPrompt()">Run GGUF + RAG</button>
        </div>
      </div>

      <div class="card">
        <h2>Exported Mobile Artifacts (Ready to Download)</h2>
        <table>
          <thead>
            <tr><th>Artifact</th><th>Format</th><th>Size</th><th>Download</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><code>hexon-135m-q4_0.gguf</code></td>
              <td>GGUF v3 (135M Compositional)</td>
              <td>93.81 MiB</td>
              <td><a class="dl" href="/download/hexon-135m-q4_0.gguf">Download</a></td>
            </tr>
            <tr>
              <td><code>hexon-nano-q4_0.gguf</code></td>
              <td>GGUF v3 (4-bit)</td>
              <td>1.63 MB</td>
              <td><a class="dl" href="/download/hexon-nano-q4_0.gguf">Download</a></td>
            </tr>
            <tr>
              <td><code>hexon-nano-q8_0.gguf</code></td>
              <td>GGUF v3 (8-bit)</td>
              <td>3.04 MB</td>
              <td><a class="dl" href="/download/hexon-nano-q8_0.gguf">Download</a></td>
            </tr>
            <tr>
              <td><code>hexon-nano-f16.gguf</code></td>
              <td>GGUF v3 (FP16)</td>
              <td>5.67 MB</td>
              <td><a class="dl" href="/download/hexon-nano-f16.gguf">Download</a></td>
            </tr>
            <tr>
              <td><code>hexon-nano-mobile-int8.pt</code></td>
              <td>PyTorch INT8</td>
              <td>3.82 MB</td>
              <td><a class="dl" href="/download/hexon-nano-mobile-int8.pt">Download</a></td>
            </tr>
          </tbody>
        </table>

        <h2 style="margin-top: 20px;">ChromaDB RAG &amp; GGUF v3 Metadata</h2>
        <pre id="gguf-meta" style="background:#080c14;padding:12px;border-radius:8px;font-size:12px;overflow-x:auto;color:#cbd5e1;border:1px solid var(--border);">Loading GGUF &amp; ChromaDB header...</pre>
      </div>
    </div>
  </div>

  <script>
    const chat = document.getElementById('chat');
    function appendMsg(role, text, meta) {
      const div = document.createElement('div');
      div.className = 'msg ' + role;
      div.innerHTML = '<div>' + text + '</div>' + (meta ? '<div class="meta">' + meta + '</div>' : '');
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }

    async function sendPrompt() {
      const input = document.getElementById('prompt');
      const quant = document.getElementById('quant').value;
      const use_rag = document.getElementById('use_rag').checked;
      const prompt = input.value.trim();
      if (!prompt) return;
      appendMsg('user', prompt);
      input.value = '';
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, quant, use_rag })
      });
      const data = await res.json();
      const ragInfo = (data.retrieved_chunks && data.retrieved_chunks.length > 0)
        ? ` • ChromaDB Top Hit: ${data.retrieved_chunks[0].id} (score=${data.retrieved_chunks[0].score})`
        : '';
      appendMsg(
        'bot',
        data.response,
        `${data.source_format.toUpperCase()} • ${data.tokens_generated} tokens • ${data.ms_per_token} ms/tok (${data.tokens_per_sec} tok/s)${ragInfo}`
      );
    }

    function ask(q) {
      document.getElementById('prompt').value = q;
      sendPrompt();
    }

    async function loadMeta() {
      const res = await fetch('/api/inspect');
      const data = await res.json();
      document.getElementById('gguf-meta').textContent = JSON.stringify(data, null, 2);
    }

    document.getElementById('prompt').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') sendPrompt();
    });

    loadMeta();
    sendPrompt();
  </script>
</body>
</html>
"""


class HexonHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: object, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            raw = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if self.path == "/api/inspect":
            target = EXPORTS_DIR / "hexon-135m-q4_0.gguf"
            if not target.exists():
                target = EXPORTS_DIR / "hexon-nano-q4_0.gguf"
            reader = GGUFReader(target)
            info = reader.summary()
            info["chromadb_rag"] = {
                "collection": get_rag().collection_name,
                "indexed_chunks": get_rag().count(),
                "embedding_dim": 384,
            }
            self._send_json(info)
            return

        if self.path.startswith("/download/"):
            fname = Path(unquote(self.path[len("/download/") :])).name
            fpath = EXPORTS_DIR / fname
            if not fpath.exists():
                self._send_json({"error": "File not found"}, status=404)
                return
            ctype, _ = mimetypes.guess_type(str(fpath))
            data = fpath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        if self.path == "/api/generate":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = str(body.get("prompt", "Who are you?"))
            quant = str(body.get("quant", "135m"))
            use_rag = bool(body.get("use_rag", True))
            engine = get_engine(quant)
            res = engine.reply(
                prompt,
                max_new_tokens=90,
                temperature=0.0,
                use_rag=use_rag,
            )
            self._send_json(res)
            return

        if self.path == "/api/rag":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            query = str(body.get("query", "How does Hexon use ChromaDB?"))
            top_k = int(body.get("top_k", 3))
            quant = str(body.get("quant", "135m"))
            engine = get_engine(quant)
            res = get_rag().query(query, engine=engine, top_k=top_k)
            self._send_json(res)
            return

        self._send_json({"error": "Not found"}, status=404)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    # Preload Q8_0 and Q4_0 GGUF engines
    get_engine("q8_0")
    get_engine("q4_0")
    server = ThreadingHTTPServer((host, port), HexonHTTPHandler)
    print(f"Hexon Mobile Playground & Export Server listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
