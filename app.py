"""Small web chat UI for the agent.  Run:  python app.py  then open http://localhost:5000"""
from dataclasses import asdict

from flask import Flask, jsonify, request, session

from agent.agent import SupportAgent

app = Flask(__name__)
app.secret_key = "dev-only"
_agents: dict[str, SupportAgent] = {}  # one agent (conversation memory) per browser session

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Koala Gear support</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem;background:#f6f7f9}
#log{background:#fff;border-radius:12px;padding:1rem;height:60vh;overflow-y:auto;box-shadow:0 1px 3px #0002}
.m{margin:.5rem 0;padding:.6rem .8rem;border-radius:10px;max-width:85%}
.u{background:#2f6fed;color:#fff;margin-left:auto}.b{background:#eef0f3}
.meta{font-size:.75rem;color:#667;margin:-.2rem 0 .6rem .2rem}
form{display:flex;gap:.5rem;margin-top:1rem}input{flex:1;padding:.7rem;border-radius:8px;border:1px solid #ccd}
button{padding:.7rem 1.2rem;border:0;border-radius:8px;background:#2f6fed;color:#fff}
</style></head><body><h2>Koala Gear support</h2><div id="log"></div>
<form id="f"><input id="q" autocomplete="off" placeholder="Ask about an order, returns, shipping..."><button>Send</button></form>
<script>
const log=document.getElementById('log');
function add(t,c,meta){const d=document.createElement('div');d.className='m '+c;d.textContent=t;log.appendChild(d);
 if(meta){const s=document.createElement('div');s.className='meta';s.textContent=meta;log.appendChild(s)}log.scrollTop=log.scrollHeight}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const q=document.getElementById('q');const t=q.value.trim();if(!t)return;
 add(t,'u');q.value='';const r=await (await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t})})).json();
 add(r.reply,'b',`intent: ${r.intent} (${r.confidence}) · action: ${r.action}${r.escalated?' · handed to a human':''}${r.sources.length?' · sources: '+r.sources.join(', '):''}`)};
add('Hi! Try "where is KG1002?", "refund KG1001" or "do you ship to NZ?"','b');
</script></body></html>"""


@app.get("/")
def index():
    return PAGE


@app.post("/chat")
def chat():
    sid = session.setdefault("sid", str(len(_agents) + 1))
    agent = _agents.setdefault(sid, SupportAgent())
    message = (request.get_json(silent=True) or {}).get("message", "")
    return jsonify(asdict(agent.handle(message)))


if __name__ == "__main__":
    app.run(debug=True)
