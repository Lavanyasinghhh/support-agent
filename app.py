"""Small web chat UI for the agent.  Run:  python app.py  then open http://localhost:5000

The agent (models, help-centre index, order data) is created ONCE when the app
starts and shared by everyone. Each browser gets a random session id (UUID), and
the agent keeps conversation state per session id, so multiple people can chat
at the same time without seeing each other's context.
"""
import os
import secrets
import uuid
from dataclasses import asdict

from flask import Flask, jsonify, request, session

from agent.agent import SupportAgent

app = Flask(__name__)
# Use a real secret in production: export FLASK_SECRET_KEY=...
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)

agent = SupportAgent()  # loaded once, shared across requests

MAX_MESSAGE_CHARS = 1000

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
<form id="f"><input id="q" autocomplete="off" maxlength="1000" placeholder="Ask about an order, returns, shipping..."><button>Send</button></form>
<script>
const log=document.getElementById('log');
function add(t,c,meta){const d=document.createElement('div');d.className='m '+c;d.textContent=t;log.appendChild(d);
 if(meta){const s=document.createElement('div');s.className='meta';s.textContent=meta;log.appendChild(s)}log.scrollTop=log.scrollHeight}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const q=document.getElementById('q');const t=q.value.trim();if(!t)return;
 add(t,'u');q.value='';
 try{const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t})});
  const r=await res.json(); if(!res.ok){add(r.error||'Something went wrong.','b');return}
  add(r.reply,'b',`intent: ${r.intent} (${r.confidence}) · action: ${r.action}${r.escalated?' · handed to a human':''}${r.sources.length?' · sources: '+r.sources.join(', '):''}`)}
 catch(err){add('Could not reach the server.','b')}};
add('Hi! Try "where is KG1002?", "refund KG1001" or "do you ship to NZ?"','b');
</script></body></html>"""


@app.get("/")
def index():
    return PAGE


@app.post("/chat")
def chat():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex  # unguessable, never reused
    message = ((request.get_json(silent=True) or {}).get("message") or "").strip()
    if not message:
        return jsonify(error="Message is empty."), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify(error=f"Message is too long (max {MAX_MESSAGE_CHARS} characters)."), 400
    return jsonify(asdict(agent.handle(message, session_id=session["sid"])))


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
