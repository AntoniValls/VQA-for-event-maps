import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_from_directory

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_PATH = Path("../data/America/BuenosAires")
GT_PATH = BASE_PATH / "ground_truth_labels.jsonl"
IMG_DIR = BASE_PATH / "images_selected"
MODELS = ["llava", "instructblip", "qwen-vl", "vilt"]

def normalize_yes_no(answer: str) -> str:
    if not isinstance(answer, str): return str(answer).lower()
    a = answer.strip().lower().rstrip('.') # Remove trailing periods
    if a.startswith("yes"): return "yes"
    if a.startswith("no"): return "no"
    return a

def load_data():
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        gt_data = [json.loads(line) for line in f if line.strip()]
    
    model_results = {}
    for m in MODELS:
        path = BASE_PATH / f"results/{m}/answers.jsonl"
        if path.exists():
            content = open(path, 'r', encoding='utf-8').read().replace('}{', '}\n{')
            model_results[m] = [json.loads(l) for l in content.splitlines() if l.strip()]
        else:
            model_results[m] = []
    return gt_data, model_results

GT_ENTRIES, MODEL_ANSWERS = load_data()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>GT Corrector</title>
    <style>
        body { font-family: sans-serif; display: flex; height: 100vh; margin: 0; background: #121212; color: white; }
        #sidebar { width: 350px; overflow-y: auto; border-right: 1px solid #333; padding: 15px; background: #1a1a1a; }
        #main { flex-grow: 1; padding: 20px; display: flex; flex-direction: column; align-items: center; overflow-y: auto; }
        .counter-badge { background: #ff4444; padding: 5px 12px; border-radius: 20px; font-weight: bold; margin-bottom: 10px; display: inline-block; }
        .thumb-container { margin-bottom: 15px; cursor: pointer; border: 2px solid #333; padding: 5px; border-radius: 5px; }
        .thumb-container.active { border-color: #007bff; background: #222; }
        .thumb-img { width: 100%; border-radius: 3px; }
        img#large-view { max-height: 50vh; max-width: 95%; border: 2px solid #444; }
        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; width: 100%; margin-top: 20px; }
        .card { background: #222; padding: 10px; border-radius: 5px; border: 1px solid #444; text-align: center; }
        .card.disagree { border-color: #ff4444; background: #321; }
        .btn { padding: 15px 40px; font-size: 1.2em; cursor: pointer; border: none; border-radius: 8px; margin: 10px; color: white; font-weight: bold; }
        .btn-yes { background: #2e7d32; }
        .btn-no { background: #c62828; }
    </style>
</head>
<body>
    <div id="sidebar">
        <div class="counter-badge">Suspicious Frames: {{ review_indices|length }}</div>
        {% for idx in review_indices %}
        <div class="thumb-container" id="thumb-{{ idx }}" onclick="loadFrame({{ idx }})">
            <img src="/img/{{ data[idx].image_name }}" class="thumb-img">
            <div style="font-size: 0.8em; color: #aaa; margin-top: 5px;">{{ data[idx].short_label }}</div>
        </div>
        {% endfor %}
    </div>
    <div id="main">
        <h2 id="q-text">Select an image</h2>
        <img id="large-view" src="" style="display:none;">
        <div id="controls" style="display:none; text-align:center;">
            <h3>GT: <span id="current-gt"></span></h3>
            <button class="btn btn-yes" onclick="updateGT('yes')">YES</button>
            <button class="btn btn-no" onclick="updateGT('no')">NO</button>
        </div>
        <div class="grid" id="model-grid"></div>
    </div>

    <script>
        let currentIndex = -1;
        const gtEntries = {{ data|tojson }};
        const modelAnswers = {{ model_answers|tojson }};

        function norm(s) {
            if (!s) return "";
            let res = s.toString().toLowerCase().trim();
            if (res.endsWith('.')) res = res.slice(0, -1);
            if (res.startsWith('yes')) return 'yes';
            if (res.startsWith('no')) return 'no';
            return res;
        }

        function loadFrame(idx) {
            currentIndex = idx;
            const item = gtEntries[idx];
            document.querySelectorAll('.thumb-container').forEach(e => e.classList.remove('active'));
            document.getElementById('thumb-'+idx).classList.add('active');
            
            const img = document.getElementById('large-view');
            img.src = "/img/" + item.image_name;
            img.style.display = 'block';
            
            document.getElementById('q-text').innerText = item.question;
            document.getElementById('current-gt').innerText = item.answer.toUpperCase();
            document.getElementById('current-gt').style.color = item.answer === 'yes' ? '#4caf50' : '#f44336';
            document.getElementById('controls').style.display = 'block';
            
            let html = '';
            for (const [model, answers] of Object.entries(modelAnswers)) {
                const ansObj = answers.find(a => a.image_name === item.image_name && a.question_id === item.question_id);
                const raw = ansObj ? ansObj.answer : "N/A";
                const isDiff = norm(raw) !== norm(item.answer);
                html += `<div class="card ${isDiff ? 'disagree' : ''}"><strong>${model}</strong><p>${raw}</p></div>`;
            }
            document.getElementById('model-grid').innerHTML = html;
        }

        function updateGT(val) {
            fetch('/update', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ index: currentIndex, answer: val })
            }).then(() => {
                gtEntries[currentIndex].answer = val;
                loadFrame(currentIndex);
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    review_indices = []
    for i, entry in enumerate(GT_ENTRIES):
        cur_gt = normalize_yes_no(entry['answer'])
        for m in MODELS:
            ans_obj = next((a for a in MODEL_ANSWERS[m] if a['image_name'] == entry['image_name'] and a['question_id'] == entry['question_id']), None)
            if ans_obj and normalize_yes_no(ans_obj['answer']) != cur_gt:
                review_indices.append(i)
                break
    return render_template_string(HTML_TEMPLATE, data=GT_ENTRIES, model_answers=MODEL_ANSWERS, review_indices=review_indices)

@app.route('/img/<filename>')
def get_img(filename):
    return send_from_directory(IMG_DIR, filename)

@app.route('/update', methods=['POST'])
def update():
    req = request.json
    idx, new_ans = req['index'], req['answer']
    GT_ENTRIES[idx]['answer'] = new_ans
    GT_ENTRIES[idx]['timestamp'] = datetime.now().isoformat()
    
    # Safety Backup
    shutil.copy2(GT_PATH, GT_PATH.with_suffix('.jsonl.bak'))
    
    # FIXED SAVING LOGIC: No escaping newlines
    with open(GT_PATH, 'w', encoding='utf-8') as f:
        for entry in GT_ENTRIES:
            f.write(json.dumps(entry) + '\n') # Real newline character
            
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=False, port=5000)