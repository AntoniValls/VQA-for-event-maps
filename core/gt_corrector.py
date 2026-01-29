import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_from_directory

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_PATH = Path("../data/Africa/Marrakesh")  
GT_PATH = BASE_PATH / "ground_truth_labels.jsonl"
IMG_DIR = BASE_PATH / "images_selected"
MODELS = ["llava", "instructblip", "qwen-vl", "vilt"]

# Global storage for the session
gt_data = []

def load_data():
    global gt_data
    # Load GT
    with open(GT_PATH, 'r') as f:
        gt_data = [json.loads(line) for line in f]
    
    # Load Models
    model_results = {}
    for m in MODELS:
        path = BASE_PATH / f"results/{m}/answers.jsonl"
        content = open(path, 'r').read().replace('}{', '}\n{') if path.exists() else ""
        model_results[m] = [json.loads(l) for l in content.splitlines() if l.strip()]
    
    return model_results

MODEL_ANSWERS = load_data()

# --- HTML TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>GT Corrector</title>
    <style>
        body { font-family: sans-serif; display: flex; height: 100vh; margin: 0; background: #1e1e1e; color: white; overflow: hidden; }
        #sidebar { width: 300px; overflow-y: auto; border-right: 1px solid #444; padding: 10px; height: 100%; }
        #main { flex-grow: 1; padding: 20px; display: flex; flex-direction: column; align-items: center; overflow-y: auto; }
        .thumb { width: 100%; cursor: pointer; margin-bottom: 10px; border: 4px solid transparent; opacity: 0.6; transition: 0.2s; }
        .thumb.active { border-color: #00ff00; opacity: 1; }
        .thumb.error { border-color: #ff4444; }
        img#large { max-height: 55vh; max-width: 90%; border: 1px solid #555; }
        .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px; width: 80%; }
        .card { background: #333; padding: 10px; border-radius: 8px; text-align: center; }
        .btn { padding: 10px 25px; font-size: 1.2em; cursor: pointer; border: none; border-radius: 5px; margin: 5px; font-weight: bold; }
        .yes { background: #28a745; color: white; }
        .no { background: #dc3545; color: white; }
        #counter { background: #444; padding: 5px 15px; border-radius: 20px; margin-bottom: 10px; font-weight: bold; color: #00ff00; }
        kbd { background: #555; padding: 2px 4px; border-radius: 3px; font-size: 0.8em; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h3>Suspicious ({{ review_indices|length }})</h3>
        {% for idx in review_indices %}
        <img src="/img/{{data[idx].image_name }}" class="thumb error" onclick="loadByReviewPos({{ loop.index0 }})" id="thumb-{{ idx }}">
        {% endfor %}
    </div>
    <div id="main">
        <div id="counter">Item 0 / 0</div>
        <h2 id="q-text">Select an image to start</h2>
        <img id="large" src="">
        
        <div id="controls" style="display:none; text-align: center; margin-top: 10px;">
            <h3>Current GT: <span id="current-gt"></span></h3>
            <button class="btn yes" onclick="updateGT('yes')">YES <kbd>↑</kbd></button>
            <button class="btn no" onclick="updateGT('no')">NO <kbd>↓</kbd></button>
            <p style="color: #aaa;">Use <kbd>←</kbd> <kbd>→</kbd> to navigate</p>
        </div>
        
        <div class="grid" id="model-grid"></div>
    </div>

    <script>
        let currentReviewPos = 0; // The index within the reviewIndices array
        const gtData = {{ data|tojson }};
        const modelAnswers = {{ model_answers|tojson }};
        const reviewIndices = {{ review_indices|tojson }};

        function loadByReviewPos(pos) {
            if (pos < 0 || pos >= reviewIndices.length) return;
            currentReviewPos = pos;
            const globalIndex = reviewIndices[pos];
            loadFrame(globalIndex);
        }

        function loadFrame(idx) {
            const item = gtData[idx];
            
            // Determine which question text to use
            let fullPrompt = item.question || ""; // Ensure it's at least an empty string
            for (const m of {{ MODELS|tojson }}) {
                const modelMatch = modelAnswers[m].find(a => 
                    a.image_name === item.image_name && a.question_id === item.question_id
                );
                if (modelMatch && modelMatch.question) {
                    fullPrompt = modelMatch.question;
                    break;
                }
            }

            // Clean the prompt safely
            let displayQuestion = fullPrompt;
            if (typeof fullPrompt === 'string' && fullPrompt.includes("Yes or No.")) {
                const parts = fullPrompt.split("Yes or No.");
                displayQuestion = parts[parts.length - 1].trim();
            }

            // Update Text & Image
            document.getElementById('large').src = "/img/" + item.image_name;
            document.getElementById('q-text').innerText = displayQuestion;
            document.getElementById('current-gt').innerText = item.answer.toUpperCase();
            document.getElementById('controls').style.display = 'block';
            
            // Update Counter
            document.getElementById('counter').innerText = `Item ${currentReviewPos + 1} / ${reviewIndices.length}`;

            // Update Sidebar Focus
            document.querySelectorAll('.thumb').forEach(img => img.classList.remove('active'));
            const activeThumb = document.getElementById('thumb-' + idx);
            if (activeThumb) {
                activeThumb.classList.add('active');
                activeThumb.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

            // Render Model Comparison
            let gridHtml = '';
            const norm = (s) => s.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"").trim();

            for (const [model, answers] of Object.entries(modelAnswers)) {
                const ans = answers.find(a => a.image_name === item.image_name && a.question_id === item.question_id);
                const answerText = ans ? ans.answer : 'N/A';
                const isDifferent = ans && norm(answerText) !== norm(item.answer);
                const style = isDifferent ? 'border: 2px solid #ff4444;' : 'border: 1px solid #555;';
                const textStyle = isDifferent ? 'color:#ff4444; font-weight:bold;' : '';
                
                gridHtml += `
                    <div class="card" style="${style}">
                        <strong>${model}</strong>
                        <p style="${textStyle}">${answerText}</p>
                    </div>`;
            }
            document.getElementById('model-grid').innerHTML = gridHtml;
        }

        function updateGT(val) {
            const globalIndex = reviewIndices[currentReviewPos];
            fetch('/update', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ index: globalIndex, answer: val })
            }).then(() => {
                gtData[globalIndex].answer = val;
                loadFrame(globalIndex);
                document.getElementById('thumb-' + globalIndex).style.borderColor = '#28a745';
            });
        }

        // Keyboard Navigation
        document.addEventListener('keydown', (e) => {
            if (reviewIndices.length === 0) return;

            if (e.key === 'ArrowRight') {
                loadByReviewPos((currentReviewPos + 1) % reviewIndices.length);
            } else if (e.key === 'ArrowLeft') {
                loadByReviewPos((currentReviewPos - 1 + reviewIndices.length) % reviewIndices.length);
            } else if (e.key === 'ArrowUp') {
                updateGT('yes');
            } else if (e.key === 'ArrowDown') {
                updateGT('no');
            }
        });

        window.onload = () => {
            console.log("Review Indices:", reviewIndices); // Debug check
            if (typeof reviewIndices !== 'undefined' && reviewIndices.length > 0) {
                loadByReviewPos(0);
            } else {
                document.getElementById('q-text').innerText = "No suspicious items found.";
            }
        };
    </script>
</body>
</html>
"""

import string

def normalize(text):
    """Standardizes answers: lowercases, strips whitespace, and removes punctuation."""
    if not text:
        return ""
    # Standardize 'no.' or 'No' to 'no'
    return text.lower().strip().translate(str.maketrans('', '', string.punctuation))

@app.route('/')
def index():
    # Only show frames where at least 3 models (75%) disagree with current GT
    review_indices = []
    
    for i, entry in enumerate(gt_data):
        current_gt = normalize(entry['answer'])
        disagreement_count = 0
        
        for m in MODELS:
            # Find matching model answer
            ans_obj = next((a for a in MODEL_ANSWERS[m] 
                           if a['image_name'] == entry['image_name'] 
                           and a['question_id'] == entry['question_id']), None)
            
            if ans_obj:
                if normalize(ans_obj['answer']) != current_gt:
                    disagreement_count += 1
        
        # Suggestion: Require at least 3 models to disagree before flagging for review
        if disagreement_count >= 3:
            review_indices.append(i)
            
    return render_template_string(
        HTML_TEMPLATE, 
        data=gt_data, 
        model_answers=MODEL_ANSWERS, 
        review_indices=review_indices,
        MODELS=MODELS 
    )

@app.route('/img/<filename>')
def get_img(filename):
    return send_from_directory(IMG_DIR, filename)

@app.route('/update', methods=['POST'])
def update():
    req = request.json
    idx = req['index']
    new_ans = req['answer']
    gt_data[idx]['answer'] = new_ans
    gt_data[idx]['timestamp'] = datetime.now().isoformat()
    
    # Save after every change for safety
    backup_path = GT_PATH.with_suffix('.jsonl.bak')
    if not backup_path.exists(): shutil.copy2(GT_PATH, backup_path)
    
    with open(GT_PATH, 'w') as f:
        for entry in gt_data:
            f.write(json.dumps(entry) + '\n')
        f.flush()
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)