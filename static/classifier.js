async function runClassify() {
  const text = editor.value;
  const words = getWords(text);
  if (words.length < 5) { showSaveMsg('Type at least 5 words first!', false); return; }

  const panel = document.getElementById('clf-panel');
  const msg   = document.getElementById('clf-msg');
  panel.style.display = 'block';
  msg.textContent = 'Classifying...';
  ['clf-class-name','clf-emoji','clf-range'].forEach(id => document.getElementById(id).textContent = '...');
  document.getElementById('clf-conf-bar').style.width = '0%';
  document.getElementById('clf-conf-pct').textContent = '0%';
  document.getElementById('clf-probs').innerHTML = '';

  const spaces = (text.match(/ /g) || []).length;
  try {
    const res = await fetch('/api/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        char_count: text.length, word_count: words.length,
        space_count: spaces, duration_seconds: Math.max(currentDuration, 1)
      })
    });
    const d = await res.json();
    if (d.success) {
      document.getElementById('clf-emoji').textContent       = d.emoji;
      document.getElementById('clf-class-name').textContent  = d.predicted_class;
      document.getElementById('clf-class-name').style.color  = d.color;
      document.getElementById('clf-range').textContent       = d.wpm_range;
      document.getElementById('clf-conf-bar').style.width    = d.confidence + '%';
      document.getElementById('clf-conf-bar').style.background = d.color;
      document.getElementById('clf-conf-pct').textContent    = d.confidence + '%';
      document.getElementById('clf-model-badge').textContent = d.model_used + ' - Speed Classifier';
      document.getElementById('clf-dot').style.background    = d.color;
      msg.textContent = '';

      const order  = ['Beginner','Average','Above Average','Proficient','Fast','Expert'];
      const colors = { 'Beginner':'#e25c5c','Average':'#e2945c','Above Average':'#e2d45c',
                       'Proficient':'#5ce27a','Fast':'#5cb8e2','Expert':'#a55ce2' };
      document.getElementById('clf-probs').innerHTML = order
        .filter(cls => d.class_probs[cls] !== undefined)
        .map(cls => {
          const pct = d.class_probs[cls] || 0;
          const col = colors[cls] || '#aaa';
          return '<div class="prob-row">' +
            '<span class="prob-label">' + cls + '</span>' +
            '<div class="prob-bar-wrap"><div class="prob-bar" style="width:' + pct + '%;background:' + col + '"></div></div>' +
            '<span class="prob-pct">' + pct + '%</span></div>';
        }).join('');
    } else {
      msg.textContent = 'Warning: ' + d.error;
      document.getElementById('clf-dot').style.background = '#e25c5c';
    }
  } catch(e) { msg.textContent = 'Error: Could not connect to server.'; }
}

async function trainClassifier() {
  const btn = document.getElementById('clf-train-btn');
  const el  = document.getElementById('clf-train-result');
  btn.textContent = 'Training...';
  btn.disabled = true;
  el.style.display = 'none';
  try {
    const res = await fetch('/api/train/classifier', { method: 'POST' });
    const d   = await res.json();
    el.style.display = 'block';
    if (d.success) {
      const knnAcc = d.cv_scores['KNN']          !== undefined ? (d.cv_scores['KNN']*100).toFixed(1)+'%'          : 'N/A';
      const rfAcc  = d.cv_scores['RandomForest'] !== undefined ? (d.cv_scores['RandomForest']*100).toFixed(1)+'%' : 'N/A';
      const dist   = Object.entries(d.class_dist).map(([k,v]) => k+'('+v+')').join(', ');
      el.innerHTML = '<div class="train-result ok">' +
        '<div class="tr-row"><span>Winner</span><span>' + d.winner_model + '</span></div>' +
        '<div class="tr-row"><span>KNN CV Accuracy</span><span>' + knnAcc + '</span></div>' +
        '<div class="tr-row"><span>RF CV Accuracy</span><span>' + rfAcc + '</span></div>' +
        '<div class="tr-row"><span>Sessions used</span><span>' + d.sessions_used + '</span></div>' +
        '<div class="tr-row"><span>Train Accuracy</span><span>' + (d.accuracy*100).toFixed(1) + '%</span></div>' +
        '<div class="tr-row"><span>Classes found</span><span style="font-size:11px">' + dist + '</span></div>' +
        '</div>';
    } else {
      el.innerHTML = '<div class="train-result err">Warning: ' + d.error + '</div>';
    }
  } catch(e) { el.innerHTML = '<div class="train-result err">Error: Server error.</div>'; }
  btn.textContent = 'Train';
  btn.disabled = false;
}
