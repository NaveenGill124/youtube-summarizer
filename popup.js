/* Popup JS: capture active tab's video id, call backend to ingest, then query. */
}
if(vid){
currentVideoId = vid;
statusEl.textContent = 'Video detected: ' + vid;
} else {
statusEl.textContent = 'No YouTube video id found on the active tab.';
}
return vid;
}catch(e){
statusEl.textContent = 'Error detecting tab: ' + e.message;
return null;
}
}


async function ingestVideo(videoId){
statusEl.textContent = 'Ingesting transcript & building index...';
try{
const resp = await fetch(backendBase + '/ingest', {
method: 'POST',
headers: {'Content-Type':'application/json'},
body: JSON.stringify({video_id: videoId})
});
const j = await resp.json();
if(resp.ok){
statusEl.textContent = 'Ingestion complete. Ready!';
appendMessage('Ingestion complete — you can ask questions now.');
} else {
statusEl.textContent = 'Ingestion failed: ' + (j.detail || JSON.stringify(j));
appendMessage('Ingestion failed: ' + (j.detail || JSON.stringify(j)));
}
}catch(e){
statusEl.textContent = 'Ingestion error: ' + e.message;
appendMessage('Ingestion error: ' + e.message);
}
}


async function queryBackend(question){
if(!currentVideoId){ appendMessage('No video selected.'); return; }
appendMessage(question, 'user');
appendMessage('Thinking...');
try{
const r = await fetch(backendBase + '/query', {
method: 'POST',
headers: {'Content-Type':'application/json'},
body: JSON.stringify({video_id: currentVideoId, question})
});
const j = await r.json();
// remove 'Thinking...' last bot bubble
const lastBot = Array.from(chatEl.querySelectorAll('.bot')).pop();
if(lastBot) lastBot.remove();
if(r.ok){
appendMessage(j.answer, 'bot');
} else {
appendMessage('Error: ' + (j.detail || JSON.stringify(j)), 'bot');
}
}catch(e){
appendMessage('Error querying backend: ' + e.message, 'bot');
}
}


// initial flow
(async ()=>{
const vid = await detectVideoIdFromActiveTab();
if(vid) await ingestVideo(vid);
})();


sendBtn.addEventListener('click', ()=>{
const q = inputEl.value.trim();
if(!q) return;
queryBackend(q);
inputEl.value = '';
});


inputEl.addEventListener('keydown', (e)=>{
if(e.key === 'Enter') sendBtn.click();
});