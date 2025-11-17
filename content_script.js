
// const BACKEND_URL = "http://localhost:8000";

// if (!window.__YT_SUMMARIZER_INJECTED__) {
//   window.__YT_SUMMARIZER_INJECTED__ = true;
//   initSummarizerOverlay();
// }

// function initSummarizerOverlay() {
//   const container = document.createElement('div');
//   container.id = 'yt-summarizer-root';
//   container.style.all = 'initial';
//   document.documentElement.appendChild(container);
//   const shadow = container.attachShadow({ mode: 'open' });

//   // =========================
//   // Enhanced Styles
//   // =========================
//   const style = document.createElement('style');
//   style.textContent = `
//     * {
//       box-sizing: border-box;
//     }
//     .float-btn {
//       position: fixed;
//       right: 24px;
//       bottom: 24px;
//       width: 56px;
//       height: 56px;
//       border-radius: 50%;
//       box-shadow: 0 6px 20px rgba(0,0,0,0.25);
//       background: linear-gradient(135deg, #4f46e5, #06b6d4);
//       color: white;
//       display: flex;
//       align-items: center;
//       justify-content: center;
//       font-weight: bold;
//       font-size: 22px;
//       cursor: pointer;
//       z-index: 2147483647;
//       transition: transform 0.2s;
//     }
//     .float-btn:hover {
//       transform: scale(1.05);
//     }
//     .badge {
//       position: absolute;
//       top: -6px;
//       right: -6px;
//       background: #ff4757;
//       color: white;
//       font-size: 11px;
//       padding: 2px 6px;
//       border-radius: 999px;
//       font-weight: 600;
//     }
//     .overlay {
//       position: fixed;
//       right: 24px;
//       bottom: 92px;
//       width: 480px;
//       max-height: 600px;
//       background: #ffffff;
//       border-radius: 16px;
//       box-shadow: 0 12px 40px rgba(0,0,0,0.4);
//       display: flex;
//       flex-direction: column;
//       overflow: hidden;
//       z-index: 2147483647;
//       font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Arial;
//     }
//     .header {
//       display: flex;
//       align-items: center;
//       justify-content: space-between;
//       background: linear-gradient(90deg, #4f46e5, #06b6d4);
//       color: white;
//       padding: 12px 16px;
//     }
//     .header h3 {
//       margin: 0;
//       font-size: 15px;
//       font-weight: 600;
//     }
//     .actions {
//       display: flex;
//       gap: 8px;
//     }
//     .btn-ghost {
//       background: transparent;
//       border: 1px solid rgba(255,255,255,0.4);
//       color: white;
//       font-size: 11px;
//       padding: 5px 10px;
//       border-radius: 6px;
//       cursor: pointer;
//       transition: background 0.2s;
//       font-weight: 500;
//     }
//     .btn-ghost:hover {
//       background: rgba(255,255,255,0.1);
//     }
//     .status {
//       padding: 8px 12px;
//       background: #f0f9ff;
//       border-bottom: 1px solid #e0e7ff;
//       font-size: 12px;
//       color: #1e40af;
//     }
//     .chat {
//       flex: 1;
//       background: #f9fafb;
//       overflow-y: auto;
//       padding: 16px;
//     }
//     .msg {
//       margin: 12px 0;
//       display: flex;
//       animation: slideIn 0.2s ease-out;
//     }
//     @keyframes slideIn {
//       from { opacity: 0; transform: translateY(10px); }
//       to { opacity: 1; transform: translateY(0); }
//     }
//     .msg.user {
//       justify-content: flex-end;
//     }
//     .bubble {
//       padding: 12px 14px;
//       border-radius: 12px;
//       max-width: 85%;
//       word-wrap: break-word;
//       font-size: 14px;
//       line-height: 1.6;
//     }
//     .msg.user .bubble {
//       background: #e6fffa;
//       border-bottom-right-radius: 4px;
//     }
//     .msg.bot .bubble {
//       background: white;
//       border-bottom-left-radius: 4px;
//       box-shadow: 0 1px 2px rgba(0,0,0,0.05);
//     }
//     .msg.bot .bubble.thinking {
//       background: #fef3c7;
//       font-style: italic;
//       color: #78350f;
//     }
    
//     /* Enhanced answer formatting */
//     .bubble h3 {
//       font-size: 15px;
//       font-weight: 600;
//       color: #1e293b;
//       margin: 0 0 8px 0;
//       padding-bottom: 6px;
//       border-bottom: 2px solid #e2e8f0;
//     }
//     .bubble h4 {
//       font-size: 14px;
//       font-weight: 600;
//       color: #475569;
//       margin: 12px 0 6px 0;
//     }
//     .bubble ul, .bubble ol {
//       margin: 8px 0;
//       padding-left: 20px;
//     }
//     .bubble li {
//       margin: 6px 0;
//       line-height: 1.6;
//     }
//     .bubble p {
//       margin: 8px 0;
//     }
//     .bubble strong {
//       color: #1e293b;
//       font-weight: 600;
//     }
//     .bubble em {
//       color: #64748b;
//       font-style: italic;
//     }
//     .bubble code {
//       background: #f1f5f9;
//       padding: 2px 6px;
//       border-radius: 4px;
//       font-family: 'Courier New', monospace;
//       font-size: 13px;
//     }
//     .timestamp {
//       display: inline-block;
//       background: #e0e7ff;
//       color: #4338ca;
//       padding: 2px 8px;
//       border-radius: 12px;
//       font-size: 12px;
//       font-weight: 500;
//       margin: 0 4px;
//       cursor: pointer;
//       transition: background 0.2s;
//     }
//     .timestamp:hover {
//       background: #c7d2fe;
//     }
    
//     .controls {
//       display: flex;
//       border-top: 1px solid #e5e7eb;
//       padding: 12px;
//       gap: 8px;
//       background: white;
//     }
//     .controls input {
//       flex: 1;
//       padding: 10px 12px;
//       border-radius: 8px;
//       border: 1px solid #d1d5db;
//       font-size: 14px;
//       outline: none;
//       font-family: inherit;
//     }
//     .controls input:focus {
//       border-color: #4f46e5;
//       box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
//     }
//     .controls button {
//       background: #4f46e5;
//       border: none;
//       color: white;
//       border-radius: 8px;
//       padding: 10px 16px;
//       cursor: pointer;
//       font-weight: 600;
//       transition: background 0.2s;
//     }
//     .controls button:hover {
//       background: #4338ca;
//     }
//     .controls button:disabled {
//       background: #9ca3af;
//       cursor: not-allowed;
//     }
//   `;
//   shadow.appendChild(style);

//   // =========================
//   // Floating button
//   // =========================
//   const floatBtn = document.createElement('div');
//   floatBtn.className = 'float-btn';
//   floatBtn.textContent = 'Y';
//   const badge = document.createElement('div');
//   badge.className = 'badge';
//   badge.textContent = 'AI';
//   floatBtn.appendChild(badge);
//   shadow.appendChild(floatBtn);

//   // =========================
//   // Chat overlay container
//   // =========================
//   const overlay = document.createElement('div');
//   overlay.className = 'overlay';
//   overlay.style.display = 'none';
//   overlay.innerHTML = `
//     <div class="header">
//       <h3>YouTube Summarizer</h3>
//       <div class="actions">
//         <button class="btn-ghost" id="summaryBtn">Summary</button>
//         <button class="btn-ghost" id="keypointsBtn">Key Points</button>
//         <button class="btn-ghost" id="reindexBtn">Re-index</button>
//         <button class="btn-ghost" id="closeBtn">✕</button>
//       </div>
//     </div>
//     <div class="status" id="statusBar">Ready to answer questions</div>
//     <div class="chat" id="chatArea"></div>
//     <div class="controls">
//       <input id="queryInput" placeholder="Ask something about this video..." />
//       <button id="sendBtn">Send</button>
//     </div>
//   `;
//   shadow.appendChild(overlay);

//   // =========================
//   // Elements
//   // =========================
//   const chat = overlay.querySelector('#chatArea');
//   const sendBtn = overlay.querySelector('#sendBtn');
//   const queryInput = overlay.querySelector('#queryInput');
//   const closeBtn = overlay.querySelector('#closeBtn');
//   const summaryBtn = overlay.querySelector('#summaryBtn');
//   const keypointsBtn = overlay.querySelector('#keypointsBtn');
//   const reindexBtn = overlay.querySelector('#reindexBtn');
//   const statusBar = overlay.querySelector('#statusBar');

//   // =========================
//   // State
//   // =========================
//   let currentVideoId = null;
//   let isIndexed = false;
//   let isProcessing = false;

//   // =========================
//   // CRITICAL: Stop event propagation
//   // =========================
//   queryInput.addEventListener('keydown', (e) => e.stopPropagation());
//   queryInput.addEventListener('keyup', (e) => e.stopPropagation());
//   queryInput.addEventListener('keypress', (e) => e.stopPropagation());
//   overlay.addEventListener('mousedown', (e) => e.stopPropagation());
//   overlay.addEventListener('click', (e) => e.stopPropagation());

//   // =========================
//   // Helper: Get current video ID
//   // =========================
//   function getCurrentVideoId() {
//     const urlParams = new URLSearchParams(window.location.search);
//     return urlParams.get('v');
//   }

//   // =========================
//   // Helper: Update status
//   // =========================
//   function updateStatus(message, type = 'info') {
//     statusBar.textContent = message;
//     statusBar.style.background = type === 'error' ? '#fee2e2' : 
//                                   type === 'success' ? '#d1fae5' : '#f0f9ff';
//     statusBar.style.color = type === 'error' ? '#991b1b' : 
//                            type === 'success' ? '#065f46' : '#1e40af';
//   }

//   // =========================
//   // Helper: Format markdown-like text to HTML
//   // =========================
//   function formatAnswerToHTML(text) {
//     let html = text;
    
//     // Convert ### headers to h3
//     html = html.replace(/### (.*?)(\n|$)/g, '<h3>$1</h3>');
    
//     // Convert #### headers to h4
//     html = html.replace(/#### (.*?)(\n|$)/g, '<h4>$1</h4>');
    
//     // Convert **bold** to strong
//     html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
//     // Convert *italic* to em
//     html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
//     // Convert numbered lists
//     html = html.replace(/^\d+\.\s(.+)$/gm, '<li>$1</li>');
//     html = html.replace(/(<li>.*<\/li>)/s, '<ol>$1</ol>');
    
//     // Convert bullet lists
//     html = html.replace(/^[-•]\s(.+)$/gm, '<li>$1</li>');
    
//     // Convert timestamps [MM:SS] to clickable elements
//     html = html.replace(/\[?(\d+:\d+)\]?/g, '<span class="timestamp" data-time="$1">$1</span>');
    
//     // Convert line breaks to paragraphs
//     html = html.split('\n\n').map(p => p.trim() ? `<p>${p}</p>` : '').join('');
    
//     return html;
//   }

//   // =========================
//   // Helper: Append message with formatting
//   // =========================
//   function appendMessage(text, who = 'bot', isThinking = false) {
//     const msg = document.createElement('div');
//     msg.className = `msg ${who}`;
//     const bubble = document.createElement('div');
//     bubble.className = isThinking ? 'bubble thinking' : 'bubble';
    
//     if (who === 'bot' && !isThinking) {
//       bubble.innerHTML = formatAnswerToHTML(text);
      
//       // Add click handlers for timestamps
//       bubble.querySelectorAll('.timestamp').forEach(ts => {
//         ts.addEventListener('click', () => {
//           const time = ts.getAttribute('data-time');
//           const [minutes, seconds] = time.split(':').map(Number);
//           const totalSeconds = minutes * 60 + seconds;
          
//           // Try to seek the video
//           const video = document.querySelector('video');
//           if (video) {
//             video.currentTime = totalSeconds;
//             video.play();
//           }
//         });
//       });
//     } else {
//       bubble.textContent = text;
//     }
    
//     msg.appendChild(bubble);
//     chat.appendChild(msg);
//     chat.scrollTop = chat.scrollHeight;
//     return bubble;
//   }

//   // =========================
//   // Helper: Remove last bot message
//   // =========================
//   function removeLastBotMessage() {
//     const botMessages = chat.querySelectorAll('.msg.bot');
//     if (botMessages.length > 0) {
//       botMessages[botMessages.length - 1].remove();
//     }
//   }

//   // =========================
//   // Function: Ingest video
//   // =========================
//   async function ingestVideo(videoId) {
//     updateStatus('Indexing transcript...', 'info');
//     try {
//       const response = await fetch(`${BACKEND_URL}/ingest`, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ video_id: videoId })
//       });

//       const data = await response.json();

//       if (response.ok) {
//         isIndexed = true;
//         updateStatus(`✓ Indexed ${data.chunks || 0} chunks`, 'success');
//         return true;
//       } else {
//         updateStatus('Failed to index transcript', 'error');
//         appendMessage(`Error: ${data.detail || 'Failed to index'}`, 'bot');
//         return false;
//       }
//     } catch (e) {
//       updateStatus('Backend connection error', 'error');
//       appendMessage(`Connection error: ${e.message}`, 'bot');
//       return false;
//     }
//   }

//   // =========================
//   // Function: Query backend
//   // =========================
//   async function queryBackend(question) {
//     if (!currentVideoId) {
//       appendMessage('No video detected. Please open a YouTube video.', 'bot');
//       return;
//     }

//     if (isProcessing) return;

//     isProcessing = true;
//     sendBtn.disabled = true;
//     queryInput.disabled = true;

//     appendMessage(question, 'user');
//     const thinkingBubble = appendMessage('Thinking...', 'bot', true);

//     try {
//       if (!isIndexed) {
//         const indexed = await ingestVideo(currentVideoId);
//         if (!indexed) {
//           removeLastBotMessage();
//           isProcessing = false;
//           sendBtn.disabled = false;
//           queryInput.disabled = false;
//           return;
//         }
//       }

//       const response = await fetch(`${BACKEND_URL}/query`, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ 
//           video_id: currentVideoId, 
//           question: question 
//         })
//       });

//       const data = await response.json();
//       removeLastBotMessage();

//       if (response.ok) {
//         appendMessage(data.answer, 'bot');
//         updateStatus('Ready', 'success');
//       } else {
//         appendMessage(`Error: ${data.detail || 'Query failed'}`, 'bot');
//         updateStatus('Query failed', 'error');
//       }
//     } catch (e) {
//       removeLastBotMessage();
//       appendMessage(`Error: ${e.message}`, 'bot');
//       updateStatus('Connection error', 'error');
//     } finally {
//       isProcessing = false;
//       sendBtn.disabled = false;
//       queryInput.disabled = false;
//       queryInput.focus();
//     }
//   }

//   // =========================
//   // Predefined queries
//   // =========================
//   async function generateSummary() {
//     await queryBackend('Please provide a comprehensive summary of this video, including the main topic, key speakers, and important takeaways. Format it with clear sections.');
//   }

//   async function generateKeyPoints() {
//     await queryBackend('Write detailed key points and learnings from this podcast/video. Include: 1) Main themes, 2) Key speakers and their contributions, 3) Important concepts discussed with timestamps, 4) Actionable learnings and takeaways. Format it clearly with headers and bullet points.');
//   }

//   // =========================
//   // Event: Toggle overlay
//   // =========================
//   floatBtn.addEventListener('click', () => {
//     const isVisible = overlay.style.display === 'flex';
//     overlay.style.display = isVisible ? 'none' : 'flex';
    
//     if (!isVisible) {
//       const vid = getCurrentVideoId();
//       if (vid !== currentVideoId) {
//         currentVideoId = vid;
//         isIndexed = false;
//         chat.innerHTML = '';
        
//         if (vid) {
//           updateStatus(`Video detected: ${vid}`, 'info');
//           appendMessage('👋 Hi! Ask me anything about this video, or click "Key Points" for a detailed breakdown.', 'bot');
//         } else {
//           updateStatus('No video detected', 'error');
//           appendMessage('Please open a YouTube video page.', 'bot');
//         }
//       }
//       queryInput.focus();
//     }
//   });

//   closeBtn.addEventListener('click', () => {
//     overlay.style.display = 'none';
//   });

//   sendBtn.addEventListener('click', () => {
//     const question = queryInput.value.trim();
//     if (!question || isProcessing) return;
//     queryBackend(question);
//     queryInput.value = '';
//   });

//   queryInput.addEventListener('keydown', (e) => {
//     if (e.key === 'Enter' && !e.shiftKey) {
//       e.preventDefault();
//       e.stopPropagation();
//       sendBtn.click();
//     }
//   });

//   summaryBtn.addEventListener('click', () => {
//     if (!isProcessing) generateSummary();
//   });

//   keypointsBtn.addEventListener('click', () => {
//     if (!isProcessing) generateKeyPoints();
//   });

//   reindexBtn.addEventListener('click', async () => {
//     if (isProcessing || !currentVideoId) return;
//     isIndexed = false;
//     appendMessage('Re-indexing transcript...', 'bot');
//     const success = await ingestVideo(currentVideoId);
//     if (success) {
//       appendMessage('✓ Re-indexing complete!', 'bot');
//     }
//   });

//   // Detect video changes
//   let lastUrl = location.href;
//   new MutationObserver(() => {
//     const currentUrl = location.href;
//     if (currentUrl !== lastUrl) {
//       lastUrl = currentUrl;
//       const newVid = getCurrentVideoId();
      
//       if (newVid && newVid !== currentVideoId) {
//         currentVideoId = newVid;
//         isIndexed = false;
        
//         if (overlay.style.display === 'flex') {
//           chat.innerHTML = '';
//           updateStatus(`New video detected: ${newVid}`, 'info');
//           appendMessage('👋 New video detected! Ask me anything.', 'bot');
//         }
//       }
//     }
//   }).observe(document, { subtree: true, childList: true });
// }


// content_script.js
// YouTube Summarizer Overlay - Production Version with Beautiful Formatting

const BACKEND_URL = "http://localhost:8000";

if (!window.__YT_SUMMARIZER_INJECTED__) {
  window.__YT_SUMMARIZER_INJECTED__ = true;
  initSummarizerOverlay();
}

function initSummarizerOverlay() {
  const container = document.createElement('div');
  container.id = 'yt-summarizer-root';
  container.style.all = 'initial';
  document.documentElement.appendChild(container);
  const shadow = container.attachShadow({ mode: 'open' });

  // =========================
  // Enhanced Styles
  // =========================
  const style = document.createElement('style');
  style.textContent = `
    * {
      box-sizing: border-box;
    }
    .float-btn {
      position: fixed;
      right: 24px;
      bottom: 24px;
      width: 56px;
      height: 56px;
      border-radius: 50%;
      box-shadow: 0 6px 20px rgba(0,0,0,0.25);
      background: linear-gradient(135deg, #4f46e5, #06b6d4);
      color: white;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: bold;
      font-size: 22px;
      cursor: pointer;
      z-index: 2147483647;
      transition: transform 0.2s;
    }
    .float-btn:hover {
      transform: scale(1.05);
    }
    .badge {
      position: absolute;
      top: -6px;
      right: -6px;
      background: #ff4757;
      color: white;
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 999px;
      font-weight: 600;
    }
    .overlay {
      position: fixed;
      right: 24px;
      bottom: 92px;
      width: 480px;
      max-height: 600px;
      background: #ffffff;
      border-radius: 16px;
      box-shadow: 0 12px 40px rgba(0,0,0,0.4);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      z-index: 2147483647;
      font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Arial;
    }
    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: linear-gradient(90deg, #4f46e5, #06b6d4);
      color: white;
      padding: 12px 16px;
    }
    .header h3 {
      margin: 0;
      font-size: 15px;
      font-weight: 600;
    }
    .actions {
      display: flex;
      gap: 8px;
    }
    .btn-ghost {
      background: transparent;
      border: 1px solid rgba(255,255,255,0.4);
      color: white;
      font-size: 11px;
      padding: 5px 10px;
      border-radius: 6px;
      cursor: pointer;
      transition: background 0.2s;
      font-weight: 500;
    }
    .btn-ghost:hover {
      background: rgba(255,255,255,0.1);
    }
    .status {
      padding: 8px 12px;
      background: #f0fdf4;
      border-bottom: 1px solid #dcfce7;
      font-size: 12px;
      color: #15803d;
      font-weight: 500;
    }
    .chat {
      flex: 1;
      background: #f9fafb;
      overflow-y: auto;
      padding: 16px;
      scroll-behavior: smooth;
    }
    .chat::-webkit-scrollbar {
      width: 8px;
    }
    .chat::-webkit-scrollbar-track {
      background: #f1f5f9;
    }
    .chat::-webkit-scrollbar-thumb {
      background: #cbd5e1;
      border-radius: 4px;
    }
    .chat::-webkit-scrollbar-thumb:hover {
      background: #94a3b8;
    }
    .msg {
      margin: 12px 0;
      display: flex;
      animation: slideIn 0.2s ease-out;
    }
    @keyframes slideIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .msg.user {
      justify-content: flex-end;
    }
    .bubble {
      padding: 12px 14px;
      border-radius: 12px;
      max-width: 85%;
      word-wrap: break-word;
      font-size: 14px;
      line-height: 1.6;
    }
    .msg.user .bubble {
      background: #e6fffa;
      border-bottom-right-radius: 4px;
    }
    .msg.bot .bubble {
      background: white;
      border-bottom-left-radius: 4px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .msg.bot .bubble.thinking {
      background: #fef3c7;
      font-style: italic;
      color: #78350f;
    }
    
    /* Enhanced answer formatting */
    .bubble h3 {
      font-size: 15px;
      font-weight: 600;
      color: #1e293b;
      margin: 0 0 8px 0;
      padding-bottom: 6px;
      border-bottom: 2px solid #e2e8f0;
    }
    .bubble h4 {
      font-size: 14px;
      font-weight: 600;
      color: #475569;
      margin: 12px 0 6px 0;
    }
    .bubble ul, .bubble ol {
      margin: 8px 0;
      padding-left: 20px;
    }
    .bubble li {
      margin: 6px 0;
      line-height: 1.6;
    }
    .bubble p {
      margin: 8px 0;
    }
    .bubble strong {
      color: #1e293b;
      font-weight: 600;
    }
    .bubble em {
      color: #64748b;
      font-style: italic;
    }
    .bubble code {
      background: #f1f5f9;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: 'Courier New', monospace;
      font-size: 13px;
    }
    .timestamp {
      display: inline-block;
      background: #e0e7ff;
      color: #4338ca;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 500;
      margin: 0 4px;
      cursor: pointer;
      transition: background 0.2s;
    }
    .timestamp:hover {
      background: #c7d2fe;
    }
    
    .controls {
      display: flex;
      border-top: 1px solid #e5e7eb;
      padding: 12px;
      gap: 8px;
      background: white;
    }
    .controls input {
      flex: 1;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid #d1d5db;
      font-size: 14px;
      outline: none;
      font-family: inherit;
    }
    .controls input:focus {
      border-color: #4f46e5;
      box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }
    .controls button {
      background: #4f46e5;
      border: none;
      color: white;
      border-radius: 8px;
      padding: 10px 16px;
      cursor: pointer;
      font-weight: 600;
      transition: background 0.2s;
    }
    .controls button:hover {
      background: #4338ca;
    }
    .controls button:disabled {
      background: #9ca3af;
      cursor: not-allowed;
    }
  `;
  shadow.appendChild(style);

  // =========================
  // Floating button
  // =========================
  const floatBtn = document.createElement('div');
  floatBtn.className = 'float-btn';
  floatBtn.textContent = 'Y';
  const badge = document.createElement('div');
  badge.className = 'badge';
  badge.textContent = 'AI';
  floatBtn.appendChild(badge);
  shadow.appendChild(floatBtn);

  // =========================
  // Chat overlay container
  // =========================
  const overlay = document.createElement('div');
  overlay.className = 'overlay';
  overlay.style.display = 'none';
  overlay.innerHTML = `
    <div class="header">
      <h3>YouTube Summarizer</h3>
      <div class="actions">
        <button class="btn-ghost" id="summaryBtn">Summary</button>
        <button class="btn-ghost" id="keypointsBtn">Key Points</button>
        <button class="btn-ghost" id="reindexBtn">Re-index</button>
        <button class="btn-ghost" id="closeBtn">✕</button>
      </div>
    </div>
    <div class="status" id="statusBar">Ready to answer questions</div>
    <div class="chat" id="chatArea"></div>
    <div class="controls">
      <input id="queryInput" placeholder="Ask something about this video..." />
      <button id="sendBtn">Send</button>
    </div>
  `;
  shadow.appendChild(overlay);

  // =========================
  // Elements
  // =========================
  const chat = overlay.querySelector('#chatArea');
  const sendBtn = overlay.querySelector('#sendBtn');
  const queryInput = overlay.querySelector('#queryInput');
  const closeBtn = overlay.querySelector('#closeBtn');
  const summaryBtn = overlay.querySelector('#summaryBtn');
  const keypointsBtn = overlay.querySelector('#keypointsBtn');
  const reindexBtn = overlay.querySelector('#reindexBtn');
  const statusBar = overlay.querySelector('#statusBar');

  // =========================
  // State
  // =========================
  let currentVideoId = null;
  let isIndexed = false;
  let isProcessing = false;

  // =========================
  // CRITICAL: Stop event propagation
  // =========================
  queryInput.addEventListener('keydown', (e) => e.stopPropagation());
  queryInput.addEventListener('keyup', (e) => e.stopPropagation());
  queryInput.addEventListener('keypress', (e) => e.stopPropagation());
  overlay.addEventListener('mousedown', (e) => e.stopPropagation());
  overlay.addEventListener('click', (e) => e.stopPropagation());

  // =========================
  // Helper: Get current video ID
  // =========================
  function getCurrentVideoId() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('v');
  }

  // =========================
  // Helper: Update status
  // =========================
  function updateStatus(message, type = 'info') {
    statusBar.textContent = message;
    statusBar.style.background = type === 'error' ? '#fee2e2' : 
                                  type === 'success' ? '#d1fae5' : '#f0f9ff';
    statusBar.style.color = type === 'error' ? '#991b1b' : 
                           type === 'success' ? '#065f46' : '#1e40af';
  }

  // =========================
  // Helper: Format markdown-like text to HTML
  // =========================
  function formatAnswerToHTML(text) {
    let html = text;
    
    // Convert ### headers to h3
    html = html.replace(/### (.*?)(\n|$)/g, '<h3>$1</h3>');
    
    // Convert #### headers to h4
    html = html.replace(/#### (.*?)(\n|$)/g, '<h4>$1</h4>');
    
    // Convert **bold** to strong
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Convert *italic* to em
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Convert numbered lists
    html = html.replace(/^\d+\.\s(.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ol>$1</ol>');
    
    // Convert bullet lists
    html = html.replace(/^[-•]\s(.+)$/gm, '<li>$1</li>');
    
    // Convert timestamps [MM:SS] to clickable elements
    html = html.replace(/\[?(\d+:\d+)\]?/g, '<span class="timestamp" data-time="$1">$1</span>');
    
    // Convert line breaks to paragraphs
    html = html.split('\n\n').map(p => p.trim() ? `<p>${p}</p>` : '').join('');
    
    return html;
  }

  // =========================
  // Helper: Append message with formatting
  // =========================
  function appendMessage(text, who = 'bot', isThinking = false) {
    const msg = document.createElement('div');
    msg.className = `msg ${who}`;
    const bubble = document.createElement('div');
    bubble.className = isThinking ? 'bubble thinking' : 'bubble';
    
    if (who === 'bot' && !isThinking) {
      bubble.innerHTML = formatAnswerToHTML(text);
      
      // Add click handlers for timestamps
      bubble.querySelectorAll('.timestamp').forEach(ts => {
        ts.addEventListener('click', () => {
          const time = ts.getAttribute('data-time');
          const [minutes, seconds] = time.split(':').map(Number);
          const totalSeconds = minutes * 60 + seconds;
          
          // Try to seek the video
          const video = document.querySelector('video');
          if (video) {
            video.currentTime = totalSeconds;
            video.play();
          }
        });
      });
    } else {
      bubble.textContent = text;
    }
    
    msg.appendChild(bubble);
    chat.appendChild(msg);
    chat.scrollTop = chat.scrollHeight;
    return bubble;
  }

  // =========================
  // Helper: Remove last bot message
  // =========================
  function removeLastBotMessage() {
    const botMessages = chat.querySelectorAll('.msg.bot');
    if (botMessages.length > 0) {
      botMessages[botMessages.length - 1].remove();
    }
  }

  // =========================
  // Function: Ingest video
  // =========================
  async function ingestVideo(videoId) {
    updateStatus('Indexing transcript...', 'info');
    try {
      const response = await fetch(`${BACKEND_URL}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId })
      });

      const data = await response.json();

      if (response.ok) {
        isIndexed = true;
        updateStatus(`✓ Indexed ${data.chunks || 0} chunks`, 'success');
        return true;
      } else {
        updateStatus('Failed to index transcript', 'error');
        appendMessage(`Error: ${data.detail || 'Failed to index'}`, 'bot');
        return false;
      }
    } catch (e) {
      updateStatus('Backend connection error', 'error');
      appendMessage(`Connection error: ${e.message}`, 'bot');
      return false;
    }
  }

  // =========================
  // Function: Query backend
  // =========================
  async function queryBackend(question) {
    if (!currentVideoId) {
      appendMessage('No video detected. Please open a YouTube video.', 'bot');
      return;
    }

    if (isProcessing) return;

    isProcessing = true;
    sendBtn.disabled = true;
    queryInput.disabled = true;

    appendMessage(question, 'user');
    const thinkingBubble = appendMessage('Thinking...', 'bot', true);

    try {
      if (!isIndexed) {
        const indexed = await ingestVideo(currentVideoId);
        if (!indexed) {
          removeLastBotMessage();
          isProcessing = false;
          sendBtn.disabled = false;
          queryInput.disabled = false;
          return;
        }
      }

      const response = await fetch(`${BACKEND_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          video_id: currentVideoId, 
          question: question 
        })
      });

      const data = await response.json();
      removeLastBotMessage();

      if (response.ok) {
        appendMessage(data.answer, 'bot');
        updateStatus('Ready', 'success');
      } else {
        appendMessage(`Error: ${data.detail || 'Query failed'}`, 'bot');
        updateStatus('Query failed', 'error');
      }
    } catch (e) {
      removeLastBotMessage();
      appendMessage(`Error: ${e.message}`, 'bot');
      updateStatus('Connection error', 'error');
    } finally {
      isProcessing = false;
      sendBtn.disabled = false;
      queryInput.disabled = false;
      queryInput.focus();
    }
  }

  // =========================
  // Predefined queries
  // =========================
  async function generateSummary() {
    await queryBackend('Please provide a comprehensive summary of this video, including the main topic, key speakers, and important takeaways. Format it with clear sections.');
  }

  async function generateKeyPoints() {
    await queryBackend('Write detailed key points and learnings from this podcast/video. Include: 1) Main themes, 2) Key speakers and their contributions, 3) Important concepts discussed with timestamps, 4) Actionable learnings and takeaways. Format it clearly with headers and bullet points.');
  }

  // =========================
  // Event: Toggle overlay
  // =========================
  floatBtn.addEventListener('click', () => {
    const isVisible = overlay.style.display === 'flex';
    overlay.style.display = isVisible ? 'none' : 'flex';
    
    if (!isVisible) {
      const vid = getCurrentVideoId();
      if (vid !== currentVideoId) {
        currentVideoId = vid;
        isIndexed = false;
        chat.innerHTML = '';
        
        if (vid) {
          updateStatus(`Video detected: ${vid}`, 'info');
          appendMessage('👋 Hi! Ask me anything about this video, or click "Key Points" for a detailed breakdown.', 'bot');
        } else {
          updateStatus('No video detected', 'error');
          appendMessage('Please open a YouTube video page.', 'bot');
        }
      }
      queryInput.focus();
    }
  });

  closeBtn.addEventListener('click', () => {
    overlay.style.display = 'none';
  });

  sendBtn.addEventListener('click', () => {
    const question = queryInput.value.trim();
    if (!question || isProcessing) return;
    queryBackend(question);
    queryInput.value = '';
  });

  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      e.stopPropagation();
      sendBtn.click();
    }
  });

  summaryBtn.addEventListener('click', () => {
    if (!isProcessing) generateSummary();
  });

  keypointsBtn.addEventListener('click', () => {
    if (!isProcessing) generateKeyPoints();
  });

  reindexBtn.addEventListener('click', async () => {
    if (isProcessing || !currentVideoId) return;
    isIndexed = false;
    appendMessage('Re-indexing transcript...', 'bot');
    const success = await ingestVideo(currentVideoId);
    if (success) {
      appendMessage('✓ Re-indexing complete!', 'bot');
    }
  });

  // Detect video changes
  let lastUrl = location.href;
  new MutationObserver(() => {
    const currentUrl = location.href;
    if (currentUrl !== lastUrl) {
      lastUrl = currentUrl;
      const newVid = getCurrentVideoId();
      
      if (newVid && newVid !== currentVideoId) {
        currentVideoId = newVid;
        isIndexed = false;
        
        if (overlay.style.display === 'flex') {
          chat.innerHTML = '';
          updateStatus(`New video detected: ${newVid}`, 'info');
          appendMessage('👋 New video detected! Ask me anything.', 'bot');
        }
      }
    }
  }).observe(document, { subtree: true, childList: true });
}