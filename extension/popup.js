const API_BASE = "http://127.0.0.1:8000";

const chat = document.getElementById("chat");
const vidNode = document.getElementById("vid");
const q = document.getElementById("q");
const askBtn = document.getElementById("ask");

function appendMsg(who, text) {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.innerHTML = `<span>${escapeHtml(text)}</span>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function escapeHtml(str) {
  return (str || "").replace(/[&<>"']/g, m => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;" }[m]));
}

function getVideoId() {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type: "GET_VIDEO_ID" }, (res) => {
      resolve(res?.videoId || null);
    });
  });
}

async function init() {
  const vid = await getVideoId();
  vidNode.textContent = vid || "Not on a video";
  if (!vid) appendMsg("bot", "Open a YouTube video page to start.");
}
init();

async function ask() {
  const question = q.value.trim();
  if (!question) return;
  const videoId = await getVideoId();
  if (!videoId) {
    appendMsg("bot", "No video detected. Open a YouTube video.");
    return;
  }

  q.value = "";
  appendMsg("you", question);
  appendMsg("bot", "…thinking…");

  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId, question })
    });

    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    chat.lastChild.remove(); // remove thinking
    appendMsg("bot", data.answer || "don't know");
  } catch (e) {
    chat.lastChild.remove();
    appendMsg("bot", `Error: ${e.message}`);
  }
}

askBtn.addEventListener("click", ask);
q.addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
