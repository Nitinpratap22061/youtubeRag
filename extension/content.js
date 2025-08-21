function getVideoId() {
  // standard watch page
  const params = new URLSearchParams(location.search);
  const v = params.get("v");
  if (v) return v;

  // shorts
  const m = location.pathname.match(/\/shorts\/([a-zA-Z0-9_-]+)/);
  if (m) return m[1];

  return null;
}

function sendVideoIdToBackground() {
  chrome.runtime.sendMessage({ type: "VIDEO_ID", videoId: getVideoId() });
}

// run once and whenever the page changes (YouTube SPA)
sendVideoIdToBackground();
const observer = new MutationObserver(() => sendVideoIdToBackground());
observer.observe(document.body, { childList: true, subtree: true });
