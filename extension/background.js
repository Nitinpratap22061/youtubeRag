let currentVideoId = null;

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "VIDEO_ID") {
    currentVideoId = message.videoId || null;
  }
  if (message?.type === "GET_VIDEO_ID") {
    sendResponse({ videoId: currentVideoId });
  }
});
