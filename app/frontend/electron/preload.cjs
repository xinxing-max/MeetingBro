const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("meetingbro", {
  backendHttp: "http://127.0.0.1:8765",
  backendWs: "ws://127.0.0.1:8765",
});
