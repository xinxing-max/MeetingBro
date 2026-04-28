const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("meetingbro", {
  backendHttp: "http://127.0.0.1:8765",
  backendWs: "ws://127.0.0.1:8765",
  selectExportDirectory: (suggestedName) => ipcRenderer.invoke("meetingbro:select-export-directory", suggestedName),
});
