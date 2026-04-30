const { app, BrowserWindow, Menu, dialog, ipcMain, screen } = require("electron");
const path = require("node:path");

const isDev = !app.isPackaged;

function createWindow() {
  const { width: workWidth, height: workHeight } = screen.getPrimaryDisplay().workAreaSize;
  const width = Math.min(1680, Math.max(1280, Math.floor(workWidth * 0.92)));
  const height = Math.min(1000, Math.max(820, Math.floor(workHeight * 0.9)));

  const win = new BrowserWindow({
    width,
    height,
    minWidth: 1280,
    minHeight: 780,
    center: true,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.setMenu(null);
  win.setMenuBarVisibility(false);

  if (isDev) {
    win.loadURL("http://localhost:5173");
  } else {
    win.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }
}

app.whenReady().then(() => {
  Menu.setApplicationMenu(null);

  ipcMain.handle("meetingbro:select-export-directory", async (_event, suggestedName) => {
    const defaultName = suggestedName || `meetingbro_export_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}`;
    const result = await dialog.showSaveDialog({
      title: "Export meeting as folder",
      buttonLabel: "Use This Folder",
      defaultPath: path.join(app.getPath("documents"), defaultName),
      properties: ["createDirectory", "showOverwriteConfirmation"],
    });
    if (result.canceled || !result.filePath) {
      return null;
    }
    return result.filePath;
  });

  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
