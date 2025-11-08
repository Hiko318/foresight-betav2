const path = require('path');
const fs = require('fs');
const initSqlJs = require('sql.js');
const { app } = require('electron');

let SQL = null;
let db = null;
let dbPathGlobal = null;

async function initDetectionDatabase() {
  const dbDir = app.getPath('userData');
  const dbPath = path.join(dbDir, 'foresight.db');
  dbPathGlobal = dbPath;

  if (!fs.existsSync(dbDir)) {
    fs.mkdirSync(dbDir, { recursive: true });
  }

  SQL = await initSqlJs({});

  if (fs.existsSync(dbPath)) {
    const fileBuffer = fs.readFileSync(dbPath);
    db = new SQL.Database(fileBuffer);
  } else {
    db = new SQL.Database();
  }

  db.run(`
    CREATE TABLE IF NOT EXISTS detection_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      type TEXT NOT NULL,
      timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_detection_logs_timestamp ON detection_logs(timestamp);
  `);

  flush();
  return { dbPath };
}

function logDetection(type, timestamp = new Date().toISOString()) {
  if (!db) return;
  const stmt = db.prepare('INSERT INTO detection_logs (type, timestamp) VALUES (?, ?)');
  stmt.run([type, timestamp]);
  stmt.free();
  flush();
}

function getRecentDetections(limit = 50) {
  if (!db) return [];
  const stmt = db.prepare('SELECT id, type, timestamp FROM detection_logs ORDER BY id DESC LIMIT ?');
  stmt.bind([limit]);
  const rows = [];
  while (stmt.step()) {
    const row = stmt.getAsObject();
    rows.push(row);
  }
  stmt.free();
  return rows;
}

function flush() {
  if (!db || !dbPathGlobal) return;
  const data = db.export();
  const buffer = Buffer.from(data);
  fs.writeFileSync(dbPathGlobal, buffer);
}

function closeDetectionDatabase() {
  if (db) {
    try { flush(); } catch (_) {}
    try { db.close(); } catch (_) {}
    db = null;
  }
}

module.exports = {
  initDetectionDatabase,
  logDetection,
  getRecentDetections,
  closeDetectionDatabase
};