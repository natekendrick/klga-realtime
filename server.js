import express from "express";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const app = express();
const PORT = process.env.PORT || 5177;
const execFileAsync = promisify(execFile);

const SYNOPTIC_TOKEN = "609187d0557b4819bf6c432d356faecf";

app.use(express.static(process.cwd(), { extensions: ["html"] }));

const apiCache = {};
const CACHE_TTL_MS = 60 * 1000;

async function getCachedOrFetch(cacheKey, url, headers = {}) {
  const now = Date.now();
  if (apiCache[cacheKey] && (now - apiCache[cacheKey].timestamp < CACHE_TTL_MS)) {
    return { status: 200, ok: true, body: apiCache[cacheKey].data };
  }
  const r = await fetch(url, { headers });
  const body = await r.text();
  if (r.ok) {
    apiCache[cacheKey] = { data: body, timestamp: now };
  }
  return { status: r.status, ok: r.ok, body };
}

function synopticTimeseriesUrl({ stid, recentMinutes }) {
  const vars = ["air_temp", "dew_point_temperature", "wind_speed", "wind_direction", "wind_gust", "altimeter", "visibility", "ceiling", "low_cloud_coverage", "mid_cloud_coverage", "high_cloud_coverage", "sky_cov_layer_base_1", "sky_cov_layer_base_2", "sky_cov_layer_base_3", "weather_cond_code"].join(",");
  return `https://api.synopticdata.com/v2/stations/timeseries?stid=${encodeURIComponent(stid)}&recent=${encodeURIComponent(String(recentMinutes))}&hfmetars=1&vars=${encodeURIComponent(vars)}&units=english,speed|mph&showemptyvars=1&token=${encodeURIComponent(SYNOPTIC_TOKEN)}`;
}

function synopticLatestUrl({ stid }) {
  return `https://api.synopticdata.com/v2/stations/latest?stid=${encodeURIComponent(stid)}&hfmetars=1&within=180&units=english,speed|mph&showemptyvars=1&token=${encodeURIComponent(SYNOPTIC_TOKEN)}`;
}

app.get("/api/hf_latest", async (req, res) => {
  const stid = (req.query.station || "KLGA1M").toString().toUpperCase();
  const url = synopticLatestUrl({ stid });
  try {
    const { status, ok, body } = await getCachedOrFetch(`latest_${stid}`, url, { "User-Agent": "klga-local" });
    if (!ok) return res.status(status).send(body);
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.send(body);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.get("/api/hf_96", async (req, res) => {
  const stid = (req.query.station || "KLGA1M").toString().toUpperCase();
  const url = synopticTimeseriesUrl({ stid, recentMinutes: 1440 });
  try {
    const { status, ok, body } = await getCachedOrFetch(`96h_${stid}`, url, { "User-Agent": "klga-local" });
    if (!ok) return res.status(status).send(body);
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.send(body);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.get("/api/hf_30d", async (req, res) => {
  const stid = (req.query.station || "KLGA1M").toString().toUpperCase();
  const url = synopticTimeseriesUrl({ stid, recentMinutes: 43200 });
  try {
    const { status, ok, body } = await getCachedOrFetch(`30d_${stid}`, url, { "User-Agent": "klga-local" });
    if (!ok) return res.status(status).send(body);
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.send(body);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.get("/api/wu_forecast", async (req, res) => {
  const url = "https://api.weather.com/v3/wx/forecast/hourly/2day?apiKey=e1f10a1e78da46f5b10a1e78da96f525&geocode=40.7769,-73.8740&format=json&units=e&language=en-US";
  try {
    const { status, ok, body } = await getCachedOrFetch("wu_forecast_klga", url, { 
        "Origin": "https://www.wunderground.com",
        "User-Agent": "Mozilla/5.0"
    });
    if (!ok) return res.status(status).send(body);
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.send(body);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.get("/api/predict/latest", async (req, res) => {
  const stid = (req.query.station || "KLGA1M").toString().toUpperCase();
  try {
    const { stdout } = await execFileAsync("python3", ["predict_klga.py", "--station", stid], {
      cwd: process.cwd(),
      timeout: 120000,
      maxBuffer: 5 * 1024 * 1024,
    });
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.send(stdout);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.listen(PORT, () => {
  console.log(`Open http://localhost:${PORT}`);
});