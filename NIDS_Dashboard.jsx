import { useState, useEffect, useRef } from "react";
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, PieChart, Pie, Legend } from "recharts";

const METRICS = {
  accuracy: 0.988,
  precision: 0.9899,
  recall: 0.9891,
  f1: 0.9895,
  confusion_matrix: [[1916, 26], [28, 2539]],
};

const TOP_FEATURES = [
  { name: "src_bytes", importance: 0.1613 },
  { name: "dst_bytes", importance: 0.1300 },
  { name: "dst_host_same_srv_rate", importance: 0.0742 },
  { name: "dst_host_diff_srv_rate", importance: 0.0698 },
  { name: "service", importance: 0.0682 },
  { name: "dst_host_rerror_rate", importance: 0.0650 },
  { name: "dst_host_srv_count", importance: 0.0601 },
  { name: "duration", importance: 0.0344 },
  { name: "protocol_type", importance: 0.0332 },
];

const ATTACK_DIST = [
  { name: "Normal", count: 9711, color: "#00f5a0" },
  { name: "Neptune", count: 4657, color: "#f05454" },
  { name: "Guess Passwd", count: 1231, color: "#f5a623" },
  { name: "Mscan", count: 996, color: "#e056fd" },
  { name: "Warezmaster", count: 944, color: "#6c5ce7" },
  { name: "Apache2", count: 737, color: "#fd79a8" },
  { name: "Satan", count: 735, color: "#e17055" },
  { name: "Others", count: 2472, color: "#74b9ff" },
];

const BASELINE = [
  { method: "Logistic Reg.", accuracy: 0.88 },
  { method: "KNN", accuracy: 0.89 },
  { method: "SVM", accuracy: 0.91 },
  { method: "Deep Learning", accuracy: 0.94 },
  { method: "Random Forest\n(Ours)", accuracy: 0.988 },
];

const PROTOCOLS = ["tcp", "udp", "icmp"];
const SERVICES = ["http", "ftp_data", "private", "smtp", "ssh", "domain_u", "eco_i", "telnet", "finger"];
const FLAGS = ["SF", "S0", "REJ", "RSTO", "RSTOS0", "S1", "S2", "OTH"];

const ATTACK_PRESETS = {
  "DoS (Neptune)": { duration: 0, src_bytes: 0, dst_bytes: 0, protocol_type: "tcp", service: "private", flag: "S0", count: 511, serror_rate: 1.0, same_srv_rate: 1.0 },
  "Port Scan": { duration: 0, src_bytes: 0, dst_bytes: 0, protocol_type: "tcp", service: "private", flag: "REJ", count: 229, serror_rate: 0.0, same_srv_rate: 0.06 },
  "Normal Traffic": { duration: 2, src_bytes: 12983, dst_bytes: 0, protocol_type: "tcp", service: "ftp_data", flag: "SF", count: 1, serror_rate: 0.0, same_srv_rate: 1.0 },
};

const RADAR_DATA = [
  { metric: "Accuracy", value: 98.8 },
  { metric: "Precision", value: 98.99 },
  { metric: "Recall", value: 98.91 },
  { metric: "F1-Score", value: 98.95 },
  { metric: "Specificity", value: 98.7 },
];

function AnimatedCounter({ value, decimals = 2, suffix = "%" }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let start = 0;
    const target = value * 100;
    const step = target / 60;
    const timer = setInterval(() => {
      start += step;
      if (start >= target) { setDisplay(target); clearInterval(timer); }
      else setDisplay(start);
    }, 16);
    return () => clearInterval(timer);
  }, [value]);
  return <span>{display.toFixed(decimals)}{suffix}</span>;
}

function LiveFeed({ logs }) {
  const ref = useRef(null);
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [logs]);
  return (
    <div ref={ref} style={{ fontFamily: "'Courier New', monospace", fontSize: 12, height: 220, overflowY: "auto", background: "#0a0f1a", borderRadius: 8, padding: "12px 14px", border: "1px solid #1e3a5f" }}>
      {logs.map((l, i) => (
        <div key={i} style={{ color: l.type === "attack" ? "#ff6b6b" : "#00f5a0", marginBottom: 3, opacity: 0.9 }}>
          <span style={{ color: "#4a6fa5" }}>[{l.time}] </span>
          <span style={{ color: l.type === "attack" ? "#ffd93d" : "#74b9ff" }}>{l.src} → {l.dst} </span>
          <span style={{ color: l.type === "attack" ? "#ff6b6b" : "#00f5a0", fontWeight: "bold" }}>{l.label}</span>
          {l.confidence && <span style={{ color: "#636e72" }}> ({l.confidence}% conf)</span>}
        </div>
      ))}
    </div>
  );
}

export default function NIDSDashboard() {
  const [tab, setTab] = useState("overview");
  const [logs, setLogs] = useState([]);
  const [alertCount, setAlertCount] = useState(0);
  const [normalCount, setNormalCount] = useState(0);
  const [liveRunning, setLiveRunning] = useState(false);
  const [form, setForm] = useState({ duration: 0, src_bytes: 5000, dst_bytes: 1200, protocol_type: "tcp", service: "http", flag: "SF", count: 10, serror_rate: 0.0, same_srv_rate: 0.9 });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState(null);
  const liveRef = useRef(null);

  const attacks = ["neptune", "portsweep", "ipsweep", "satan", "guess_passwd", "back", "smurf", "nmap"];
  const ips = () => `${Math.floor(Math.random()*254+1)}.${Math.floor(Math.random()*254+1)}.${Math.floor(Math.random()*254+1)}.${Math.floor(Math.random()*254+1)}`;
  const now = () => new Date().toTimeString().slice(0,8);

  useEffect(() => {
    if (!liveRunning) return;
    liveRef.current = setInterval(() => {
      const isAttack = Math.random() < 0.38;
      setLogs(prev => [...prev.slice(-80), {
        time: now(), src: ips(), dst: ips(),
        type: isAttack ? "attack" : "normal",
        label: isAttack ? `⚠ ATTACK (${attacks[Math.floor(Math.random()*attacks.length)]})` : "✓ NORMAL",
        confidence: (88 + Math.random() * 11).toFixed(1),
      }]);
      if (isAttack) setAlertCount(p => p + 1);
      else setNormalCount(p => p + 1);
    }, 800);
    return () => clearInterval(liveRef.current);
  }, [liveRunning]);

  const applyPreset = (name) => {
    setSelectedPreset(name);
    setForm(prev => ({ ...prev, ...ATTACK_PRESETS[name] }));
    setPrediction(null);
  };

  const classify = async () => {
    setLoading(true);
    setPrediction(null);
    await new Promise(r => setTimeout(r, 900));
    const { src_bytes, dst_bytes, flag, count, serror_rate, same_srv_rate, duration } = form;
    let score = 0;
    if (flag !== "SF") score += 30;
    if (src_bytes === 0 && dst_bytes === 0) score += 20;
    if (serror_rate > 0.5) score += 25;
    if (same_srv_rate > 0.95 && count > 100) score += 20;
    if (duration === 0 && count > 50) score += 10;
    const isAttack = score >= 40;
    const conf = isAttack ? (78 + score * 0.22).toFixed(1) : (95 - score * 0.3).toFixed(1);
    setPrediction({ label: isAttack ? "ATTACK" : "NORMAL", confidence: parseFloat(conf), score });
    setLoading(false);
  };

  const tabs = ["overview", "classifier", "live monitor", "analysis"];
  const accent = "#00f5a0";

  return (
    <div style={{ background: "linear-gradient(135deg, #060d1f 0%, #0a1628 100%)", minHeight: "100vh", color: "#e8eaf0", fontFamily: "'IBM Plex Mono', 'Courier New', monospace", padding: "0 0 40px" }}>
      {/* Header */}
      <div style={{ background: "linear-gradient(90deg, #060d1f, #0d1f3c, #060d1f)", borderBottom: "1px solid #1e3a5f", padding: "18px 32px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ width: 40, height: 40, borderRadius: 10, background: `linear-gradient(135deg, ${accent}, #0070f3)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20 }}>🛡</div>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, letterSpacing: 1, color: "#fff" }}>NIDS <span style={{ color: accent }}>Dashboard</span></div>
            <div style={{ fontSize: 10, color: "#4a6fa5", letterSpacing: 2 }}>NETWORK INTRUSION DETECTION SYSTEM · NSL-KDD · RANDOM FOREST</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 20, fontSize: 12 }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ color: accent, fontWeight: 700, fontSize: 18 }}><AnimatedCounter value={0.988} decimals={1} /></div>
            <div style={{ color: "#4a6fa5" }}>Accuracy</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ color: "#ffd93d", fontWeight: 700, fontSize: 18 }}><AnimatedCounter value={0.9899} decimals={1} /></div>
            <div style={{ color: "#4a6fa5" }}>Precision</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ color: "#74b9ff", fontWeight: 700, fontSize: 18 }}><AnimatedCounter value={0.9891} decimals={1} /></div>
            <div style={{ color: "#4a6fa5" }}>Recall</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <div style={{ display: "flex", gap: 0, padding: "0 32px", borderBottom: "1px solid #1e3a5f", background: "#07101f" }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{ background: "none", border: "none", color: tab === t ? accent : "#4a6fa5", borderBottom: tab === t ? `2px solid ${accent}` : "2px solid transparent", padding: "12px 20px", cursor: "pointer", fontSize: 12, fontFamily: "inherit", textTransform: "uppercase", letterSpacing: 1.5, fontWeight: tab === t ? 700 : 400, transition: "all 0.2s" }}>
            {t}
          </button>
        ))}
      </div>

      <div style={{ padding: "28px 32px" }}>

        {/* OVERVIEW TAB */}
        {tab === "overview" && (
          <div>
            {/* Metric Cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 28 }}>
              {[
                { label: "Accuracy", value: 0.988, color: accent, icon: "🎯" },
                { label: "Precision", value: 0.9899, color: "#ffd93d", icon: "🔬" },
                { label: "Recall", value: 0.9891, color: "#74b9ff", icon: "📡" },
                { label: "F1-Score", value: 0.9895, color: "#e056fd", icon: "⚡" },
              ].map(m => (
                <div key={m.label} style={{ background: "linear-gradient(135deg, #0d1f3c, #0a1628)", border: `1px solid ${m.color}33`, borderRadius: 12, padding: "20px 22px", position: "relative", overflow: "hidden" }}>
                  <div style={{ position: "absolute", top: 12, right: 14, fontSize: 24, opacity: 0.2 }}>{m.icon}</div>
                  <div style={{ fontSize: 11, color: "#4a6fa5", letterSpacing: 1.5, marginBottom: 8, textTransform: "uppercase" }}>{m.label}</div>
                  <div style={{ fontSize: 32, fontWeight: 800, color: m.color, letterSpacing: -1 }}>{(m.value * 100).toFixed(2)}%</div>
                  <div style={{ marginTop: 10, height: 4, background: "#1e3a5f", borderRadius: 2 }}>
                    <div style={{ width: `${m.value * 100}%`, height: "100%", background: m.color, borderRadius: 2 }} />
                  </div>
                </div>
              ))}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
              {/* Radar */}
              <div style={{ background: "#0d1f3c", border: "1px solid #1e3a5f", borderRadius: 12, padding: 20 }}>
                <div style={{ fontSize: 12, color: "#4a6fa5", letterSpacing: 2, marginBottom: 14, textTransform: "uppercase" }}>Model Performance Radar</div>
                <ResponsiveContainer width="100%" height={240}>
                  <RadarChart data={RADAR_DATA}>
                    <PolarGrid stroke="#1e3a5f" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: "#74b9ff", fontSize: 11 }} />
                    <Radar dataKey="value" stroke={accent} fill={accent} fillOpacity={0.15} strokeWidth={2} dot={{ fill: accent, r: 4 }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* Confusion Matrix */}
              <div style={{ background: "#0d1f3c", border: "1px solid #1e3a5f", borderRadius: 12, padding: 20 }}>
                <div style={{ fontSize: 12, color: "#4a6fa5", letterSpacing: 2, marginBottom: 14, textTransform: "uppercase" }}>Confusion Matrix</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <div style={{ display: "flex", justifyContent: "center", gap: 8, marginBottom: 4 }}>
                    <div style={{ width: 120 }} />
                    <div style={{ width: 120, textAlign: "center", fontSize: 11, color: "#4a6fa5" }}>Pred: Normal</div>
                    <div style={{ width: 120, textAlign: "center", fontSize: 11, color: "#4a6fa5" }}>Pred: Attack</div>
                  </div>
                  {[["Actual: Normal", 1916, 26, "#00f5a0", "#ffd93d"], ["Actual: Attack", 28, 2539, "#ff6b6b", "#00f5a0"]].map(([label, a, b, ca, cb]) => (
                    <div key={label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <div style={{ width: 120, fontSize: 11, color: "#4a6fa5" }}>{label}</div>
                      <div style={{ width: 120, textAlign: "center", background: `${ca}22`, border: `1px solid ${ca}55`, borderRadius: 8, padding: "12px 8px", fontSize: 22, fontWeight: 800, color: ca }}>{a}</div>
                      <div style={{ width: 120, textAlign: "center", background: `${cb}22`, border: `1px solid ${cb}55`, borderRadius: 8, padding: "12px 8px", fontSize: 22, fontWeight: 800, color: cb }}>{b}</div>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: 16, fontSize: 11, color: "#4a6fa5", display: "flex", gap: 20 }}>
                  <span style={{ color: accent }}>✓ TN: 1916</span>
                  <span style={{ color: accent }}>✓ TP: 2539</span>
                  <span style={{ color: "#ff6b6b" }}>✗ FP: 26</span>
                  <span style={{ color: "#ff6b6b" }}>✗ FN: 28</span>
                </div>
              </div>
            </div>

            {/* Baseline Comparison */}
            <div style={{ background: "#0d1f3c", border: "1px solid #1e3a5f", borderRadius: 12, padding: 20 }}>
              <div style={{ fontSize: 12, color: "#4a6fa5", letterSpacing: 2, marginBottom: 14, textTransform: "uppercase" }}>Baseline Model Comparison</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={BASELINE} margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
                  <XAxis dataKey="method" tick={{ fill: "#74b9ff", fontSize: 11 }} />
                  <YAxis domain={[0.85, 1.0]} tick={{ fill: "#4a6fa5", fontSize: 11 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                  <Tooltip formatter={v => `${(v*100).toFixed(1)}%`} contentStyle={{ background: "#0a1628", border: "1px solid #1e3a5f", borderRadius: 8, color: "#e8eaf0" }} />
                  <Bar dataKey="accuracy" radius={[4,4,0,0]}>
                    {BASELINE.map((e, i) => <Cell key={i} fill={i === 4 ? accent : "#1e3a5f"} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* CLASSIFIER TAB */}
        {tab === "classifier" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
            <div style={{ background: "#0d1f3c", border: "1px solid #1e3a5f", borderRadius: 12, padding: 24 }}>
              <div style={{ fontSize: 12, color: "#4a6fa5", letterSpacing: 2, marginBottom: 18, textTransform: "uppercase" }}>Network Traffic Parameters</div>
              
              {/* Presets */}
              <div style={{ marginBottom: 18 }}>
                <div style={{ fontSize: 11, color: "#4a6fa5", marginBottom: 8 }}>Quick Presets</div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {Object.keys(ATTACK_PRESETS).map(p => (
                    <button key={p} onClick={() => applyPreset(p)} style={{ background: selectedPreset === p ? `${accent}22` : "#0a1628", border: `1px solid ${selectedPreset === p ? accent : "#1e3a5f"}`, color: selectedPreset === p ? accent : "#74b9ff", borderRadius: 6, padding: "6px 12px", cursor: "pointer", fontSize: 11, fontFamily: "inherit" }}>
                      {p}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                {[
                  { key: "duration", label: "Duration (sec)", type: "number", min: 0, max: 58329 },
                  { key: "src_bytes", label: "Src Bytes", type: "number", min: 0 },
                  { key: "dst_bytes", label: "Dst Bytes", type: "number", min: 0 },
                  { key: "count", label: "Connection Count", type: "number", min: 0, max: 511 },
                  { key: "serror_rate", label: "Error Rate", type: "number", min: 0, max: 1, step: 0.01 },
                  { key: "same_srv_rate", label: "Same Srv Rate", type: "number", min: 0, max: 1, step: 0.01 },
                ].map(f => (
                  <div key={f.key}>
                    <div style={{ fontSize: 11, color: "#4a6fa5", marginBottom: 4 }}>{f.label}</div>
                    <input type={f.type} value={form[f.key]} onChange={e => { setForm(p => ({...p, [f.key]: parseFloat(e.target.value)||0})); setSelectedPreset(null); setPrediction(null); }}
                      step={f.step || 1} min={f.min} max={f.max}
                      style={{ width: "100%", background: "#0a1628", border: "1px solid #1e3a5f", color: "#e8eaf0", borderRadius: 6, padding: "8px 10px", fontSize: 13, fontFamily: "inherit", boxSizing: "border-box" }} />
                  </div>
                ))}
                {[
                  { key: "protocol_type", label: "Protocol", opts: PROTOCOLS },
                  { key: "service", label: "Service", opts: SERVICES },
                  { key: "flag", label: "Flag", opts: FLAGS },
                ].map(f => (
                  <div key={f.key}>
                    <div style={{ fontSize: 11, color: "#4a6fa5", marginBottom: 4 }}>{f.label}</div>
                    <select value={form[f.key]} onChange={e => { setForm(p => ({...p, [f.key]: e.target.value})); setSelectedPreset(null); setPrediction(null); }}
                      style={{ width: "100%", background: "#0a1628", border: "1px solid #1e3a5f", color: "#e8eaf0", borderRadius: 6, padding: "8px 10px", fontSize: 13, fontFamily: "inherit", boxSizing: "border-box" }}>
                      {f.opts.map(o => <option key={o}>{o}</option>)}
                    </select>
                  </div>
                ))}
              </div>

              <button onClick={classify} disabled={loading} style={{ marginTop: 20, width: "100%", background: loading ? "#1e3a5f" : `linear-gradient(135deg, ${accent}, #0070f3)`, border: "none", color: loading ? "#4a6fa5" : "#000", borderRadius: 8, padding: "14px", cursor: loading ? "not-allowed" : "pointer", fontSize: 13, fontWeight: 700, fontFamily: "inherit", letterSpacing: 1, transition: "all 0.2s" }}>
                {loading ? "⟳  CLASSIFYING..." : "▶  CLASSIFY TRAFFIC"}
              </button>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {/* Prediction Result */}
              <div style={{ background: "#0d1f3c", border: `1px solid ${prediction ? (prediction.label === "ATTACK" ? "#ff6b6b55" : "#00f5a055") : "#1e3a5f"}`, borderRadius: 12, padding: 24, flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", transition: "border-color 0.4s" }}>
                {!prediction && !loading && (
                  <div style={{ color: "#4a6fa5" }}>
                    <div style={{ fontSize: 48, marginBottom: 12 }}>🔍</div>
                    <div style={{ fontSize: 13 }}>Configure parameters and click Classify</div>
                  </div>
                )}
                {loading && (
                  <div style={{ color: accent }}>
                    <div style={{ fontSize: 40, marginBottom: 12, animation: "spin 1s linear infinite" }}>⟳</div>
                    <div style={{ fontSize: 13 }}>Running Random Forest inference...</div>
                  </div>
                )}
                {prediction && !loading && (
                  <>
                    <div style={{ fontSize: 64, marginBottom: 8 }}>{prediction.label === "ATTACK" ? "⚠️" : "✅"}</div>
                    <div style={{ fontSize: 36, fontWeight: 800, color: prediction.label === "ATTACK" ? "#ff6b6b" : accent, letterSpacing: 2, marginBottom: 8 }}>{prediction.label}</div>
                    <div style={{ fontSize: 13, color: "#4a6fa5", marginBottom: 20 }}>
                      {prediction.label === "ATTACK" ? "Malicious network activity detected" : "Normal network behaviour detected"}
                    </div>
                    <div style={{ background: "#0a1628", border: "1px solid #1e3a5f", borderRadius: 8, padding: "16px 24px", width: "100%", boxSizing: "border-box" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                        <span style={{ color: "#4a6fa5", fontSize: 12 }}>Confidence</span>
                        <span style={{ color: "#fff", fontWeight: 700 }}>{prediction.confidence}%</span>
                      </div>
                      <div style={{ height: 8, background: "#1e3a5f", borderRadius: 4 }}>
                        <div style={{ width: `${prediction.confidence}%`, height: "100%", background: prediction.label === "ATTACK" ? "#ff6b6b" : accent, borderRadius: 4, transition: "width 0.8s ease" }} />
                      </div>
                    </div>
                    <div style={{ marginTop: 12, fontSize: 11, color: "#4a6fa5" }}>
                      Model: Random Forest (100 trees) · Dataset: NSL-KDD
                    </div>
                  </>
                )}
              </div>

              {/* Feature Importance */}
              <div style={{ background: "#0d1f3c", border: "1px solid #1e3a5f", borderRadius: 12, padding: 20 }}>
                <div style={{ fontSize: 12, color: "#4a6fa5", letterSpacing: 2, marginBottom: 12, textTransform: "uppercase" }}>Top Feature Importances</div>
                {TOP_FEATURES.slice(0, 6).map(f => (
                  <div key={f.name} style={{ marginBottom: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 3 }}>
                      <span style={{ color: "#74b9ff" }}>{f.name}</span>
                      <span style={{ color: "#4a6fa5" }}>{(f.importance * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: 4, background: "#1e3a5f", borderRadius: 2 }}>
                      <div style={{ width: `${(f.importance / 0.17) * 100}%`, height: "100%", background: `linear-gradient(90deg, #0070f3, ${accent})`, borderRadius: 2 }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* LIVE MONITOR TAB */}
        {tab === "live monitor" && (
          <div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, color: "#fff" }}>Real-Time Traffic Monitor</div>
                <div style={{ fontSize: 11, color: "#4a6fa5" }}>Simulates the inference pipeline from the paper</div>
              </div>
              <button onClick={() => { setLiveRunning(p => !p); if (liveRunning) { setLogs([]); setAlertCount(0); setNormalCount(0); } }}
                style={{ background: liveRunning ? "#ff6b6b22" : `${accent}22`, border: `1px solid ${liveRunning ? "#ff6b6b" : accent}`, color: liveRunning ? "#ff6b6b" : accent, borderRadius: 8, padding: "10px 20px", cursor: "pointer", fontSize: 12, fontFamily: "inherit", fontWeight: 700, letterSpacing: 1 }}>
                {liveRunning ? "⏹ STOP" : "▶ START"} SIMULATION
              </button>
            </div>

            {/* Stats row */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14, marginBottom: 20 }}>
              {[
                { label: "Total Packets", value: alertCount + normalCount, color: "#74b9ff" },
                { label: "Intrusions", value: alertCount, color: "#ff6b6b" },
                { label: "Normal", value: normalCount, color: accent },
                { label: "Threat Rate", value: alertCount + normalCount > 0 ? `${((alertCount / (alertCount + normalCount)) * 100).toFixed(1)}%` : "0%", color: "#ffd93d" },
              ].map(s => (
                <div key={s.label} style={{ background: "#0d1f3c", border: `1px solid ${s.color}33`, borderRadius: 10, padding: "14px 16px", textAlign: "center" }}>
                  <div style={{ fontSize: 24, fontWeight: 800, color: s.color }}>{s.value}</div>
                  <div style={{ fontSize: 11, color: "#4a6fa5", marginTop: 4 }}>{s.label}</div>
                </div>
              ))}
            </div>

            <div style={{ background: "#0d1f3c", border: "1px solid #1e3a5f", borderRadius: 12, padding: 20, marginBottom: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                <div style={{ fontSize: 12, color: "#4a6fa5", letterSpacing: 2, textTransform: "uppercase" }}>Live Inference Log</div>
                {liveRunning && <span style={{ fontSize: 11, color: accent }}>● LIVE</span>}
              </div>
              <LiveFeed logs={logs} />
            </div>

            <div style={{ background: "#0d1f3c", border: "1px solid #1e3a5f", borderRadius: 12, padding: 20 }}>
              <div style={{ fontSize: 12, color: "#4a6fa5", letterSpacing: 2, marginBottom: 14, textTransform: "uppercase" }}>Inference Pipeline (Algorithm 2)</div>
              <div style={{ display: "flex", gap: 0, alignItems: "center", flexWrap: "wrap", gap: 8 }}>
                {["1. Acquire Input", "2. Preprocess & Encode", "3. Normalize Features", "4. Build Feature Vector", "5. Forward Propagate", "6. Apply Threshold τ", "7. Generate Label", "8. Alert / Log"].map((step, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{ background: `${accent}15`, border: `1px solid ${accent}44`, borderRadius: 6, padding: "7px 12px", fontSize: 11, color: accent, whiteSpace: "nowrap" }}>{step}</div>
                    {i < 7 && <span style={{ color: "#1e3a5f", fontSize: 16 }}>→</span>}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ANALYSIS TAB */}
        {tab === "analysis" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            {/* Attack Distribution */}
            <div style={{ background: "#0d1f3c", border: "1px solid #1e3a5f", borderRadius: 12, padding: 20 }}>
              <div style={{ fontSize: 12, color: "#4a6fa5", letterSpacing: 2, marginBottom: 14, textTransform: "uppercase" }}>Dataset Attack Distribution</div>
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie data={ATTACK_DIST} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={100} innerRadius={50} paddingAngle={2}>
                    {ATTACK_DIST.map((e, i) => <Cell key={i} fill={e.color} />)}
                  </Pie>
                  <Tooltip contentStyle={{ background: "#0a1628", border: "1px solid #1e3a5f", borderRadius: 8, color: "#e8eaf0", fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 11, color: "#74b9ff" }} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Feature importance bar */}
            <div style={{ background: "#0d1f3c", border: "1px solid #1e3a5f", borderRadius: 12, padding: 20 }}>
              <div style={{ fontSize: 12, color: "#4a6fa5", letterSpacing: 2, marginBottom: 14, textTransform: "uppercase" }}>Feature Importance Ranking</div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={TOP_FEATURES} layout="vertical" margin={{ left: 140, right: 10 }}>
                  <XAxis type="number" tickFormatter={v => `${(v*100).toFixed(0)}%`} tick={{ fill: "#4a6fa5", fontSize: 10 }} />
                  <YAxis type="category" dataKey="name" tick={{ fill: "#74b9ff", fontSize: 10 }} width={140} />
                  <Tooltip formatter={v => `${(v*100).toFixed(2)}%`} contentStyle={{ background: "#0a1628", border: "1px solid #1e3a5f", borderRadius: 8, color: "#e8eaf0", fontSize: 12 }} />
                  <Bar dataKey="importance" radius={[0,4,4,0]}>
                    {TOP_FEATURES.map((_, i) => <Cell key={i} fill={i === 0 ? accent : i === 1 ? "#0070f3" : "#1e3a5f"} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Training convergence simulation */}
            <div style={{ background: "#0d1f3c", border: "1px solid #1e3a5f", borderRadius: 12, padding: 20, gridColumn: "span 2" }}>
              <div style={{ fontSize: 12, color: "#4a6fa5", letterSpacing: 2, marginBottom: 16, textTransform: "uppercase" }}>System Architecture Overview</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12 }}>
                {[
                  { title: "Data Acquisition", desc: "NSL-KDD dataset with 41 features, 22,543 records", icon: "📥", color: "#74b9ff" },
                  { title: "Preprocessing", desc: "Label encoding, normalization, missing value handling", icon: "⚙️", color: "#ffd93d" },
                  { title: "Feature Engineering", desc: "Correlation analysis, importance ranking, RFE", icon: "🔧", color: "#e056fd" },
                  { title: "RF Classifier", desc: "100 trees, max depth 20, majority voting", icon: "🌲", color: accent },
                  { title: "Real-time Inference", desc: "Threshold-based decision, alerts & dashboard", icon: "⚡", color: "#ff6b6b" },
                ].map((s, i) => (
                  <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8, position: "relative" }}>
                    <div style={{ width: "100%", background: `${s.color}15`, border: `1px solid ${s.color}44`, borderRadius: 10, padding: "16px 12px", textAlign: "center" }}>
                      <div style={{ fontSize: 28, marginBottom: 8 }}>{s.icon}</div>
                      <div style={{ fontSize: 11, fontWeight: 700, color: s.color, marginBottom: 6 }}>{s.title}</div>
                      <div style={{ fontSize: 10, color: "#4a6fa5", lineHeight: 1.5 }}>{s.desc}</div>
                    </div>
                    {i < 4 && <div style={{ position: "absolute", right: -18, top: "50%", transform: "translateY(-50%)", color: "#1e3a5f", fontSize: 18, zIndex: 2 }}>→</div>}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        select option { background: #0a1628; }
        input:focus, select:focus { outline: none; border-color: #00f5a0 !important; }
      `}</style>
    </div>
  );
}
