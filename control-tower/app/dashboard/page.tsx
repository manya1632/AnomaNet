"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Zap, RefreshCw, BarChart3, Fingerprint, Shuffle,
  Link2, Network, Boxes, ChevronRight, Waves, LayoutDashboard, 
  Bell, FileSearch, Briefcase, FileText, UserCircle, Maximize2, 
  Search, Filter, ExternalLink, Download, Play, ShieldAlert,
  ArrowUpRight, Clock, MoreHorizontal
} from "lucide-react";
import * as Recharts from 'recharts';
import { Activity, Square, 
  Terminal, Cpu, HardDrive, Database, Info, 
  Settings2, Flame, Loader2 
} from "lucide-react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

import dynamic from "next/dynamic";
import NodeInspector from "../components/graph/NodeInspector";
import type { GraphNode } from "../components/graph/FraudGraph3D";

const FraudGraph3D = dynamic(
  () => import("../components/graph/FraudGraph3D"),
  { ssr: false }
);

// --- Constants & Types ---
const ML_URL  = "http://56.228.10.113:8001";


const NAV = [
  { group: "CORE", items: [
    { id: "dashboard",  label: "Dashboard",  icon: LayoutDashboard, notifications: null },
    { id: "alerts",     label: "Alerts",     icon: Bell, notifications: 12 },
    { id: "graph",      label: "Graph Explorer", icon: FileSearch, notifications: null },
  ]},
  { group: "INVESTIGATION", items: [
    { id: "cases",      label: "Cases",      icon: Briefcase, notifications: 3 },
    { id: "reports",    label: "Reports",    icon: FileText, notifications: null },
    { id: "simulator",  label: "Simulator",  icon: Zap, notifications: null },
  ]}
] as const;

type View = typeof NAV[number]["items"][number]["id"];



const ALERT_FEED_MOCK = [
    { id: "ACC-8821904", score: 0.91, pattern: "CIRCULAR", detail: "3-hop cycle", time: "2s ago" },
    { id: "ACC-2291847", score: 0.83, pattern: "STRUCTURING", detail: "₹9.6L×3", time: "48s ago" },
    { id: "ACC-5503912", score: 0.74, pattern: "LAYERING", detail: "fan-out 7", time: "2m ago" },
    { id: "ACC-9012283", score: 0.68, pattern: "DORMANT", detail: "₹1.8Cr activation", time: "5m ago" },
    { id: "ACC-3381029", score: 0.71, pattern: "PROFILE MISMATCH", detail: "", time: "9m ago" },
];

const TRANSACTION_VOLUME_MOCK = Array.from({ length: 24 }, (_, i) => ({
    time: `${String(i).padStart(2, '0')}:00`,
    volume: Math.floor(Math.random() * 500) + 100,
}));

// --- Helper UI Components ---
const Card = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <div className={`relative border border-white/[0.09] rounded-[1.5rem] bg-[#0a0a0a] overflow-hidden ${className}`}
    style={{ boxShadow: "inset 0 1px 0 rgba(255,255,255,0.06), 0 4px 24px rgba(0,0,0,0.4)" }}>
    {children}
  </div>
);
const AttackVector = ({ label, active, onClick }: { label: string, active: boolean, onClick: () => void }) => (
  <button 
    onClick={onClick}
    className={`w-full flex items-center justify-between p-3 rounded-xl border transition-all ${
      active 
      ? "bg-[#ef4444]/10 border-[#ef4444]/40 text-[#ef4444]" 
      : "bg-white/[0.03] border-white/10 text-white/40 hover:border-white/20"
    }`}
  >
    <span className="text-[10px] font-black uppercase tracking-widest">{label}</span>
    {active ? <Flame size={14} /> : <div className="w-1.5 h-1.5 rounded-full bg-white/10" />}
  </button>
);

const MetricSmall = ({ icon: Icon, label, value, unit }: any) => (
  <div className="flex flex-col gap-1.5">
    <div className="flex items-center gap-2 text-white/40">
      <Icon size={12} />
      <span className="text-[9px] font-bold uppercase tracking-tighter">{label}</span>
    </div>
    <p className="text-sm font-black text-white font-mono">{value}<span className="text-[10px] ml-0.5 opacity-40">{unit}</span></p>
  </div>
);

const Eyebrow = ({ children }: { children: React.ReactNode }) => (
  <p className="text-[10px] font-bold uppercase tracking-[0.3em] text-white/60 mb-2">{children}</p>
);

const hex = (s: number) => s >= 0.75 ? "#ef4444" : s >= 0.45 ? "#facc15" : "#CAFF33";
const tc  = (s: number) => s >= 0.75 ? "text-red-400" : s >= 0.45 ? "text-yellow-400" : "text-[#CAFF33]";

// --- Section 1: ALERTS VIEW ---
const AlertsSection = () => {
  const alerts = [
    { id: "AL-9022", target: "ACC_88291", type: "Circular Flow", score: 0.94, time: "2m ago", status: "Critical" },
    { id: "AL-9021", target: "ACC_11204", type: "High Velocity", score: 0.82, time: "14m ago", status: "Warning" },
    { id: "AL-9020", target: "ACC_55032", type: "Mule Activity", score: 0.76, time: "1h ago", status: "Warning" },
  ];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <Eyebrow>Monitoring</Eyebrow>
          <h2 className="text-3xl font-black text-white">Active Alerts</h2>
        </div>
        <div className="flex gap-3">
          <button className="px-4 py-2 bg-white/5 rounded-xl border border-white/10 text-xs font-bold flex items-center gap-2">
            <Filter size={14} /> Filter
          </button>
        </div>
      </div>
      <Card className="overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-white/[0.02] border-b border-white/[0.06]">
              <th className="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-white/40">ID</th>
              <th className="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-white/40">Target Account</th>
              <th className="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-white/40">Pattern Type</th>
              <th className="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-white/40">Score</th>
              <th className="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-white/40">Status</th>
              <th className="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-white/40 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/[0.04]">
            {alerts.map((a) => (
              <tr key={a.id} className="hover:bg-white/[0.02] transition-colors group">
                <td className="px-6 py-5 font-mono text-sm text-white/80">{a.id}</td>
                <td className="px-6 py-5 font-bold text-white">{a.target}</td>
                <td className="px-6 py-5 text-sm text-white/60">{a.type}</td>
                <td className={`px-6 py-5 font-black font-mono ${tc(a.score)}`}>{a.score.toFixed(2)}</td>
                <td className="px-6 py-5">
                  <span className={`px-2 py-1 rounded text-[9px] font-bold uppercase ${a.status === 'Critical' ? 'bg-red-500/10 text-red-400' : 'bg-yellow-500/10 text-yellow-400'}`}>
                    {a.status}
                  </span>
                </td>
                <td className="px-6 py-5 text-right">
                  <button className="text-white/40 group-hover:text-[#CAFF33] transition-colors"><ExternalLink size={16}/></button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
};

function LiveBadge({ loading, error, text='live' }: { loading?: boolean; error?: boolean, text?: string }) {
  if (loading) return <span className="text-[9px] text-white/70 animate-pulse font-mono">fetching…</span>;
  if (error)   return <span className="text-[9px] text-yellow-400/70 font-mono">⚠ unreachable</span>;
  return (
    <span className="flex items-center gap-1.5 text-[9px] text-[#CAFF33]/70 font-mono font-bold uppercase tracking-widest">
      <span className="w-1.5 h-1.5 rounded-full bg-[#CAFF33]" style={{ boxShadow: "0 0 6px #CAFF33" }} />
      {text}
    </span>
  );
}

function StatCard({ label, value, sub, color = "text-[#CAFF33]", accent, trend }: { label: string; value: React.ReactNode; sub: string; color?: string; accent?: string, trend?: { value: string, direction: 'up' | 'down' } }) {
  return (
    <Card className="p-6 relative overflow-hidden flex flex-col h-full">
      {accent && <div className="absolute top-0 right-0 w-24 h-24 rounded-full opacity-[0.03]" style={{ background: accent, filter: "blur(24px)", transform: "translate(30%,-30%)" }} />}
      <p className="text-[10px] text-white/70 uppercase tracking-[0.2em] mb-3 font-bold">{label}</p>
      <div className="flex items-baseline gap-2 mb-2">
        <p className={`text-4xl font-black leading-none ${color}`}>{value}</p>
        {trend && (
            <span className={`text-[11px] font-bold ${trend.direction === 'up' ? 'text-red-400' : 'text-[#CAFF33]'}`}>
                {trend.direction === 'up' ? '↑' : '↓'} {trend.value}
            </span>
        )}
      </div>
      <p className="text-xs text-white/60 mt-auto">{sub}</p>
    </Card>
  );
}
// --- Section 1: DASHBOARD (Command Center) ---
const DashboardSection = () => (
    <div className="flex flex-col gap-8 h-full">
        <div className="flex items-center justify-between">
            <div>
                <Eyebrow>Operations</Eyebrow>
                <h2 className="text-3xl font-black text-white">Command Center</h2>
            </div>
            <div className="flex items-center gap-6">
                <LiveBadge text="Live · 3,241 tx/min" />
                <p className="text-xs text-white/70 font-bold uppercase tracking-wider">Avg AnomaScore <span className="text-red-400">0.71</span></p>
            </div>
        </div>

        <div className="grid grid-cols-4 gap-5">
            <StatCard label="ACTIVE ALERTS" value="12" sub="4 in last hour" color="text-red-400" trend={{ value: '4', direction: 'up' }} accent="#ef4444" />
            <StatCard label="HIGH-RISK ACCOUNTS" value="47" sub="score > 0.70 today" color="text-yellow-400" accent="#facc15" />
            <StatCard label="TRANSACTIONS / MIN" value="3,241" sub="+ 8% vs yesterday" trend={{ value: '8%', direction: 'up' }} color="text-[#CAFF33]" accent="#CAFF33" />
            <StatCard label="AVG ANOMASCORE" value="0.71" sub="threshold 0.65" color="text-red-400" accent="#ef4444" />
        </div>

        <div className="grid grid-cols-[1fr_420px] gap-6 flex-1 min-h-[400px]">
            <Card className="p-8">
                <div className="flex justify-between items-center mb-8">
                    <div>
                        <Eyebrow>Trend Analysis</Eyebrow>
                        <p className="text-lg font-bold text-white italic">Transaction Volume · 24h</p>
                    </div>
                    <LiveBadge text="Live Polling" />
                </div>
                <div className="h-[280px] w-full">
                    <Recharts.ResponsiveContainer width="100%" height="100%">
                        <Recharts.AreaChart data={TRANSACTION_VOLUME_MOCK} margin={{ top: 10, right: 10, left: -30, bottom: 0 }}>
                            <Recharts.XAxis dataKey="time" interval={5} tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)', fontWeight: 'bold' }} axisLine={false} tickLine={false} />
                            <Recharts.YAxis hide />
                            <Recharts.Tooltip contentStyle={{ background: '#0a0a0a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px' }} />
                            <defs>
                                <linearGradient id="colorVolume" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                                </linearGradient>
                            </defs>
                            <Recharts.Area type="monotone" dataKey="volume" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorVolume)" dot={{ r: 0 }} activeDot={{ r: 6, fill: '#fff', stroke: '#3b82f6', strokeWidth: 2 }} />
                        </Recharts.AreaChart>
                    </Recharts.ResponsiveContainer>
                </div>
            </Card>

            <Card className="p-8 flex flex-col">
                <div className="flex justify-between items-center mb-8">
                    <p className="text-lg font-black text-white uppercase tracking-tighter italic">Live Alert Feed</p>
                    <LiveBadge text="WS Live" />
                </div>
                <div className="space-y-1 flex-1 overflow-y-auto pr-2 -mr-2 custom-scrollbar">
                    {ALERT_FEED_MOCK.map(alert => (
                        <div key={alert.id} className="flex items-center gap-5 py-4 border-b border-white/[0.04] last:border-0 hover:bg-white/[0.02] transition-colors rounded-2xl px-3">
                            <div className={`text-2xl font-black font-mono w-16 text-center shrink-0 ${tc(alert.score)}`}>
                                {alert.score.toFixed(2)}
                            </div>
                            <div className="flex-1 min-w-0">
                                <p className="text-sm font-bold text-white font-mono tracking-tight">{alert.id}</p>
                                <p className="text-[10px] text-white/40 font-black uppercase tracking-widest mt-0.5">
                                    {alert.pattern} {alert.detail && <span className="text-white/20">/ {alert.detail}</span>}
                                </p>
                            </div>
                            <div className="text-[9px] text-white/30 font-bold font-mono uppercase shrink-0">
                                {alert.time}
                            </div>
                        </div>
                    ))}
                </div>
            </Card>
        </div>
    </div>
);

// --- Section 2: GRAPH EXPLORER ---
function GraphExplorerSection() {
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  return (
    <div className="w-full h-full flex flex-col">

      {/* 🔹 HEADER */}
      <div className="flex justify-between items-center mb-4 px-2">
        <h2 className="text-2xl font-black text-white">
          Network Graph Explorer
        </h2>

        <div className="bg-white/5 border border-white/10 rounded-xl p-1 flex gap-1">
          <button className="px-3 py-1.5 bg-[#CAFF33] text-black text-[10px] font-bold rounded-lg uppercase">
            3D View
          </button>
          <button className="px-3 py-1.5 text-white/60 text-[10px] font-bold rounded-lg uppercase">
            Cluster
          </button>
        </div>
      </div>

      {/* 🔥 GRAPH AREA (FULL RIGHT SIDE) */}
      <div className="flex-1 relative rounded-2xl overflow-hidden border border-white/10">

        {/* GRAPH */}
        <FraudGraph3D
          onNodeSelect={setSelectedNode}
          selectedNode={selectedNode}
        />

        {/* 🔥 INSPECTOR OVERLAY */}
        {selectedNode && (
          <div className="absolute top-0 right-0 h-full z-20">
            <NodeInspector
              node={selectedNode}
              onClose={() => setSelectedNode(null)}
            />
          </div>
        )}

      </div>
    </div>
  );
}

// --- Section 3: CASES VIEW ---
const CasesSection = () => (
  <div className="space-y-6">
    <div className="flex justify-between items-center">
      <h2 className="text-3xl font-black text-white">Case Management</h2>
      <button className="px-6 py-2.5 bg-[#CAFF33] text-black font-black text-[10px] uppercase rounded-xl">+ New Case</button>
    </div>
    <div className="grid grid-cols-3 gap-6">
      {[1, 2, 3].map(i => (
        <Card key={i} className="p-6 hover:border-[#CAFF33]/30 transition-all cursor-pointer">
          <div className="flex justify-between items-start mb-6">
             <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center"><Briefcase size={20} className="text-white/60" /></div>
             <span className="text-[10px] font-bold px-2 py-1 bg-white/5 border border-white/10 rounded text-white/40 uppercase">In Progress</span>
          </div>
          <p className="text-[10px] text-[#CAFF33] font-black uppercase mb-1">CASE #102{i}</p>
          <h3 className="text-lg font-bold text-white mb-2 underline decoration-white/10">Suspected Ring: Sector 09</h3>
          <p className="text-xs text-white/50 leading-relaxed mb-4">Investigation into a cluster of 12 accounts with high-frequency circular transfers originating from Kolkata node.</p>
          <div className="flex items-center gap-4 border-t border-white/5 pt-4">
             <div className="flex -space-x-2">
                <div className="w-6 h-6 rounded-full bg-blue-500 border border-black" />
                <div className="w-6 h-6 rounded-full bg-purple-500 border border-black" />
             </div>
             <span className="text-[10px] text-white/40 font-bold uppercase tracking-widest">2 Investigators</span>
          </div>
        </Card>
      ))}
    </div>
  </div>
);

// --- Section 4: REPORTS VIEW ---
const ReportsSection = () => (
  <div className="space-y-6">
    <div className="flex justify-between items-center">
      <h2 className="text-3xl font-black text-white">Intelligence Reports</h2>
      <div className="flex gap-2">
        <button className="p-2 bg-white/5 border border-white/10 rounded-lg text-white/60"><Search size={18}/></button>
        <button className="p-2 bg-white/5 border border-white/10 rounded-lg text-white/60"><Download size={18}/></button>
      </div>
    </div>
    <div className="space-y-3">
      {[
        { title: "Quarterly Fraud Landscape Q1 2026", date: "Mar 12, 2026", size: "4.2 MB", type: "PDF" },
        { title: "Mule Ring Detection: Advanced GNN Patterns", date: "Mar 08, 2026", size: "1.8 MB", type: "DOCX" },
        { title: "Weekly Anomaly Summary #44", date: "Mar 01, 2026", size: "890 KB", type: "PDF" },
      ].map((doc, idx) => (
        <Card key={idx} className="p-4 flex items-center justify-between group hover:bg-white/[0.01] transition-colors">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center"><FileText size={24} className="text-white/40" /></div>
            <div>
              <p className="font-bold text-white group-hover:text-[#CAFF33] transition-colors">{doc.title}</p>
              <p className="text-[10px] text-white/40 uppercase font-bold tracking-widest">{doc.date} · {doc.size}</p>
            </div>
          </div>
          <button className="text-white/20 group-hover:text-white transition-colors"><Download size={20}/></button>
        </Card>
      ))}
    </div>
  </div>
);

// --- Section 5: SIMULATOR VIEW ---
export const SimulatorSection = () => {
  const [isSimulating, setIsSimulating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [activeVectors, setActiveVectors] = useState<string[]>(["Circular"]);
  const [logs, setLogs] = useState<{msg: string, type: 'info' | 'warn' | 'err'}[]>([]);

  // Simple Log Simulator
  useEffect(() => {
    if (!isSimulating) return;
    const interval = setInterval(() => {
      const msgs = [
        { msg: "Injecting synthetic batch: 500 tx/s", type: 'info' },
        { msg: "GNN Inference: High clustering detected in subgraph 0x44", type: 'warn' },
        { msg: "EIF score spike: 0.89 on Node_9921", type: 'err' },
        { msg: "Blockchain sync latency: 12ms", type: 'info' }
      ] as const;
      const randomMsg = msgs[Math.floor(Math.random() * msgs.length)];
      setLogs(prev => [randomMsg, ...prev].slice(0, 8));
      setProgress(p => (p + 1) % 100);
    }, 1500);
    return () => clearInterval(interval);
  }, [isSimulating]);

  const toggleVector = (v: string) => {
    setActiveVectors(prev => prev.includes(v) ? prev.filter(x => x !== v) : [...prev, v]);
  };

  return (
    <div className="grid grid-cols-[380px_1fr] gap-8 h-full">
      {/* Sidebar Controls */}
      <div className="space-y-6 flex flex-col h-full">
        <div>
          <Eyebrow>Simulation Engine</Eyebrow>
          <h2 className="text-2xl font-black text-white italic">Scenario Lab</h2>
        </div>

        <Card className="p-6 space-y-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-black text-white/60 uppercase tracking-widest">Target Environment</span>
              <span className="text-[9px] text-[#CAFF33] font-mono bg-[#CAFF33]/10 px-2 py-0.5 rounded">STAGING_V2</span>
            </div>
            
            <div className="space-y-1.5">
              <label className="text-[10px] font-bold text-white/40 uppercase">Synthetic Volume (TPS)</label>
              <div className="flex items-center gap-4">
                <input type="range" className="flex-1 accent-[#CAFF33]" defaultValue={45} />
                <span className="text-xs font-mono font-bold text-white w-8">450</span>
              </div>
            </div>

            <div className="space-y-3">
              <label className="text-[10px] font-bold text-white/40 uppercase">Attack Vectors</label>
              <div className="grid grid-cols-2 gap-2">
                {["Circular", "Structuring", "Layering", "Dormant"].map(v => (
                  <AttackVector 
                    key={v} 
                    label={v} 
                    active={activeVectors.includes(v)} 
                    onClick={() => toggleVector(v)} 
                  />
                ))}
              </div>
            </div>
          </div>

          <div className="pt-4 border-t border-white/10 space-y-3">
            {!isSimulating ? (
              <button 
                onClick={() => setIsSimulating(true)}
                className="w-full py-4 bg-[#CAFF33] text-black font-black rounded-xl text-[10px] uppercase flex items-center justify-center gap-2 hover:bg-[#b8e62e] transition-colors"
              >
                <Play size={14} fill="currentColor" /> Initialize Simulation
              </button>
            ) : (
              <button 
                onClick={() => setIsSimulating(false)}
                className="w-full py-4 bg-white/5 text-white/60 font-black rounded-xl border border-white/10 text-[10px] uppercase flex items-center justify-center gap-2 hover:bg-white/10"
              >
                <Square size={14} fill="currentColor" /> Terminate Scenario
              </button>
            )}
          </div>
        </Card>

        {/* Mock System Health */}
        <Card className="p-5 flex-1 bg-gradient-to-b from-[#0a0a0a] to-[#050505]">
          <p className="text-[10px] font-black text-white/40 uppercase tracking-widest mb-6">Node Status</p>
          <div className="grid grid-cols-2 gap-y-6">
            <MetricSmall icon={Cpu} label="CPU Load" value={isSimulating ? 68 + Math.floor(Math.random()*10) : 12} unit="%" />
            <MetricSmall icon={HardDrive} label="Memory" value="4.2" unit="GB" />
            <MetricSmall icon={Database} label="DB IOPS" value={isSimulating ? "1.2k" : "0.1k"} unit="" />
            <MetricSmall icon={Activity} label="Latency" value={isSimulating ? "42" : "8"} unit="ms" />
          </div>
        </Card>
      </div>

      {/* Main Simulation View */}
      <div className="flex flex-col gap-6">
        <Card className="flex-1 flex flex-col relative overflow-hidden bg-black/40 border-dashed border-white/10">
          {!isSimulating ? (
            <div className="flex-1 flex flex-col items-center justify-center p-12 text-center">
              <div className="w-24 h-24 rounded-full border border-dashed border-white/10 flex items-center justify-center mb-8 relative">
                <div className="absolute inset-0 rounded-full bg-[#CAFF33]/5 animate-ping" />
                <Zap size={40} className="text-white/10" />
              </div>
              <h3 className="text-2xl font-black text-white mb-3 italic">Awaiting Parameters</h3>
              <p className="text-white/40 text-sm max-w-sm mx-auto font-medium">
                Configure synthetic fraud weights and transaction volume to begin the stress test.
              </p>
            </div>
          ) : (
            <div className="flex-1 flex flex-col p-8">
               <div className="flex items-center justify-between mb-8">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 text-[#CAFF33] bg-[#CAFF33]/10 px-3 py-1.5 rounded-full border border-[#CAFF33]/20">
                      <Loader2 className="animate-spin" size={14} />
                      <span className="text-[10px] font-black uppercase tracking-widest">Active Simulation</span>
                    </div>
                    <span className="text-[10px] font-mono text-white/40">ID: SIM-293-AX</span>
                  </div>
                  <div className="flex gap-1">
                    {[...Array(20)].map((_, i) => (
                      <div key={i} className={`h-1 w-2 rounded-full ${i/20 < progress/100 ? 'bg-[#CAFF33]' : 'bg-white/5'}`} />
                    ))}
                  </div>
               </div>

               {/* Mock Data Visualization Area */}
               <div className="flex-1 rounded-2xl bg-white/[0.02] border border-white/[0.05] p-6 mb-6 flex flex-col justify-center items-center">
                  <div className="text-center space-y-4">
                    <ShieldAlert size={48} className="text-red-500 mx-auto animate-pulse" />
                    <p className="text-3xl font-black text-white font-mono tracking-tighter">
                      {Math.floor(progress * 4.2)} <span className="text-white/20 text-lg">Alerts Triggered</span>
                    </p>
                    <p className="text-xs text-white/40 uppercase tracking-[0.2em] font-bold">Detection Precision: 98.2%</p>
                  </div>
               </div>

               {/* Terminal Style Logs */}
               <div className="h-48 bg-black/60 rounded-xl border border-white/5 p-4 font-mono text-[11px] overflow-hidden flex flex-col shadow-inner">
                  <div className="flex items-center gap-2 mb-3 text-white/30 border-b border-white/5 pb-2">
                    <Terminal size={12} />
                    <span className="uppercase tracking-widest text-[9px] font-bold">Pipeline Telemetry</span>
                  </div>
                  <div className="space-y-1.5 overflow-y-auto custom-scrollbar flex-1">
                    {logs.map((log, i) => (
                      <div key={i} className="flex gap-3">
                        <span className="text-white/20">[{new Date().toLocaleTimeString()}]</span>
                        <span className={log.type === 'err' ? 'text-red-400' : log.type === 'warn' ? 'text-yellow-400' : 'text-blue-400'}>
                          {log.type.toUpperCase()}
                        </span>
                        <span className="text-white/60">{log.msg}</span>
                      </div>
                    ))}
                  </div>
               </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};

// --- Sidebar Component ---
const Sidebar = ({ active, setActive }: { active: View, setActive: (v: View) => void }) => {
  return (
    <aside className="w-64 shrink-0 border-r border-white/[0.07] flex flex-col py-8 px-4 h-screen sticky top-0 bg-[#070707] z-20">
      <div className="mb-12 px-2 flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#CAFF33] to-[#7fff00] p-[1px]">
          <div className="w-full h-full rounded-xl bg-black flex items-center justify-center">
            <ShieldAlert className="text-[#CAFF33]" size={18} />
          </div>
        </div>
        <div>
          <h1 className="text-xl font-black text-white leading-none tracking-tight">AnomaNet</h1>
          <p className="text-[9px] text-white/40 font-bold uppercase tracking-[0.2em] mt-1">Intell. Platform</p>
        </div>
      </div>

      <nav className="flex-1 space-y-10">
        {NAV.map(group => (
          <div key={group.group}>
            <p className="text-[9px] font-black uppercase tracking-[0.3em] text-white/30 px-3 mb-3">{group.group}</p>
            <div className="space-y-1">
              {group.items.map(({ id, label, icon: Icon, notifications }) => {
                const isActive = active === id;
                return (
                  <button key={id} onClick={() => setActive(id)}
                    className={`w-full flex items-center gap-3.5 px-3 py-3 rounded-2xl text-left transition-all duration-200 group ${isActive ? "bg-white/[0.04] border border-white/10" : "hover:bg-white/[0.02]"}`}>
                    <Icon className={`w-4 h-4 shrink-0 transition-colors ${isActive ? "text-[#CAFF33]" : "text-white/40 group-hover:text-white/70"}`} />
                    <span className={`text-[12px] font-bold tracking-tight flex-1 ${isActive ? "text-white" : "text-white/50 group-hover:text-white/80"}`}>{label}</span>
                    {notifications && (
                      <span className="text-[9px] font-black bg-red-500 text-white min-w-[18px] h-[18px] flex items-center justify-center rounded-lg leading-none">
                        {notifications}
                      </span>
                    )}
                    {isActive && <div className="w-1.5 h-1.5 rounded-full bg-[#CAFF33]" style={{boxShadow: '0 0 10px #CAFF33'}} />}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="mt-auto border-t border-white/[0.07] pt-6 px-2">
        <div className="bg-white/5 rounded-2xl p-4 flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-[10px] font-black border border-white/10">SR</div>
          <div className="flex-1 min-w-0">
             <p className="text-xs font-bold text-white truncate leading-none">S. Rathore</p>
             <p className="text-[9px] text-white/40 font-bold uppercase mt-1">Lead Investigator</p>
          </div>
          <button className="text-white/20 hover:text-white"><MoreHorizontal size={16}/></button>
        </div>
      </div>
    </aside>
  );
};

// --- Main Layout ---
export default function FraudDashboard() {
  const [active, setActive] = useState<View>("dashboard");

  const renderSection = () => {
    switch (active) {
      case "dashboard": return <DashboardSection />;
      case "alerts":    return <AlertsSection />;
      case "graph":     return <GraphExplorerSection />;
      case "cases":     return <CasesSection />;
      case "reports":   return <ReportsSection />;
      case "simulator": return <SimulatorSection />;
      default:          return <DashboardSection />;
    }
  };

  return (
  <div className="bg-[#060606] text-white min-h-screen flex flex-col">
    <Navbar />
    <div className="flex flex-1 overflow-hidden">
      <Sidebar active={active} setActive={setActive} />
      <main className="flex-1 h-full overflow-hidden">
        <div className="w-full h-full px-6 py-6 flex flex-col">
          {renderSection()}
        </div>
      </main>
    </div>
    <Footer />
  </div>
);
}