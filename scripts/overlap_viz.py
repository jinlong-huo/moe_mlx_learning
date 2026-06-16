#!/usr/bin/env python3
"""MoE pipeline overlap visualizer — micro-batch view.

Groups events by micro-batch (not training step) so cross-MB overlap is visible:
  mb_0:  [scatter ========][compute ==][gather ========]
  mb_1:           [scatter ========][compute ==][gather ========]
  mb_2:                    [scatter ========][compute ==][gather ========]
                         ↑ OVERLAP: scatter_1 ∥ compute_0

Usage:
  python scripts/overlap_viz.py outputs/traces/merged_32gpu.json -r 8
  python scripts/overlap_viz.py outputs/traces/merged_32gpu.json -r 8 -o out.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys

COLORS = {
    "scatter": "#42A5F5", "scatter_wait": "#90CAF9",
    "gather": "#AB47BC", "gather_wait": "#CE93D8",
    "compute": "#FF9800", "combine": "#E91E63",
    "route": "#4CAF50", "barrier": "#78909C", "all_to_all": "#78909C",
}
STREAM_MAP = {
    "scatter": "comm", "scatter_wait": "comm",
    "gather": "comm", "gather_wait": "comm",
    "compute": "comp", "combine": "comp", "route": "comp",
    "barrier": "sync", "all_to_all": "sync",
}


def load_events(path: str, rank: int | None = None) -> tuple[list[dict], str, str]:
    with open(path) as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    if rank is not None:
        events = [e for e in events if e.get("pid") == rank]
    for e in events:
        e["_dur"] = int(e.get("dur", 0))
        e["_start"] = int(e["ts"])
        e["_end"] = e["_start"] + e["_dur"]
        e["_op"] = e["name"].rsplit("/", 1)[-1]
        e["end"] = e["_end"]
        e["op"] = e["_op"]
    source = os.path.basename(path)
    if rank is not None:
        source += f" · rank_{rank:02d}"
    has_ov = any(e["_op"] == "scatter_wait" for e in events)
    mode = "overlap" if has_ov else "serial"
    return events, source, mode


def build_html(events: list[dict], source_label: str, mode: str) -> str:
    ev_json = json.dumps(events, separators=(",", ":"))
    co_json = json.dumps(COLORS, separators=(",", ":"))
    sm_json = json.dumps(STREAM_MAP, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MoE Pipeline Overlap — {mode.upper()}</title>
<style>
:root{{--bg:#0d1117;--sf:#161b22;--tx:#c9d1d9;--t2:#8b949e;--bd:#30363d;--ac:#58a6ff;--ov:#3fb950;--pi:#d29922;--cm:#42A5F5;--cp:#FF9800}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'SF Mono',Menlo,monospace;background:var(--bg);color:var(--tx);display:flex;flex-direction:column;height:100vh;overflow:hidden}}
.hdr{{background:var(--sf);border-bottom:1px solid var(--bd);padding:8px 20px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}}
.hdr h1{{font-size:.95rem;font-weight:600}}
.badge{{display:inline-flex;padding:2px 10px;border-radius:10px;font-size:.62rem;font-weight:700;text-transform:uppercase;margin-left:8px}}
.dash{{display:flex;gap:1px;background:var(--bd);border-bottom:1px solid var(--bd);flex-shrink:0}}
.dash .card{{flex:1;background:var(--sf);padding:8px 14px;display:flex;flex-direction:column;gap:1px;min-width:100px}}
.dash .card .l{{font-size:.58rem;color:var(--t2);text-transform:uppercase;letter-spacing:.04em}}
.dash .card .v{{font-size:1.05rem;font-weight:700}}
.dash .card .d{{font-size:.55rem;color:var(--t2)}}
.vg{{color:var(--ov)}}.vo{{color:var(--pi)}}
.bar{{display:flex;gap:12px;padding:4px 20px;background:var(--sf);border-bottom:1px solid var(--bd);flex-shrink:0;font-size:.62rem;align-items:center;flex-wrap:wrap}}
.bar .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:3px}}
.bar .sp{{color:var(--bd);margin:0 4px}}
.bar label{{display:inline-flex;align-items:center;gap:3px;color:var(--t2);margin-left:6px;cursor:pointer}}
.bar input[type=range]{{width:60px;accent-color:var(--pi)}}
.bar select{{background:var(--bd);color:var(--tx);border:1px solid var(--bd);padding:2px 6px;border-radius:3px;font-size:.6rem;font-family:inherit}}
.main{{flex:1;overflow:auto;position:relative}}
canvas{{display:block}}
.tip{{position:fixed;pointer-events:none;background:#0d1117;border:1px solid var(--bd);border-radius:8px;padding:10px 14px;font-size:.68rem;z-index:100;display:none;min-width:230px;box-shadow:0 8px 24px rgba(0,0,0,.6)}}
.tip .tn{{font-weight:600;margin-bottom:5px;font-size:.72rem}}
.tip .tr{{display:flex;justify-content:space-between;gap:14px;margin:2px 0}}
.tip .tr span:first-child{{color:var(--t2)}}
.tip .to{{margin-top:6px;padding-top:5px;border-top:1px solid var(--bd)}}
</style>
</head>
<body>
<div class="hdr">
  <h1>🔷 MoE Micro-Batch Pipeline Overlap<span class="badge" style="background:{'#3fb95020' if mode == 'overlap' else '#d2992220'};color:{'#3fb950' if mode == 'overlap' else '#d29922'}">{mode.upper()}</span></h1>
  <span style="font-size:.65rem;color:var(--t2)">{source_label} · scroll-zoom · drag-pan · hover</span>
</div>
<div class="dash" id="dash"></div>
<div class="bar">
  <span style="color:var(--t2)">Step:</span><select id="stepSel"></select>
  <span class="sp">‖</span>
  <span><span class="sw" style="background:var(--cm)"></span>comm</span>
  <span><span class="sw" style="background:var(--cp)"></span>comp</span>
  <span><span class="sw" style="background:#78909C"></span>sync</span>
  <span class="sp">‖</span>
  <span><span class="sw" style="background:rgba(63,185,80,.5);border:1px solid #3fb950"></span>cross-MB overlap</span>
  <span><span class="sw" style="background:rgba(210,153,34,.4);border:1px dashed #d29922"></span>pipeline zone</span>
  <label>gap≤<input type="range" id="gapSlider" min="100" max="50000" value="5000" step="100"><span id="gapVal">5.0µs</span></label>
</div>
<div class="main" id="main"><canvas id="c"></canvas></div>
<div class="tip" id="tip"></div>

<script>
const EV={ev_json};
const CO={co_json};
const SM={sm_json};
const ROW_H=26,ROW_GAP=6,PAD_L=170,PAD_R=50,PAD_T=20,PAD_B=30;
let PIPELINE_GAP_NS=5000;

// Group by micro-batch
const mbGroups=new Map();
let t0=Infinity,t1=-Infinity;
for(const e of EV){{
  if(e.ts<t0)t0=e.ts;if(e.end>t1)t1=e.end;
  const p=e.name.split('/');
  if(p.length>=3){{const k=p.slice(0,3).join('/');if(!mbGroups.has(k))mbGroups.set(k,[]);mbGroups.get(k).push(e);}}
}}
const mbKeys=[...mbGroups.keys()].sort();
const steps=[...new Set(mbKeys.map(k=>k.split('/').slice(0,2).join('/')))].sort();

const cv=document.getElementById('c'),ctx=cv.getContext('2d');
const wrap=document.getElementById('main');
let scaleX=1.0,offsetX=0,dragging=false,dragStart=0;
let viewStep=steps[0];

function visibleMBs(){{return mbKeys.filter(k=>k.startsWith(viewStep+'/'));}}
function totalH(){{const v=visibleMBs();return PAD_T+v.length*(ROW_H+ROW_GAP)+ROW_GAP+PAD_B;}}
function resize(){{cv.width=Math.max(wrap.clientWidth,((t1-t0)/1000)*scaleX+PAD_L+PAD_R+200);cv.height=Math.max(totalH(),200);}}
function toX(ts){{return PAD_L+((ts-t0)/1000)*scaleX+offsetX;}}
function fmtUs(v){{return(v/1000).toFixed(0)+'µs';}}
function fmtMs(v){{return(v/1e6).toFixed(2)+'ms';}}

// Cross-MB overlap: comm event in one MB ∥ comp event in a DIFFERENT MB
function findCrossOverlaps(mbs){{
  const results=[];
  for(let i=0;i<mbs.length;i++){{
    for(let j=i+1;j<mbs.length;j++){{
      const ea=mbGroups.get(mbs[i]),eb=mbGroups.get(mbs[j]);
      for(const [a,b] of[[ea,eb],[eb,ea]]){{
        for(const ca of a.filter(e=>SM[e.op]==='comm')){{
          for(const cb of b.filter(e=>SM[e.op]==='comp')){{
            const os=Math.max(ca.ts,cb.ts),oe=Math.min(ca.end,cb.end);
            const gap=os-oe;
            if(gap<PIPELINE_GAP_NS){{
              results.push({{start:os,end:oe,dur:Math.max(0,oe-os),gap,isTrue:gap<=0,
                commMB:mbs[i],compMB:mbs[j],
                commName:ca.name.split('/').slice(2).join('/'),
                compName:cb.name.split('/').slice(2).join('/')}});
            }}
          }}
        }}
      }}
    }}
  }}
  return results;
}}

function draw(){{
  resize();
  const vmbs=visibleMBs(),crossOv=findCrossOverlaps(vmbs);
  ctx.clearRect(0,0,cv.width,cv.height);
  const w=cv.width,h=cv.height;
  ctx.strokeStyle='#21262d';ctx.lineWidth=.5;
  for(let x=PAD_L+offsetX;x<w;x+=100*scaleX){{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}}

  // Draw cross-MB overlap bands
  for(const ov of crossOv){{
    const iA=vmbs.indexOf(ov.commMB),iB=vmbs.indexOf(ov.compMB);
    if(iA<0||iB<0)continue;
    const topY=PAD_T+Math.min(iA,iB)*(ROW_H+ROW_GAP),botY=PAD_T+Math.max(iA,iB)*(ROW_H+ROW_GAP)+ROW_H;
    const es=ov.isTrue?ov.start:Math.max(t0,ov.start-ov.gap);
    const ee=ov.isTrue?ov.end:Math.min(t1,ov.end+ov.gap);
    const ox=toX(es),ow=Math.max(((ee-es)/1000)*scaleX,3);
    if(ov.isTrue){{
      const g=ctx.createLinearGradient(0,topY,0,botY);
      g.addColorStop(0,'rgba(63,185,80,0.03)');g.addColorStop(0.5,'rgba(63,185,80,0.2)');g.addColorStop(1,'rgba(63,185,80,0.03)');
      ctx.fillStyle=g;ctx.fillRect(ox,topY,ow,botY-topY);
      ctx.strokeStyle='rgba(63,185,80,.55)';ctx.lineWidth=1.5;ctx.setLineDash([4,4]);
      ctx.beginPath();ctx.moveTo(ox,topY);ctx.lineTo(ox+ow,topY);ctx.stroke();
      ctx.beginPath();ctx.moveTo(ox,botY);ctx.lineTo(ox+ow,botY);ctx.stroke();ctx.setLineDash([]);
      if(ow>60){{ctx.fillStyle='#3fb950';ctx.font='bold 8px monospace';ctx.textAlign='center';ctx.fillText('OVERLAP '+fmtUs(ov.dur),ox+ow/2,topY-2);}}
    }}else{{
      const g=ctx.createLinearGradient(0,topY,0,botY);
      g.addColorStop(0,'rgba(210,153,34,0.02)');g.addColorStop(0.5,'rgba(210,153,34,0.14)');g.addColorStop(1,'rgba(210,153,34,0.02)');
      ctx.fillStyle=g;ctx.fillRect(ox,topY,ow,botY-topY);
      ctx.strokeStyle='rgba(210,153,34,.45)';ctx.lineWidth=1;ctx.setLineDash([2,5]);
      ctx.beginPath();ctx.moveTo(ox,topY);ctx.lineTo(ox+ow,topY);ctx.stroke();
      ctx.beginPath();ctx.moveTo(ox,botY);ctx.lineTo(ox+ow,botY);ctx.stroke();ctx.setLineDash([]);
      if(ow>55){{ctx.fillStyle='#d29922';ctx.font='7px monospace';ctx.textAlign='center';ctx.fillText('PIPE '+fmtUs(ov.gap)+' gap',ox+ow/2,botY+10);}}
    }}
  }}

  // Draw MB rows
  vmbs.forEach((mbKey,idx)=>{{
    const y=PAD_T+idx*(ROW_H+ROW_GAP);
    const evs=mbGroups.get(mbKey);
    const parts=mbKey.split('/');
    const label=parts[1]+' '+parts[2];
    ctx.fillStyle='#8b949e';ctx.font='bold 10px monospace';ctx.textAlign='right';
    ctx.fillText(label,PAD_L-8,y+17);
    ctx.fillStyle=idx%2===0?'rgba(22,27,34,.4)':'rgba(13,17,23,.25)';
    ctx.fillRect(PAD_L,y,w-PAD_L-PAD_R,ROW_H);
    ctx.fillStyle='rgba(48,54,61,.4)';ctx.fillRect(PAD_L,y,w-PAD_L-PAD_R,1);
    ctx.fillRect(PAD_L,y+ROW_H-1,w-PAD_L-PAD_R,1);
    for(const e of [...evs].sort((a,b)=>a.ts-b.ts)){{
      const x=toX(e.ts),bw=Math.max((e.dur/1000)*scaleX,2);
      const bc=CO[e.op]||'#546E7A';
      const by2=SM[e.op]==='sync'?y+8:y+4,bh=SM[e.op]==='sync'?ROW_H-16:ROW_H-8;
      const g=ctx.createLinearGradient(x,by2,x,by2+bh);
      g.addColorStop(0,bc);g.addColorStop(1,bc+'88');
      ctx.fillStyle=g;ctx.fillRect(x,by2,bw,bh);
      ctx.strokeStyle=bc;ctx.lineWidth=.5;ctx.strokeRect(x,by2,bw,bh);
      if(bw>e.op.length*6+8){{ctx.fillStyle='#fff';ctx.font='8px monospace';ctx.textAlign='left';ctx.fillText(e.op,x+4,by2+bh-5);}}
    }}
  }});

  const ay=h-PAD_B+12;
  ctx.strokeStyle='#30363d';ctx.beginPath();ctx.moveTo(PAD_L,ay);ctx.lineTo(w-PAD_R,ay);ctx.stroke();
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';ctx.textAlign='center';
  for(let x=PAD_L+offsetX;x<w;x+=200*scaleX){{ctx.fillText(((x-PAD_L-offsetX)/scaleX).toFixed(0)+'µs',x,ay+14);}}
}}

function buildDash(){{
  const vmbs=visibleMBs(),crossOv=findCrossOverlaps(vmbs);
  const trueC=crossOv.filter(o=>o.isTrue).length,nearC=crossOv.filter(o=>!o.isTrue).length;
  const tc=vmbs.reduce((s,k)=>s+mbGroups.get(k).filter(e=>SM[e.op]==='comm').reduce((a,e)=>a+e.dur,0),0);
  const tp=vmbs.reduce((s,k)=>s+mbGroups.get(k).filter(e=>SM[e.op]==='comp').reduce((a,e)=>a+e.dur,0),0);
  document.getElementById('dash').innerHTML=`
    <div class="card"><span class="l">Step</span><span class="v" style="color:var(--ac)">${{viewStep}}</span><span class="d">${{vmbs.length}} micro-batches</span></div>
    <div class="card"><span class="l">Cross-MB Overlaps</span><span class="v vg">${{trueC}}</span><span class="d">comm in one MB ∥ comp in another</span></div>
    <div class="card"><span class="l">Pipeline Zones</span><span class="v vo">${{nearC}}</span><span class="d">gap ≤ ${{(PIPELINE_GAP_NS/1000).toFixed(1)}}µs</span></div>
    <div class="card"><span class="l">Total Comm</span><span class="v" style="color:var(--cm)">${{fmtMs(tc)}}</span><span class="d">scatter + gather</span></div>
    <div class="card"><span class="l">Total Comp</span><span class="v" style="color:var(--cp)">${{fmtMs(tp)}}</span><span class="d">compute + combine + route</span></div>
    <div class="card"><span class="l">Pipeline Efficiency</span><span class="v vg">${{tc+tp>0?((tc+tp-(trueC+nearC)*50)/(tc+tp)*100).toFixed(1):'0'}}%</span><span class="d">overlap indicator</span></div>
  `;
}}

const ss=document.getElementById('stepSel');
steps.forEach(s=>{{const o=document.createElement('option');o.value=s;o.textContent=s;ss.appendChild(o);}});
ss.value=viewStep;
ss.addEventListener('change',()=>{{viewStep=ss.value;buildDash();draw();}});
document.getElementById('gapSlider').addEventListener('input',function(){{
  PIPELINE_GAP_NS=+this.value;
  document.getElementById('gapVal').textContent=(PIPELINE_GAP_NS/1000).toFixed(1)+'µs';
  buildDash();draw();
}});

cv.addEventListener('wheel',e=>{{e.preventDefault();const z=e.deltaY<0?1.2:1/1.2,mx=e.offsetX;offsetX=mx-(mx-offsetX)*z;scaleX*=z;scaleX=Math.max(0.02,Math.min(scaleX,60));draw();}});
cv.addEventListener('mousedown',e=>{{dragging=true;dragStart=e.offsetX-offsetX;}});
cv.addEventListener('mouseup',()=>{{dragging=false;}});
cv.addEventListener('mouseleave',()=>{{dragging=false;}});
cv.addEventListener('mousemove',e=>{{
  if(dragging){{offsetX=e.offsetX-dragStart;draw();return;}}
  const mx=e.offsetX,my=e.offsetY,vmbs=visibleMBs();
  let found=null,foundOv=null;
  for(let i=0;i<vmbs.length;i++){{
    const y=PAD_T+i*(ROW_H+ROW_GAP);
    if(my>=y&&my<=y+ROW_H)for(const e of mbGroups.get(vmbs[i])){{
      const ex=toX(e.ts),ebw=Math.max((e.dur/1000)*scaleX,2);
      if(mx>=ex&&mx<=ex+ebw){{found={{...e,mb:vmbs[i]}};break;}}
    }}
  }}
  if(!found){{const co=findCrossOverlaps(vmbs);
    for(const ov of co){{
      const iA=vmbs.indexOf(ov.commMB),iB=vmbs.indexOf(ov.compMB);
      if(iA<0||iB<0)continue;
      const topY=PAD_T+Math.min(iA,iB)*(ROW_H+ROW_GAP),botY=PAD_T+Math.max(iA,iB)*(ROW_H+ROW_GAP)+ROW_H;
      const es=ov.isTrue?ov.start:Math.max(t0,ov.start-ov.gap),ee=ov.isTrue?ov.end:Math.min(t1,ov.end+ov.gap);
      const ox=toX(es),ow=Math.max(((ee-es)/1000)*scaleX,3);
      if(mx>=ox&&mx<=ox+ow&&my>=topY&&my<=botY){{foundOv=ov;break;}}
    }}
  }}
  const tip=document.getElementById('tip');
  if(found){{
    tip.style.display='block';tip.style.left=(e.pageX+12)+'px';tip.style.top=(e.pageY-10)+'px';
    tip.innerHTML=`<div class="tn" style="color:${{CO[found.op]}}">${{found.mb.split('/').slice(1).join(' ')}} / ${{found.op}}</div><div class="tr"><span>stream</span><span style="color:${{SM[found.op]==='comm'?'#58a6ff':SM[found.op]==='comp'?'#d29922':'#78909C'}}">${{SM[found.op].toUpperCase()}}</span></div><div class="tr"><span>start</span><span>${{fmtUs(found.ts)}}</span></div><div class="tr"><span>duration</span><span>${{fmtUs(found.dur)}}</span></div><div class="tr"><span>end</span><span>${{fmtUs(found.end)}}</span></div>`;
  }}else if(foundOv){{
    tip.style.display='block';tip.style.left=(e.pageX+12)+'px';tip.style.top=(e.pageY-10)+'px';
    tip.innerHTML=`<div class="tn" style="color:${{foundOv.isTrue?'#3fb950':'#d29922'}}">${{foundOv.isTrue?'Cross-MB Overlap':'Pipeline Zone'}}</div><div class="tr"><span>comm MB</span><span style="color:#58a6ff">${{foundOv.commMB.split('/').slice(1).join(' ')}} (${{foundOv.commName}})</span></div><div class="tr"><span>comp MB</span><span style="color:#d29922">${{foundOv.compMB.split('/').slice(1).join(' ')}} (${{foundOv.compName}})</span></div>${{foundOv.isTrue?'<div class="to" style="color:#3fb950">concurrent: <b>${{fmtUs(foundOv.dur)}}</b></div>':'<div class="to" style="color:#d29922">gap: <b>${{fmtUs(foundOv.gap)}}</b> — back-to-back in pipeline</div>'}}`;
  }}else{{tip.style.display='none';}}
}});

buildDash();draw();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="MoE pipeline overlap visualizer")
    parser.add_argument("trace", help="Path to trace JSON")
    parser.add_argument("-r", "--rank", type=int, default=None, help="Filter to rank PID")
    parser.add_argument("-o", "--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    trace_path = os.path.abspath(args.trace)
    if not os.path.isfile(trace_path):
        print(f"ERROR: {trace_path} not found", file=sys.stderr)
        sys.exit(1)

    events, source_label, mode = load_events(trace_path, args.rank)
    if not events:
        print("ERROR: No events found", file=sys.stderr)
        sys.exit(1)

    html = build_html(events, source_label, mode)

    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(trace_path)[0]
        suffix = f"_rank{args.rank:02d}" if args.rank is not None else ""
        out_path = f"{base}{suffix}_overlap.html"

    with open(out_path, "w") as f:
        f.write(html)

    mbs = sorted(set(
        "/".join(e["name"].split("/")[:3])
        for e in events if e["name"].count("/") >= 3
    ))
    steps = sorted(set(mb.rsplit("/", 1)[0] for mb in mbs))
    total_comm = sum(e["_dur"] for e in events if STREAM_MAP.get(e["_op"]) == "comm")
    total_comp = sum(e["_dur"] for e in events if STREAM_MAP.get(e["_op"]) == "comp")

    print(f"✓ {out_path}")
    print(f"  Events: {len(events)}  |  Steps: {len(steps)}  |  MBs/step: {len(mbs)//max(1,len(steps))}")
    print(f"  Mode: {mode.upper()}  |  Comm: {total_comm/1e3:.1f}µs  |  Comp: {total_comp/1e3:.1f}µs")


if __name__ == "__main__":
    main()
