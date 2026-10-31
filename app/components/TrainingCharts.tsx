"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { IterationQuality, Metric } from "../types";

// Categorical slots 1-4 of the validated dark palette, in fixed order.
export const SERIES = { blue: "#3987e5", orange: "#d95926", aqua: "#199e70", yellow: "#c98500" } as const;

type Series = { key: string; label: string; color: string; values: (number | null)[] };

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(640);
  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(240, entry.contentRect.width)));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

function niceMax(value: number) {
  if (value <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  return Math.ceil(value / power * 2) / 2 * power;
}

export function LineChart({ x, xLabel, series, yMax, format, height = 220, label, directLabels = true }: {
  x: number[]; xLabel: string; series: Series[]; yMax?: number; format: (value: number) => string;
  height?: number; label: string; directLabels?: boolean;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const pad = { top: 12, right: directLabels && series.length > 1 ? 92 : 16, bottom: 26, left: 46 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const max = yMax ?? niceMax(Math.max(0, ...series.flatMap((item) => item.values.filter((v): v is number => v != null))));
  const px = (index: number) => pad.left + (x.length <= 1 ? plotW / 2 : index / (x.length - 1) * plotW);
  const py = (value: number) => pad.top + plotH - value / max * plotH;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => fraction * max);
  const paths = series.map((item) => {
    let d = "";
    item.values.forEach((value, index) => {
      if (value == null) return;
      d += `${d && item.values[index - 1] != null ? "L" : "M"}${px(index).toFixed(1)},${py(value).toFixed(1)}`;
    });
    return d;
  });
  const lastIndex = (values: (number | null)[]) => { for (let i = values.length - 1; i >= 0; i -= 1) if (values[i] != null) return i; return -1; };
  const onMove = (event: React.PointerEvent<SVGRectElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const ratio = (event.clientX - box.left) / box.width;
    setHover(Math.max(0, Math.min(x.length - 1, Math.round(ratio * (x.length - 1)))));
  };
  // Spread direct labels so they never overlap (min 13px apart).
  const endLabels = series.map((item, index) => ({ item, index, at: lastIndex(item.values) }))
    .filter((entry) => entry.at >= 0)
    .map((entry) => ({ ...entry, y: py(entry.item.values[entry.at] as number) }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < endLabels.length; i += 1) endLabels[i].y = Math.max(endLabels[i].y, endLabels[i - 1].y + 13);

  return <div className="lineChart" ref={ref}>
    <svg width={width} height={height} role="img" aria-label={label}>
      {ticks.map((tick) => <g key={tick}>
        <line x1={pad.left} x2={pad.left + plotW} y1={py(tick)} y2={py(tick)} className="gridLine" />
        <text x={pad.left - 8} y={py(tick) + 3} textAnchor="end" className="axisText">{format(tick)}</text>
      </g>)}
      {x.length > 0 && <>
        <text x={pad.left} y={height - 8} className="axisText">{xLabel} {x[0]}</text>
        <text x={pad.left + plotW} y={height - 8} textAnchor="end" className="axisText">{xLabel} {x.at(-1)}</text>
      </>}
      {series.map((item, index) => <path key={item.key} d={paths[index]} stroke={item.color} className="seriesLine" />)}
      {series.map((item) => { const at = lastIndex(item.values); return at < 0 ? null : <circle key={item.key} cx={px(at)} cy={py(item.values[at] as number)} r={4} fill={item.color} className="endDot" />; })}
      {directLabels && series.length > 1 && endLabels.map(({ item, y }) => <text key={item.key} x={pad.left + plotW + 10} y={y + 3} className="directLabel">{item.label}</text>)}
      {hover != null && <line x1={px(hover)} x2={px(hover)} y1={pad.top} y2={pad.top + plotH} className="crosshair" />}
      {hover != null && series.map((item) => item.values[hover] == null ? null : <circle key={item.key} cx={px(hover)} cy={py(item.values[hover] as number)} r={4} fill={item.color} className="endDot" />)}
      <rect x={pad.left} y={pad.top} width={Math.max(1, plotW)} height={plotH} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
    </svg>
    {hover != null && <div className="chartTooltip" style={{ left: Math.min(px(hover) + 12, width - 190), top: pad.top }}>
      <b>{xLabel} {x[hover]}</b>
      {series.map((item) => <span key={item.key}><i style={{ background: item.color }} />{item.label}<em>{item.values[hover] == null ? "—" : format(item.values[hover] as number)}</em></span>)}
    </div>}
  </div>;
}

export function Legend({ items }: { items: { label: string; color: string; note?: string }[] }) {
  return <div className="chartKey">{items.map((item) => <span key={item.label}><i style={{ background: item.color }} />{item.label}{item.note && <small>{item.note}</small>}</span>)}</div>;
}

const fixed = (digits: number) => (value: number) => value.toFixed(digits);
const pct = (value: number) => `${Math.round(value * 100)}%`;

function byIteration(metrics: Metric[]) {
  const rows = new Map<number, Metric>();
  for (const metric of metrics) rows.set(metric.iteration, metric);
  return [...rows.values()].sort((a, b) => a.iteration - b.iteration);
}

export function LossCharts({ metrics }: { metrics: Metric[] }) {
  const rows = useMemo(() => byIteration(metrics), [metrics]);
  const x = rows.map((row) => row.iteration);
  const panels = [
    { key: "policy", label: "Policy loss", note: "cross-entropy vs MCTS visits", color: SERIES.blue, values: rows.map((row) => row.policy_loss) },
    { key: "value", label: "Value loss", note: "MSE vs game result", color: SERIES.orange, values: rows.map((row) => row.value_loss) },
    { key: "total", label: "Total loss", note: "policy + value", color: SERIES.aqua, values: rows.map((row) => row.total_loss) },
  ];
  return <div>
    <Legend items={panels} />
    <div className="smallMultiples">{panels.map((panel) => <figure key={panel.key}>
      <figcaption><i style={{ background: panel.color }} />{panel.label}</figcaption>
      <LineChart x={x} xLabel="ITER" series={[panel]} format={fixed(panel.key === "value" ? 2 : 1)} height={170} label={`${panel.label} by iteration`} />
    </figure>)}</div>
  </div>;
}

export function QualityChart({ rows }: { rows: IterationQuality[] }) {
  const x = rows.map((row) => row.iteration);
  const series: Series[] = [
    { key: "draw", label: "Draw", color: SERIES.blue, values: rows.map((row) => row.rolling.draw_rate) },
    { key: "repetition", label: "Repetition", color: SERIES.orange, values: rows.map((row) => row.rolling.repetition_rate) },
    { key: "checkmate", label: "Checkmate", color: SERIES.aqua, values: rows.map((row) => row.rolling.checkmate_rate) },
    { key: "decisive", label: "Decisive", color: SERIES.yellow, values: rows.map((row) => row.rolling.decisive_rate) },
  ];
  return <div>
    <Legend items={series} />
    <LineChart x={x} xLabel="ITER" series={series} yMax={1} format={pct} label="Rolling self-play outcome rates by iteration" />
  </div>;
}

export function ThroughputChart({ metrics }: { metrics: Metric[] }) {
  const rows = useMemo(() => byIteration(metrics), [metrics]);
  const series: Series[] = [
    { key: "selfplay", label: "Self-play only", color: SERIES.blue, values: rows.map((row) => row.selfplay_games_per_minute ?? null) },
    { key: "wall", label: "Incl. optimizer", color: SERIES.orange, values: rows.map((row) => row.throughput) },
  ];
  return <div>
    <Legend items={[
      { ...series[0], note: "games ÷ self-play phase time" },
      { ...series[1], note: "games ÷ (self-play + optimizer) time" },
    ]} />
    <LineChart x={rows.map((row) => row.iteration)} xLabel="ITER" series={series} format={fixed(1)} height={200} label="Games per minute by iteration, self-play only and including optimizer" />
  </div>;
}
