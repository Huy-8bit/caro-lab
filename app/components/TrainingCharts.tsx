"use client";

import type { Metric } from "../types";

function sampled(metrics: Metric[], count = 56) {
  if (metrics.length <= count) return metrics;
  const stride = metrics.length / count;
  return Array.from({ length: count }, (_, index) => metrics[Math.floor(index * stride)]);
}

export function TrainingLossChart({ metrics }: { metrics: Metric[] }) {
  const values = sampled(metrics);
  const max = Math.max(...values.map((metric) => metric.total_loss), 0.001);
  return <div className="nativeChart" role="img" aria-label="Policy, value and total loss by optimizer step"><div className="chartLegend"><span className="policy">Policy loss</span><span className="value">Value loss</span><span className="total">Total loss</span></div><div className="chartPlot">{values.map((metric) => <div className="chartGroup" key={metric.step} title={`Step ${metric.step}: policy ${metric.policy_loss.toFixed(3)}, value ${metric.value_loss.toFixed(3)}, total ${metric.total_loss.toFixed(3)}`}><i className="policy" style={{ height: `${Math.max(1, metric.policy_loss / max * 100)}%` }} /><i className="value" style={{ height: `${Math.max(1, metric.value_loss / max * 100)}%` }} /><i className="total" style={{ height: `${Math.max(1, metric.total_loss / max * 100)}%` }} /></div>)}</div><div className="chartAxis"><span>STEP {values[0]?.step}</span><span>STEP {values.at(-1)?.step}</span></div></div>;
}

export function ThroughputChart({ metrics }: { metrics: Metric[] }) {
  const values = sampled(metrics, 40);
  const max = Math.max(...values.map((metric) => metric.throughput), 0.001);
  return <div className="nativeChart throughputChart" role="img" aria-label="Self-play games per minute by iteration"><div className="chartPlot">{values.map((metric) => <div className="chartGroup single" key={metric.step} title={`Iteration ${metric.iteration}: ${metric.throughput.toFixed(2)} games/min`}><i style={{ height: `${Math.max(2, metric.throughput / max * 100)}%` }} /></div>)}</div><div className="chartAxis"><span>ITER {values[0]?.iteration}</span><span>{max.toFixed(1)} GAMES/MIN PEAK</span><span>ITER {values.at(-1)?.iteration}</span></div></div>;
}
