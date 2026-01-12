import React from "react";

function HelpCard({ title, children, active }) {
  return (
    <div className={"helpCard" + (active ? " helpCardActive" : "")}>
      <div className="helpTitle">{title}</div>
      <div className="helpBody">{children}</div>
    </div>
  );
}

export default function MetricHelp({
  activityMethod,
  zThresh,
  aucThresh,
  meanThresh,
  promThresh,
  baselineFrac,
  minPeaks,
}) {
  const is = (m) => (activityMethod || "").toLowerCase() === m;
  const isComposite = is("composite");

  return (
    <div className="helpPanel">
      <div className="helpHeader">What do these mean?</div>

      <HelpCard title="Time window (Start–End) + dt" active={true}>
        We slice each neuron trace to your selected time window. The x-axis uses{" "}
        <b>dt</b> seconds per column, so times are in seconds.
      </HelpCard>

      <HelpCard
        title="Window-relative Z-score max (max_z)"
        active={is("max_z") || isComposite}
      >
        We compute z-scores <b>inside the selected window only</b>:
        <div className="eq">
          z(t) = (x(t) − μ_window) / (σ_window + ε)
        </div>
        Then we take <b>max_z = max(z(t))</b>. A neuron is “active” if{" "}
        <b>max_z ≥ {Number(zThresh).toFixed(2)}</b>.
      </HelpCard>

      <HelpCard title="AUC threshold (Area Under Curve)" active={is("auc") || isComposite}>
        We subtract a baseline and integrate over time:
        <div className="eq">
          AUC = ∫ (x(t) − baseline) dt
        </div>
        Higher AUC means more total activity (sustained responders). Active if{" "}
        <b>AUC ≥ {Number(aucThresh).toFixed(2)}</b>.
      </HelpCard>

      <HelpCard
        title="Mean over baseline threshold"
        active={is("mean_over_baseline") || isComposite}
      >
        Baseline is computed from the first <b>{Math.round(Number(baselineFrac) * 100)}%</b>{" "}
        of the selected window. Then:
        <div className="eq">mean_over_baseline = mean(x(t) − baseline)</div>
        Active if <b>mean_over_baseline ≥ {Number(meanThresh).toFixed(2)}</b>.
      </HelpCard>

      <HelpCard title="Peak prominence threshold" active={is("prominence") || isComposite}>
        We find peaks after baseline subtraction and measure how much a peak stands
        out from its surrounding valleys (“prominence”). This rejects one-frame noise spikes.
        Active if <b>prominence ≥ {Number(promThresh).toFixed(2)}</b> and{" "}
        <b>#peaks ≥ {Number(minPeaks)}</b>.
      </HelpCard>

      <HelpCard title="Composite (robust OR)" active={isComposite}>
        Composite labels a neuron active if it passes <b>any</b> of the enabled
        criteria (z, AUC, mean, or prominence). This catches both transient and sustained activity.
      </HelpCard>
    </div>
  );
}
