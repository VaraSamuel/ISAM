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
  minAboveSec,
  aucThresh,
  meanThresh,
  promThresh,
  baselineFrac,
  minPeaks,
}) {
  const m = (activityMethod || "").toLowerCase();
  const isComposite = m === "composite";

  return (
    <div className="helpPanel">
      <div className="helpHeader">What do these mean?</div>

      <HelpCard title="Threshold + Duration (z_duration)" active={m === "z_duration"}>
        We compute z-scores <b>inside the selected window</b>:
        <div className="eq">z(t) = (x(t) − μ_window) / (σ_window + ε)</div>
        A neuron is <b>active only if BOTH</b>:
        <ul style={{ margin: "6px 0 0 18px" }}>
          <li>
            z(t) goes above <b>{Number(zThresh).toFixed(2)}</b>, and
          </li>
          <li>
            it stays above that threshold continuously for at least{" "}
            <b>{Number(minAboveSec).toFixed(1)}s</b>.
          </li>
        </ul>
        This is applied <b>before clustering</b>.
      </HelpCard>

      <HelpCard title="Window-relative Z-score max (max_z)" active={m === "max_z" || isComposite}>
        Active if <b>max(z(t)) ≥ {Number(zThresh).toFixed(2)}</b> within the window.
      </HelpCard>

      <HelpCard title="AUC threshold (Area Under Curve)" active={m === "auc" || isComposite}>
        Baseline uses the first <b>{Math.round(Number(baselineFrac) * 100)}%</b> of the window.
        <div className="eq">AUC = ∫ (x(t) − baseline) dt</div>
        Active if <b>AUC ≥ {Number(aucThresh).toFixed(2)}</b>.
      </HelpCard>

      <HelpCard title="Mean over baseline threshold" active={m === "mean_over_baseline" || isComposite}>
        <div className="eq">mean_over_baseline = mean(x(t) − baseline)</div>
        Active if <b>mean_over_baseline ≥ {Number(meanThresh).toFixed(2)}</b>.
      </HelpCard>

      <HelpCard title="Peak prominence threshold" active={m === "prominence" || isComposite}>
        We find peaks after baseline subtraction and measure how much a peak stands out.
        Active if <b>prominence ≥ {Number(promThresh).toFixed(2)}</b> and peaks ≥ <b>{Number(minPeaks)}</b>.
      </HelpCard>

      <HelpCard title="Composite (robust OR)" active={isComposite}>
        Composite marks a neuron active if it passes <b>any</b> criterion (including z_duration).
      </HelpCard>
    </div>
  );
}
