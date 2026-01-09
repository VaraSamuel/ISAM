import React from "react";

export default function ThresholdSlider({ value, setValue, min = 0, max = 10, step = 0.1 }) {
  return (
    <div>
      <div className="sliderTop">
        <div className="section-title" style={{ marginBottom: 0 }}>
          Activity Threshold (Z-score)
        </div>
        <div className="pill">{Number(value).toFixed(1)}</div>
      </div>

      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => setValue(parseFloat(e.target.value))}
      />

      <div className="muted" style={{ marginTop: 8, fontSize: 13 }}>
        Higher threshold → stricter criterion → fewer neurons marked active.
      </div>
    </div>
  );
}
