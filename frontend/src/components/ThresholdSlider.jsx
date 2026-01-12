import React from "react";

export default function ThresholdSlider({
  label = "Threshold",
  value,
  setValue,
  min = 0,
  max = 10,
  step = 0.1,
  disabled = false,
}) {
  return (
    <div>
      <div className="sliderTop">
        <div className="strong">{label}</div>
        <div className="pill">{Number(value).toFixed(2)}</div>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => setValue(Number(e.target.value))}
        disabled={disabled}
      />
    </div>
  );
}
