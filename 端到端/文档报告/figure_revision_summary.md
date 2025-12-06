# Summary of Required Modifications for the End-to-End Energy Comparison Figure

This document summarizes the key issues and recommended improvements for refining the end-to-end system energy comparison boxplot figure to meet SCI/MDPI publication standards.

---

## 1. Boxplot Shape and Style
- The current “butterfly-shaped” boxplot is **non-standard** and may confuse reviewers.
- Use a **standard rectangular boxplot** with:
  - A clear median line.
  - Whiskers defined using 1.5×IQR.
  - Optional mean marker (diamond) — acceptable.

---

## 2. Energy Reduction Annotation
- The “Energy Reduction: 9.19% (4348.9 J)” banner is **too large and visually distracting**.
- Replace with:
  - A smaller annotation box, or  
  - Move this information to the **caption** or **title**.
- Remove bright borders or shadow effects.

---

## 3. Mean Value Labels
- Labels such as `42962.1` and `47311.0` overlap with the box body.
- Adjust by:
  - Placing labels **above** the box by a few pixels.
  - Optionally formatting values as **43.0 kJ** for readability.

---

## 4. Color Palette
- Current saturated green/red colors look **cartoonish**.
- Recommended:
  - Soft, muted tones.
  - Semi-transparent fills (alpha 0.4–0.6).
  - Colors consistent with the rest of the paper’s figures.

---

## 5. Y-Axis Scaling and Units
- Use units in **kJ** (e.g., 42 kJ, 44 kJ…) instead of large raw numbers.
- This improves readability and reduces axis clutter.

---

## 6. X-Axis Labels
- Improve naming consistency:
  - “Proposed” vs. “Baseline”
  - or “Proposed Framework” vs. “Baseline Method”

---

## 7. Statistical Validity Notes
- Boxplots imply multiple data samples.
- State clearly (in caption):
  - Number of runs (e.g., *20 Monte Carlo simulations*).
  - Whether the distribution reflects randomness in deployment positions or channel variations.

---

## 8. General Visual Professionalism
- Reduce linewidths where necessary.
- Remove unnecessary embellishments.
- Ensure font size matches other figures in the manuscript.
- Ensure subplot spacing is consistent.

---

## 9. Recommended Caption Structure
```
Figure X. End-to-end system energy consumption under the proposed two-stage framework and the baseline strategy. 
Each box summarizes N randomized simulation runs, with diamond markers indicating the mean. 
The proposed method achieves a 9.2% reduction in total mission energy compared with the baseline.
```

---

## 10. Overall Goal
The revised figure should emphasize:
- Scientific clarity  
- Minimalist professional aesthetics  
- Standardized style consistent with other figures  
- Accurate and easily interpretable statistical information  

This ensures that reviewers perceive the figure as **trustworthy, rigorous, and publication-ready**.
