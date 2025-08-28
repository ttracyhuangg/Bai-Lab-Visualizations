# Spike Count Heatmap Generator

Python tool for generating heatmaps and clustered visualizations of multi-electrode array (MEA) spike counts — supports **absolute** & **per-device relative** views, with **macro, subsite, and clustered electrode layouts**.

---

## Features
- Cleans and bins spike count data into user-defined time windows (default = 3 minutes).
- Produces:
  1. **Macro A1–D6 (collapsed subsites):** absolute + relative heatmaps  
  2. **Per-letter subsites (A–D):** absolute + relative heatmaps  
  3. **All subsites clustered:** absolute + relative heatmaps
- Outputs high-resolution `.png` heatmaps to `heatmap_outputs/`.

---

## Example Use Case
Visualize spike activity patterns across electrodes to spot:
- High-activity channels (absolute view)
- Within-device changes over time (relative view)
- Groupings of channels with similar firing patterns (clustered view)

---

## Installation

Requires **Python 3.9+** and the following dependencies:
- pandas  
- numpy  
- matplotlib  
- scipy  

Install with:

```bash
pip install -r requirements.txt
