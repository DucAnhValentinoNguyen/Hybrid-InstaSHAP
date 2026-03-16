![Python](https://img.shields.io/badge/python-3.10-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
# InstaSHAP Benchmark: Testing new proposed method of calculating SHAP values called instaSHAP

This repository contains the implementation, experiments, and analysis for the **instaSHAP** seminar project at **LMU Munich**, supervised by **Dr. Giuseppe Casalicchio**.

The goal of this project is to implement the instaSHAP method for CV and NLP tasks and benchmark them.
---

## 📂 Repository Structure

```
Hybrid-InstaSHAP/
│
├── final_paper_results/ # Experiment results 
├── experiments/         # Experiment scripts and configs
└── requirements.txt     # Dependencies
```

---

## ⚙️ Installation

```bash
git clone https://github.com/DucAnhValentinoNguyen/InstaSHAP-benchmarks

```

---

## 🧪 Running Experiments

```bash
python experiments/run_experiments.py
```

Config-based run:

```bash
python experiments/run_experiments.py --config experiments/configs/hybrid.yaml
```

Interactive exploration:

```
notebooks/
```

---

## 📊 Results Summary



---



## 📄 License

MIT License

Copyright (c) 2025 Duc-Anh Nguyen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 🔮 Future Work

- Extend to text + tabular multimodal models  
- GPU-optimised kernel thinning for large-scale compression  
