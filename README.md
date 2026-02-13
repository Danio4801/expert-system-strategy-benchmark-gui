# expert-system-strategy-benchmark

GUI application for benchmarking rule-based expert systems with multiple inference strategies with built-in features such as:

- **Smart Imputation** - automatically fills missing values using class-based statistics (mean/mode per decision class) to preserve data distribution
- **Smart Binning Advisor** - expert system module that analyzes data distribution and recommends optimal discretization method and provides clear explanation 
- **Multiple Discretization Methods** - Equal Width, Equal Frequency and more

> **Looking for the library version?**
> 
> Great, this project is  available as a pip library:
> ['placeholder`](blank)
> !Library will be publicly available in the early morning of 17 Feb 2026.
>
> Check it out, it's faster than GUI version due to lack of Flet dependencies and I spend more time building Lib, so it works faster, it is better thought out and was created after this version of the program, so I had way more experience when writing it.

---

## Features

### Inference Engine
- **Forward Chaining** - data driven inference with conflict resolution
- **Backward Chaining** - goal driven hypothesis verification
- **Conflict Resolution Strategies**: Random, First, Specificity and Recency

### Rule Generation
- **Naive** - direct rule extraction from data
- **Decision Tree** - single tree-based rule generation
- **Random Forest(recommended)** 
### Benchmarking
- Compare multiple strategy combinations
- Performance and statistics metrics
- Log and extended log files

### User Interface
- GUI built with [Flet](https://flet.dev/)
- Language: Polish and English
- Built-in datasets from:
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/):
  - [Iris](https://archive.ics.uci.edu/dataset/53/iris)
  - [Wine](https://archive.ics.uci.edu/dataset/109/wine)
  - [Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)
  - [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
  - [Breast Cancer Wisconsin](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
  - [Adult Income](https://archive.ics.uci.edu/dataset/2/adult)
  - [Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation)
  - And more...

---

## Installation

### Option 1(Recommended): Download Latest Pre-built Application (Recommended)

Download the latest release from [Releases](https://github.com/Danio4801/expert-system-strategy-benchmark/releases):

| Platform | Download |
|----------|----------|
| Windows (x64) | `ExpertSystem-vX.X.X-Windows-x64.exe` |
| macOS (Apple Silicon) | `ExpertSystem-vX.X.X-macOS-arm64.zip` |

or
### Option 2: Run from Source

```bash
# Clone the repository
git clone https://github.com/Danio4801/expert-system-strategy-benchmark.git
cd expert-system-strategy-benchmark

# Install dependencies
pip install -r requirements.txt

# Run the application
cd src
python app.py
```

---

## Project Structure

```
├── core/           # Inference engine (Forward/Backward Chaining)
├── preprocessing/  # Rule generation, discretization, imputation
├── src/            # GUI (Flet)
├── data/           # Built-in datasets
├── tests/          # Unit tests
└── docs/   # Documentation (!Temporarily avaialbe only in Polish)
```

---

## Requirements

- Python 3.10+
- Dependencies: `requirements.txt`





## License

MIT License

---
