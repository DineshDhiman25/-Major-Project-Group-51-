# Major Grp51
###### Active Repository for Major Project of Group 51 (2026–2027)

![Title Logo](./src/static/imgs/logo.png)

## Ethical Analysis of Human Rights Violations on Social Media

This project develops a structured framework for **identifying, annotating, and analyzing human rights violations** in social media discussions. Since official reports are often delayed, censored, or inaccessible, digital accounts can serve as early indicators—but they are often fragmentary, culturally constrained, and prone to misinformation.

Our framework emphasizes **ethical handling, participant protection, and contextual accuracy**, focusing on:

- Clear guidelines for annotation and justification
- Minimizing risks of personal identification and misinformation spread
- Supporting spatiotemporal and cross-context analysis
- Enabling use in education, advocacy, and policy interventions

The goal is to transform raw social media discourse into a **credible, respectful, and accountable resource** for researchers, communities, and policymakers.

---

## Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/Fairtexas5/Major_Grp51.git
cd Major_Grp51

```

### 2. Initiating virtual Environment

Create a virtual environment for ease of runnning scripts, use python 3.11 as base for environment.

```bash
# Mac/Linux
python3.11 -m venv .venv

# Windows
python -m venv .venv

```

then activate the environment

```bash
# Mac/Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

```

Alternatively conda can be used

```bash
conda create -n major51 python=3.11
conda activate major51
```

### 3. Install Packages

Install packages using below command:

```bash
pip install -e .
```
