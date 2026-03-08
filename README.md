# Major Grp51
###### Active Repository for Major Project of Group 51 (2025–2026)

![Logo](src/static/imgs/logo.png)

## Ethical Analysis of Human Rights Violations on Social Media

This project develops a structured framework for **identifying, annotating, and analyzing human rights violations** in social media discussions. Since official reports are often delayed, censored, or inaccessible, digital accounts can serve as early indicators—but they are often fragmentary, culturally constrained, and prone to misinformation.

Our framework emphasizes **ethical handling, participant protection, and contextual accuracy**, focusing on:

- Clear guidelines for annotation and justification
- Minimizing risks of personal identification and misinformation spread
- Supporting spatiotemporal and cross-context analysis
- Enabling use in education, advocacy, and policy interventions

The goal is to transform raw social media discourse into a **credible, respectful, and accountable resource** for researchers, communities, and policymakers.

---

## Project Structure

### Core Components
Contains the primary development work including model fine-tuning and data processing:

| Component | Description |
|-----------|-------------|
| **Fine-tuning Scripts** | `finetuning_mistral_instruct.py`, `finetuning_llama31_instruct.py`, `finetuning_openhermes_mistral.py` - Scripts for fine-tuning various LLMs on the HRV classification task |
| **Training Pipeline** | `train.py` - Core training pipeline with enhanced WandB logging and gradient monitoring |
| **Annotation Tool** | `annotate.ipynb` - Jupyter notebook for data annotation workflows |
| **Annotation/** | Directory containing annotated data samples |
| **dataset/** | Contains Telegram posts and data processing utilities including `id_management.ipynb` |

##### Key Data Files
- `hrv_train.jsonl`, `hrv_val.jsonl`, `hrv_test.jsonl` - Training, validation, and test splits
- `hrv_yes.csv`, `hrv_no.csv` - Labeled human rights violation data


---

## Models Used

The project fine-tunes the following base models for human rights violation detection:

- **Mistral-7B-Instruct-v0.3** - Primary model for instruction-following classification
- **Llama-3.1** - Alternative model for comparative analysis
- **OpenHermes-2.5-Mistral-7B** - Fine-tuned variant for enhanced reasoning

### Training Features
- **LoRA (Low-Rank Adaptation)** - Efficient fine-tuning with reduced memory footprint
- **Focal Loss** - Handles class imbalance in human rights violation detection
- **Assistant-only Masking** - Focuses training on model responses
- **WandB Integration** - Comprehensive experiment tracking

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Deep Learning** | PyTorch, TensorFlow, Transformers, PEFT |
| **Data Processing** | Pandas, NumPy, scikit-learn |
| **Experiment Tracking** | Weights & Biases (WandB) |
| **Visualization** | Matplotlib, Seaborn |
| **UI/Demo** | Gradio |
| **Environment** | Python 3.11, Poetry |

---

## Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/Fairtexas5/Major_Grp51.git
cd Major_Grp51
```

### 2. Set Up Virtual Environment

Create a virtual environment for ease of running scripts, use Python 3.11 as base for the environment.

```bash
# Mac/Linux
python3.11 -m venv .venv

# Windows
python -m venv .venv
```

Then activate the environment:

```bash
# Mac/Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install packages using:

```bash
pip install -e .
```

---

## Usage

### Running Fine-tuning

The fine-tuning scripts are designed to run on Kaggle notebooks with GPU support:

```python
# For Mistral model
python Aditya/finetuning_mistral_instruct.py

# For OpenHermes model
python Aditya/finetuning_openhermes_mistral.py

# For Llama 3.1 model
python Aditya/finetuning_llama31_instruct.py
```

### Data Annotation

Use the annotation notebook for labeling new data:
```bash
jupyter notebook Aditya/annotate.ipynb
```

---

## Commit Instructions

### 1. Adding and Committing

```bash
git add <Your Folder Name>
git commit -m "Your message"
```

### 2. Important! Git Pull
Do this always so that your changes are not deleted:

```bash
git pull
```

### 3. Then Git Push

```bash
git push
```

---

## License

This project is licensed under the [MPL-2.0 License](LICENSE).
