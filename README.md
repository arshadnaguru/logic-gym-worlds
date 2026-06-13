# Logic Gym Worlds 🧠

A benchmark for testing rule-following in Large Language Models — distinguishing genuine reasoning from pattern matching.

## What This Is

Logic Gym Worlds tests whether LLMs can follow **novel rule systems** they haven't seen before, using:
- **Operator-level OOD testing** (not just new words — new logical operators)
- **Bundle-level consistency checking** (are answers coherent with each other?)
- **Non-classical logic tracks** (defaults, temporal, paraconsistent)

## Project Structure

```
logic-gym-worlds/
├── configs/
│   ├── tracks/          # Per-logic-track configs (horn_fol, nonmonotonic, etc.)
│   └── splits/          # Train/dev/test OOD split definitions
├── src/
│   ├── templates/       # Rule templates and surface forms
│   ├── generators/      # Rule, world, query, instance generators
│   ├── solvers/         # Z3, Clingo, paraconsistent backends
│   ├── evaluation/      # LLM interface, metrics, consistency checks
│   └── utils/           # Helpers
├── data/
│   ├── generated/       # Raw generated instances
│   └── processed/       # Train/dev/test splits
├── experiments/
│   ├── scripts/         # Evaluation runner scripts
│   └── results/         # Model evaluation outputs
├── tests/               # Unit tests
├── notebooks/           # Colab-ready notebooks
└── requirements.txt
```

## Quick Start (Google Colab — Recommended)

```python
# In Colab:
!git clone https://github.com/YOUR_USERNAME/logic-gym-worlds.git
%cd logic-gym-worlds
!pip install -r requirements.txt

# Generate instances
!python src/generators/instance_generator.py --track horn_fol --num 1000

# Run evaluation
!python experiments/scripts/run_evaluation.py \
    --instances data/processed/test_operator_ood.json \
    --model claude-sonnet-4-6 \
    --output experiments/results/claude_operator_ood.json
```

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/logic-gym-worlds.git
cd logic-gym-worlds
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## The Four Tracks

| Track | Logic Type | Solver |
|-------|-----------|--------|
| Horn-FOL | Classical first-order logic | Z3 |
| Non-Monotonic | Defaults & exceptions | Clingo |
| Temporal | Time-step reasoning | Z3 (BMC) |
| Paraconsistent | Contradiction-tolerant | Custom 4-valued |

## The Three OOD Splits

| Split | What Changes |
|-------|-------------|
| Lexical-OOD | New entity/predicate names, same operators |
| Operator-OOD | Entirely new logical operators (count, exactly-N, iff) |
| Compositional-OOD | Novel combinations of seen operators |

## Evaluation Metrics

- **Query-level**: Accuracy, Macro-F1 per label
- **Bundle-level**: Contradiction rate, Joint satisfiability, Paraphrase invariance

## Citation

If you use this benchmark, please cite:
```bibtex
@misc{logicgymworlds2026,
  title={Logic Gym Worlds: A Benchmark for Rule-Following in LLMs},
  author={Your Name},
  year={2026}
}
```
