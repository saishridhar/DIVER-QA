# DIVER-QA Dataset

This repository contains the DIVER-QA (Diverse Question Answering) dataset for evaluating the effectiveness of LLM performance metrics in comparison to human evaluations.

## Dataset Description

DIVER-QA is a dataset for evaluating how well automated metrics align with human judgments of LLM-generated answers. It contains questions, reference answers, model-generated answers, and human binary correctness evaluations.

### Schema

The dataset is provided as a CSV file with the following columns:

- `questions`: The question taken from the parent dataset (string)
- `answers`: The reference/ground truth answers (string)
- `dataset`: The parent dataset of this question (string)
- `prediction`: The answer generated by the candidate model to the question (string)
- `model`: The candidate model that answered the question (string)
- `eval`: The human rating on whether the candidate model's answer was correct or not (1 = correct, 0 = incorrect)

## Models Used:
- Claude-3.5-Sonnet
- Mixtral-8x7B-Instruct
- Llama-3-70B-Instruct
- Llama-3-8B-Instruct
- Phi-3-mini

## Datasets Used:
| Dataset      | No. of questions |
|--------------|------------------|
| AdversarialQA| 120              |
| SQuAD        | 120              |
| MedQA        | 120              |
| HotpotQA     | 120              |
| TriviaQA     | 120              |
| Total        | 600              |


   
## Usage

### Basic Usage

```bash
# Install requirements
pip install -r requirements.txt

# Run evaluation using a custom metric
python evaluate_metrics.py --dataset diver_qa.csv --metric your_metric.py

# Include question as input to the metric
python evaluate_metrics.py --dataset diver_qa.csv --metric your_metric.py --include-question

# Show breakdown by dataset and model
python evaluate_metrics.py --dataset diver_qa.csv --metric your_metric.py --by-dataset --by-model

# Use threshold for continuous metrics
python evaluate_metrics.py --dataset diver_qa.csv --metric your_metric.py --threshold 0.5
```

### Creating Your Own Metric

1. Create a Python file with a function named `metric_function`
2. The function should take two arguments: `reference_answer` and `candidate_answer`
3. If you want to use the question in your metric, add a third argument: `question` (and use the `--include-question` flag)
4. The function should return a score:
   - Binary score (0 or 1) for classification metrics
   - Continuous score (0.0 to 1.0) for regression metrics (use the `--threshold` flag to convert to binary)


## Evaluation Metrics

The script compares metric predictions against human evaluations using:

- **MCC** (Matthews Correlation Coefficient): A balanced measure for binary classification
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Percentage of correct predictions




## License

MIT
