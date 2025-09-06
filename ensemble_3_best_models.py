# =============================================================================
# ENSEMBLE MAJORITY VOTING EVALUATION - TOP 3 MODELS
# =============================================================================
# Testing ensemble of best 3 models with majority voting
# =============================================================================

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
import os
from tqdm import tqdm
import sys
from collections import Counter

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

print(f"Evaluation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# SECTION 1: LOAD DATASET
# =============================================================================

def load_unified_dataset(file_path='unified_dataset_filled.csv'):
    """Load and prepare the unified golden dataset"""
    print("\nLoading Unified Golden Dataset...")

    if not os.path.exists(file_path):
        print(f"Dataset file not found: {file_path}")
        sys.exit(1)

    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")

    original_size = len(df)
    df = df.dropna(subset=['text', 'label'])
    print(f"Removed {original_size - len(df)} rows with missing data")
    print(f"Final dataset shape: {df.shape}")
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(df):,}")

    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "Safe" if label == 0 else "Injection"
        percentage = (count / len(df)) * 100
        print(f"  {label} ({label_name}): {count:,} ({percentage:.1f}%)")

    if 'source' in df.columns:
        print(f"\nSource distribution:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {source}: {count:,} ({percentage:.1f}%)")

    return df

# =============================================================================
# SECTION 2: MODEL LOADER CLASS
# =============================================================================

class EnsembleModel:
    """Class to handle ensemble of multiple models"""

    def __init__(self, model_configs):
        self.model_configs = model_configs
        self.models = {}
        self.tokenizers = {}
        self.device = device

    def load_models(self):
        """Load all models in the ensemble"""

        for model_name, model_path in self.model_configs.items():
            try:
                if not os.path.exists(model_path):
                    print(f"Model directory not found: {model_path}")
                    continue

                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)

                model.to(self.device)
                model.eval()

                self.tokenizers[model_name] = tokenizer
                self.models[model_name] = model

                print(f"Successfully loaded {model_name}")

            except Exception as e:
                print(f"Error loading {model_name}: {str(e)}")

        print(f"\nLoaded {len(self.models)} models successfully")
        return len(self.models) > 0

    def predict_single_model(self, model_name, texts, batch_size=16):
        """Make predictions with a single model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        predictions = []
        probabilities = []

        for i in tqdm(range(0, len(texts), batch_size),
                      desc=f"Predicting {model_name}", leave=False):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                probs = torch.nn.functional.softmax(logits, dim=-1)
                batch_predictions = torch.argmax(logits, dim=-1)

                predictions.extend(batch_predictions.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        return np.array(predictions), np.array(probabilities)

    def predict_ensemble(self, texts, batch_size=16):
        """Make ensemble predictions using majority voting"""
        print(f"\nMaking Ensemble Predictions...")

        all_predictions = {}
        all_probabilities = {}

        for model_name in self.models.keys():
            print(f"Getting predictions from {model_name}...")
            preds, probs = self.predict_single_model(model_name, texts, batch_size)
            all_predictions[model_name] = preds
            all_probabilities[model_name] = probs

        ensemble_predictions = []
        ensemble_confidences = []
        voting_details = []

        for i in range(len(texts)):
            votes = [all_predictions[model][i] for model in self.models.keys()]

            vote_counts = Counter(votes)

            majority_label = vote_counts.most_common(1)[0][0]
            vote_confidence = vote_counts[majority_label] / len(votes)

            ensemble_predictions.append(majority_label)
            ensemble_confidences.append(vote_confidence)

            voting_details.append({
                'votes': {model: all_predictions[model][i] for model in self.models.keys()},
                'majority_label': majority_label,
                'vote_confidence': vote_confidence,
                'unanimous': len(vote_counts) == 1
            })

        return (np.array(ensemble_predictions),
                np.array(ensemble_confidences),
                all_predictions,
                all_probabilities,
                voting_details)

# =============================================================================
# SECTION 3: METRICS CALCULATION
# =============================================================================

def calculate_metrics(true_labels, pred_labels, model_name="Model"):
    """Calculate evaluation metrics"""
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', zero_division=0
    )

    # ROC AUC require probablities, if not possible, skip
    try:
        cm = confusion_matrix(true_labels, pred_labels)
    except:
        cm = np.array([[0, 0], [0, 0]])

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'num_samples': len(true_labels)
    }


def calculate_metrics_per_source(df, predictions, source_column='source'):
    """Calculate metrics for each source dataset"""
    print(f"\nCalculating per-source metrics...")

    source_metrics = {}
    sources = df[source_column].unique()

    for source in sources:
        source_mask = df[source_column] == source
        source_true = df[source_mask]['label'].values
        source_pred = predictions[source_mask]

        metrics = calculate_metrics(source_true, source_pred, f"{source}")
        source_metrics[source] = metrics

        print(f"  {source}: {len(source_true)} samples - "
              f"Acc: {metrics['accuracy']:.3f}, "
              f"F1: {metrics['f1_score']:.3f}")

    return source_metrics

# =============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# =============================================================================

def plot_confusion_matrix(cm, title, ax=None):
    """Plot confusion matrix"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    labels = ['Safe (0)', 'Injection (1)']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')


def plot_metrics_comparison(overall_metrics, source_metrics):
    """Plot metrics comparison"""
    all_metrics = {'Overall': overall_metrics}
    all_metrics.update(source_metrics)

    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    model_names = list(all_metrics.keys())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_names):
        values = [all_metrics[model][metric] for model in model_names]
        colors = ['red' if model == 'Overall' else 'lightblue' for model in model_names]

        bars = axes[i].bar(range(len(model_names)), values, alpha=0.7, color=colors)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylim(0, 1)
        axes[i].set_xticks(range(len(model_names)))
        axes[i].set_xticklabels(model_names, rotation=45, ha='right')

        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_voting_analysis(voting_details):
    """Analyze voting patterns"""
    print(f"\nVoting Analysis:")

    unanimous_count = sum(1 for v in voting_details if v['unanimous'])
    split_count = len(voting_details) - unanimous_count

    print(f"  Unanimous decisions: {unanimous_count:,} ({unanimous_count / len(voting_details) * 100:.1f}%)")
    print(f"  Split decisions: {split_count:,} ({split_count / len(voting_details) * 100:.1f}%)")

    confidences = [v['vote_confidence'] for v in voting_details]

    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=3, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Vote Confidence (Fraction of models agreeing)')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Voting Confidence')
    plt.xticks([1 / 3, 2 / 3, 1.0], ['33% (1/3)', '67% (2/3)', '100% (3/3)'])
    plt.grid(True, alpha=0.3)
    plt.show()

# =============================================================================
# SECTION 5: MAIN EXECUTION
# =============================================================================

def main():
    df = load_unified_dataset()

    # top 3 models based on previous evaluation results
    top_models = {
        'SafeGuard': 'fine-tuning/safe guard/safeguard_model_complete/model',
        'xxz224': 'fine-tuning/xxz224/xxz224_prompt_injection_model_complete/model',
        'Combined': 'fine-tuning/combined/combined_model_complete/model'
    }

    print(f"\nTop 3 Models for Ensemble:")
    for name, path in top_models.items():
        print(f"  - {name}: {path}")

    ensemble = EnsembleModel(top_models)

    if not ensemble.load_models():
        print("Failed to load models. Exiting.")
        return

    texts = df['text'].tolist()
    true_labels = df['label'].values

    (ensemble_preds, ensemble_confidences,
     individual_preds, individual_probs, voting_details) = ensemble.predict_ensemble(texts)

    overall_metrics = calculate_metrics(true_labels, ensemble_preds, "Ensemble")

    print(f"\nOverall Ensemble Performance:")
    print(f"  Accuracy:  {overall_metrics['accuracy']:.4f}")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall:    {overall_metrics['recall']:.4f}")
    print(f"  F1-Score:  {overall_metrics['f1_score']:.4f}")

    source_metrics = calculate_metrics_per_source(df, ensemble_preds)

    # individual model performance for comparison
    print(f"\nIndividual Model Performance (for comparison):")
    individual_metrics = {}
    for model_name, preds in individual_preds.items():
        metrics = calculate_metrics(true_labels, preds, model_name)
        individual_metrics[model_name] = metrics
        print(f"  {model_name}: Acc: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")

    plot_voting_analysis(voting_details)

    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(overall_metrics['confusion_matrix'], 'Ensemble - Overall')
    plt.show()

    plot_metrics_comparison(overall_metrics, source_metrics)

    sources = list(source_metrics.keys())
    n_sources = len(sources)
    cols = min(3, n_sources)
    rows = (n_sources + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_sources == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()

    for i, source in enumerate(sources):
        ax = axes[i] if i < len(axes) else None
        if ax is not None:
            plot_confusion_matrix(source_metrics[source]['confusion_matrix'],
                                  f'Ensemble - {source}', ax)

    for j in range(len(sources), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    print(f"\nSaving Results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"ensemble_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    summary_data = []

    summary_data.append({
        'Dataset': 'Overall',
        'Samples': overall_metrics['num_samples'],
        'Accuracy': overall_metrics['accuracy'],
        'Precision': overall_metrics['precision'],
        'Recall': overall_metrics['recall'],
        'F1_Score': overall_metrics['f1_score']
    })

    for source, metrics in source_metrics.items():
        summary_data.append({
            'Dataset': source,
            'Samples': metrics['num_samples'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{results_dir}/ensemble_summary_metrics.csv", index=False)

    results_df = df.copy()
    results_df['ensemble_prediction'] = ensemble_preds
    results_df['vote_confidence'] = ensemble_confidences

    for model_name, preds in individual_preds.items():
        results_df[f'{model_name}_prediction'] = preds

    results_df.to_csv(f"{results_dir}/ensemble_detailed_predictions.csv", index=False)

    voting_df = pd.DataFrame([
        {
            'text_index': i,
            'majority_label': detail['majority_label'],
            'vote_confidence': detail['vote_confidence'],
            'unanimous': detail['unanimous'],
            **{f'{model}_vote': detail['votes'][model] for model in top_models.keys()}
        }
        for i, detail in enumerate(voting_details)
    ])
    voting_df.to_csv(f"{results_dir}/voting_details.csv", index=False)

    all_metrics = {
        'overall': {
            k: float(v) if isinstance(v, (np.integer, np.floating)) else v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in overall_metrics.items() if k != 'confusion_matrix'},
        'per_source': {source: {
            k: float(v) if isinstance(v, (np.integer, np.floating)) else v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items() if k != 'confusion_matrix'}
                       for source, metrics in source_metrics.items()},
        'individual_models': {model: {
            k: float(v) if isinstance(v, (np.integer, np.floating)) else v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items() if k != 'confusion_matrix'}
                              for model, metrics in individual_metrics.items()},
        'ensemble_config': {
            'models': list(top_models.keys()),
            'voting_method': 'majority',
            'evaluation_timestamp': timestamp,
            'dataset_size': len(df)
        }
    }

    with open(f"{results_dir}/ensemble_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Results saved to {results_dir}/")

    print(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ensemble F1-Score: {overall_metrics['f1_score']:.4f}")

    # compare with best individual model
    best_individual = max(individual_metrics.values(), key=lambda x: x['f1_score'])
    improvement = overall_metrics['f1_score'] - best_individual['f1_score']
    print(f"Best Individual F1: {best_individual['f1_score']:.4f} ({best_individual['model_name']})")
    if improvement > 0:
        print(f"Ensemble Improvement: +{improvement:.4f}")
    else:
        print(f"Ensemble Performance: {improvement:.4f}")

    print(f"\nPer-Source Performance Summary:")
    for source, metrics in source_metrics.items():
        print(
            f"  {source}: F1={metrics['f1_score']:.3f}, Acc={metrics['accuracy']:.3f} ({metrics['num_samples']} samples)")

    print(f"\nVoting Summary:")
    unanimous = sum(1 for v in voting_details if v['unanimous'])
    print(f"  Unanimous: {unanimous}/{len(voting_details)} ({unanimous / len(voting_details) * 100:.1f}%)")

if __name__ == "__main__":
    main()
