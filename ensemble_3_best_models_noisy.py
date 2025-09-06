# =============================================================================
# MAJORITY VOTING ENSEMBLE EVALUATION
# =============================================================================

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification
)
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

def load_dataset(file_path='unified_noisy_dataset.csv'):
    """Load and prepare the unified noisy dataset"""
    print(f"\nLoading dataset from: {file_path}")

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
    print(f"Label distribution:")
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

    if 'noise_type' in df.columns:
        print(f"\nNoise type distribution:")
        noise_counts = df['noise_type'].value_counts()
        for noise_type, count in noise_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {noise_type}: {count:,} ({percentage:.1f}%)")

    return df

# =============================================================================
# SECTION 2: MODEL PREDICTOR CLASS
# =============================================================================

class ModelPredictor:
    """Class to handle individual model predictions"""

    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = device

    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            print(f"Loading {self.model_name}...")

            if not os.path.exists(self.model_path):
                print(f"Model directory not found: {self.model_path}")
                return False

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()

            print(f"Successfully loaded {self.model_name}")
            return True

        except Exception as e:
            print(f"Error loading {self.model_name}: {str(e)}")
            return False

    def predict_batch(self, texts, batch_size=16):
        """Make predictions on a batch of texts"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        predictions = []
        probabilities = []

        for i in tqdm(range(0, len(texts), batch_size),
                      desc=f"Predicting {self.model_name}"):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                probs = torch.nn.functional.softmax(logits, dim=-1)
                batch_predictions = torch.argmax(logits, dim=-1)

                predictions.extend(batch_predictions.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        return np.array(predictions), np.array(probabilities)

# =============================================================================
# SECTION 3: ENSEMBLE VOTING CLASS
# =============================================================================

class MajorityVotingEnsemble:
    """Majority voting ensemble of multiple models"""

    def __init__(self, model_configs):
        self.model_configs = model_configs
        self.models = []
        self.predictions = {}
        self.probabilities = {}
        self.ensemble_predictions = None
        self.ensemble_probabilities = None

    def load_models(self):
        """Load all models in the ensemble"""
        print(f"\nLoading {len(self.model_configs)} models for ensemble...")

        for config in self.model_configs:
            predictor = ModelPredictor(config['path'], config['name'])
            if predictor.load_model():
                self.models.append(predictor)
            else:
                print(f"Failed to load {config['name']}, skipping...")

        print(f"Successfully loaded {len(self.models)} models")
        return len(self.models) > 0

    def predict_ensemble(self, texts):
        """Make predictions using all models and combine with majority voting"""
        if len(self.models) == 0:
            raise ValueError("No models loaded")

        print(f"\nMaking predictions with {len(self.models)} models...")

        all_predictions = []
        all_probabilities = []

        for model in self.models:
            preds, probs = model.predict_batch(texts)
            self.predictions[model.model_name] = preds
            self.probabilities[model.model_name] = probs
            all_predictions.append(preds)
            all_probabilities.append(probs)

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        ensemble_predictions = []
        ensemble_probabilities = []

        n_samples = all_predictions.shape[1]

        for i in range(n_samples):
            sample_predictions = all_predictions[:, i]
            sample_probabilities = all_probabilities[:, i, :]

            vote_counts = Counter(sample_predictions)
            majority_prediction = vote_counts.most_common(1)[0][0]

            if len(vote_counts) > 1 and vote_counts.most_common(1)[0][1] == vote_counts.most_common(2)[1][1]:
                avg_probs = np.mean(sample_probabilities, axis=0)
                majority_prediction = np.argmax(avg_probs)

            avg_probabilities = np.mean(sample_probabilities, axis=0)

            ensemble_predictions.append(majority_prediction)
            ensemble_probabilities.append(avg_probabilities)

        self.ensemble_predictions = np.array(ensemble_predictions)
        self.ensemble_probabilities = np.array(ensemble_probabilities)

        return self.ensemble_predictions, self.ensemble_probabilities

    def get_voting_statistics(self):
        """Get statistics about the voting process"""
        if len(self.predictions) == 0:
            return None

        model_names = list(self.predictions.keys())
        n_samples = len(self.ensemble_predictions)

        agreement_stats = {
            'unanimous_agreement': 0,
            'majority_agreement': 0,
            'disagreement_stats': {}
        }

        all_preds = np.array([self.predictions[name] for name in model_names])

        for i in range(n_samples):
            sample_preds = all_preds[:, i]
            unique_preds = np.unique(sample_preds)

            if len(unique_preds) == 1:
                agreement_stats['unanimous_agreement'] += 1
            else:
                agreement_stats['majority_agreement'] += 1

        agreement_stats['unanimous_percentage'] = (agreement_stats['unanimous_agreement'] / n_samples) * 100
        agreement_stats['majority_percentage'] = (agreement_stats['majority_agreement'] / n_samples) * 100

        return agreement_stats

# =============================================================================
# SECTION 4: EVALUATION FUNCTIONS
# =============================================================================

def calculate_metrics(true_labels, pred_labels, pred_probs):
    """Calculate evaluation metrics"""
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary'
    )

    if pred_probs.ndim > 1:
        roc_auc = roc_auc_score(true_labels, pred_probs[:, 1])
    else:
        roc_auc = roc_auc_score(true_labels, pred_probs)

    cm = confusion_matrix(true_labels, pred_labels)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }


def evaluate_by_groups(df, predictions, probabilities, group_col):
    """Evaluate performance by groups (source or noise_type)"""
    if group_col not in df.columns:
        print(f"Column {group_col} not found in dataset")
        return {}

    results = {}
    unique_groups = df[group_col].unique()

    print(f"\nEvaluating by {group_col}:")

    for group in unique_groups:
        group_mask = df[group_col] == group
        group_true = df.loc[group_mask, 'label'].values
        group_pred = predictions[group_mask]
        group_probs = probabilities[group_mask]

        if len(group_true) == 0:
            continue

        metrics = calculate_metrics(group_true, group_pred, group_probs)
        metrics['sample_count'] = len(group_true)
        metrics['label_distribution'] = df.loc[group_mask, 'label'].value_counts().to_dict()

        results[group] = metrics

        print(f"  {group}: Samples={len(group_true)}, "
              f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")

    return results


# =============================================================================
# SECTION 5: VISUALIZATION FUNCTIONS
# =============================================================================

def plot_confusion_matrix(cm, title):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    labels = ['Safe (0)', 'Injection (1)']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_group_performance(results_dict, group_name, metric='f1_score'):
    """Plot performance by groups"""
    groups = list(results_dict.keys())
    values = [results_dict[group][metric] for group in groups]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(groups, values, alpha=0.7)
    plt.title(f'{metric.replace("_", " ").title()} by {group_name}')
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_model_agreement(individual_predictions, ensemble_predictions, model_names):
    """Plot agreement between individual models and ensemble"""
    n_models = len(model_names)
    n_samples = len(ensemble_predictions)

    agreements = {}
    for name in model_names:
        agreement = np.mean(individual_predictions[name] == ensemble_predictions)
        agreements[name] = agreement

    plt.figure(figsize=(10, 6))
    bars = plt.bar(agreements.keys(), agreements.values(), alpha=0.7)
    plt.title('Individual Model Agreement with Ensemble')
    plt.ylabel('Agreement Rate')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')

    for bar, value in zip(bars, agreements.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 6: MAIN EXECUTION
# =============================================================================

def main():
    ensemble_models = [
        {'path': 'fine-tuning/combined/combined_model_complete/model', 'name': 'Combined-FineTuned'},
        {'path': 'fine-tuning/safe guard/safeguard_model_complete/model', 'name': 'SafeGuard-FineTuned'},
        {'path': 'fine-tuning/xxz224/xxz224_prompt_injection_model_complete/model', 'name': 'xxz224-FineTuned'},
    ]

    print("Models in ensemble:")
    for model in ensemble_models:
        print(f"  - {model['name']}: {model['path']}")

    df = load_dataset()

    ensemble = MajorityVotingEnsemble(ensemble_models)

    if not ensemble.load_models():
        print("Failed to load any models. Exiting.")
        return

    texts = df['text'].tolist()
    true_labels = df['label'].tolist()

    ensemble_preds, ensemble_probs = ensemble.predict_ensemble(texts)

    voting_stats = ensemble.get_voting_statistics()
    if voting_stats:
        print(f"\nVoting Statistics:")
        print(
            f"  Unanimous agreement: {voting_stats['unanimous_agreement']} samples ({voting_stats['unanimous_percentage']:.1f}%)")
        print(
            f"  Majority decision: {voting_stats['majority_agreement']} samples ({voting_stats['majority_percentage']:.1f}%)")

    overall_metrics = calculate_metrics(true_labels, ensemble_preds, ensemble_probs)

    print(f"\nOverall Performance:")
    print(f"  Accuracy:  {overall_metrics['accuracy']:.4f}")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall:    {overall_metrics['recall']:.4f}")
    print(f"  F1-Score:  {overall_metrics['f1_score']:.4f}")
    print(f"  ROC AUC:   {overall_metrics['roc_auc']:.4f}")

    plot_confusion_matrix(overall_metrics['confusion_matrix'], "Majority Voting Ensemble")

    source_results = evaluate_by_groups(df, ensemble_preds, ensemble_probs, 'source')
    if source_results:
        plot_group_performance(source_results, 'Source Dataset', 'f1_score')
        plot_group_performance(source_results, 'Source Dataset', 'accuracy')

    noise_results = evaluate_by_groups(df, ensemble_preds, ensemble_probs, 'noise_type')
    if noise_results:
        plot_group_performance(noise_results, 'Noise Type', 'f1_score')
        plot_group_performance(noise_results, 'Noise Type', 'accuracy')

    if len(ensemble.predictions) > 1:
        model_names = list(ensemble.predictions.keys())
        plot_model_agreement(ensemble.predictions, ensemble_preds, model_names)

    comparison_results = []

    for model_name in ensemble.predictions:
        individual_preds = ensemble.predictions[model_name]
        individual_probs = ensemble.probabilities[model_name]

        metrics = calculate_metrics(true_labels, individual_preds, individual_probs)
        metrics['model_name'] = model_name
        metrics['model_type'] = 'Individual'
        comparison_results.append(metrics)

        print(f"\n{model_name} (Individual):")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

    ensemble_metrics = overall_metrics.copy()
    ensemble_metrics['model_name'] = 'Majority Voting Ensemble'
    ensemble_metrics['model_type'] = 'Ensemble'
    comparison_results.append(ensemble_metrics)

    print(f"\nMajority Voting Ensemble:")
    print(f"  Accuracy:  {ensemble_metrics['accuracy']:.4f}")
    print(f"  Precision: {ensemble_metrics['precision']:.4f}")
    print(f"  Recall:    {ensemble_metrics['recall']:.4f}")
    print(f"  F1-Score:  {ensemble_metrics['f1_score']:.4f}")
    print(f"  ROC AUC:   {ensemble_metrics['roc_auc']:.4f}")

    comparison_df = pd.DataFrame([
        {
            'Model': result['model_name'],
            'Type': result['model_type'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'ROC AUC': result['roc_auc']
        }
        for result in comparison_results
    ])

    print(f"\nComparison Summary:")
    print(comparison_df.round(4).to_string(index=False))

    best_f1_idx = comparison_df['F1-Score'].idxmax()
    best_model = comparison_df.loc[best_f1_idx, 'Model']
    best_f1 = comparison_df.loc[best_f1_idx, 'F1-Score']

    print(f"\nBest performing approach: {best_model} (F1: {best_f1:.4f})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"noisy_ensemble_evaluation_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    comparison_df.to_csv(f"{results_dir}/ensemble_comparison.csv", index=False)

    predictions_df = pd.DataFrame({
        'text': df['text'],
        'true_label': true_labels,
        'ensemble_prediction': ensemble_preds,
        'ensemble_prob_safe': ensemble_probs[:, 0],
        'ensemble_prob_injection': ensemble_probs[:, 1],
        'source': df['source'] if 'source' in df.columns else 'unknown',
        'noise_type': df['noise_type'] if 'noise_type' in df.columns else 'unknown'
    })

    for model_name in ensemble.predictions:
        predictions_df[f'{model_name}_prediction'] = ensemble.predictions[model_name]
        predictions_df[f'{model_name}_prob_safe'] = ensemble.probabilities[model_name][:, 0]
        predictions_df[f'{model_name}_prob_injection'] = ensemble.probabilities[model_name][:, 1]

    predictions_df.to_csv(f"{results_dir}/detailed_predictions.csv", index=False)

    if source_results:
        source_df = pd.DataFrame([
            {
                'source': source,
                'sample_count': metrics['sample_count'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc']
            }
            for source, metrics in source_results.items()
        ])
        source_df.to_csv(f"{results_dir}/metrics_by_source.csv", index=False)

    if noise_results:
        noise_df = pd.DataFrame([
            {
                'noise_type': noise_type,
                'sample_count': metrics['sample_count'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc']
            }
            for noise_type, metrics in noise_results.items()
        ])
        noise_df.to_csv(f"{results_dir}/metrics_by_noise_type.csv", index=False)

    if voting_stats:
        with open(f"{results_dir}/voting_statistics.json", 'w') as f:
            json.dump(voting_stats, f, indent=2)

    print(f"\nResults saved to: {results_dir}/")
    print("Files created:")
    print("  - ensemble_comparison.csv: Overall comparison of individual models vs ensemble")
    print("  - detailed_predictions.csv: Predictions from all models for each sample")
    print("  - metrics_by_source.csv: Performance metrics by source dataset")
    print("  - metrics_by_noise_type.csv: Performance metrics by noise type")
    print("  - voting_statistics.json: Voting agreement statistics")


if __name__ == "__main__":
    main()
