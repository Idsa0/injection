# =============================================================================
# SECTION 1: SETUP AND IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
import os
from tqdm import tqdm
import sys

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

print(f"Evaluation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# SECTION 2: LOAD AND PREPARE UNIFIED DATASET
# =============================================================================

def load_unified_dataset(file_path='unified_dataset_filled.csv'):
    """Load and prepare the unified golden dataset"""
    print("\nLoading Unified Golden Dataset...")

    if not os.path.exists(file_path):
        print(f"Dataset file not found: {file_path}")
        print("Please make sure unified_dataset.csv is in the current directory")
        sys.exit(1)

    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")

    original_size = len(df)
    df = df.dropna(subset=['text', 'label'])
    print(f"Removed {original_size - len(df)} rows with missing labels/text")
    print(f"Final dataset shape: {df.shape}")

    print("\nDataset Statistics:")
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

    return df


unified_df = load_unified_dataset()

# =============================================================================
# SECTION 3: MODEL LOADING AND PREDICTION FUNCTIONS
# =============================================================================

class ModelEvaluator:
    """Class to handle model loading and evaluation"""

    def __init__(self, model_path, model_name, is_zero_shot=False):
        self.model_path = model_path
        self.model_name = model_name
        self.is_zero_shot = is_zero_shot
        self.model = None
        self.tokenizer = None
        self.classifier_pipeline = None
        self.results = {}
        self.device = device

    def load_model(self):
        """Load the fine-tuned model and tokenizer or setup zero-shot"""
        try:
            print(f"\nLoading {self.model_name}...")

            if self.is_zero_shot:
                print("Setting up zero-shot classification...")
                # use pre-trained DistilBERT for zero-shot
                model_name = "distilbert-base-uncased"
                self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    model_name, num_labels=2
                )
                # initialize with random classification head for zero-shot
                torch.nn.init.xavier_uniform_(self.model.classifier.weight)
                torch.nn.init.zeros_(self.model.classifier.bias)

            else:
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

        if 'TEST_MODE' in globals() and TEST_MODE:
            batch_size = min(batch_size, 8)

        predictions = []
        probabilities = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Predicting {self.model_name}"):
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

    def evaluate(self, df):
        """Evaluate the model on the given dataset"""
        print(f"\nEvaluating {self.model_name}...")

        texts = df['text'].tolist()
        true_labels = df['label'].tolist()

        pred_labels, pred_probs = self.predict_batch(texts)

        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary'
        )

        roc_auc = roc_auc_score(true_labels, pred_probs[:, 1])
        cm = confusion_matrix(true_labels, pred_labels)

        self.results = {
            'model_name': self.model_name,
            'model_type': 'zero-shot' if self.is_zero_shot else 'fine-tuned',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': pred_labels,
            'probabilities': pred_probs,
            'true_labels': true_labels
        }

        print(f"Results for {self.model_name}:")
        print(f"  Type:      {'Zero-shot' if self.is_zero_shot else 'Fine-tuned'}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")

        return self.results

# =============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# =============================================================================

def plot_confusion_matrix(cm, model_name, ax=None):
    """Plot confusion matrix"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    labels = ['Safe (0)', 'Injection (1)']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_metrics_comparison(results_list):
    """Plot comparison of metrics across models"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    model_names = [r['model_name'] for r in results_list]
    model_types = [r['model_type'] for r in results_list]

    colors = ['lightblue' if t == 'zero-shot' else 'lightcoral' for t in model_types]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results_list]
        bars = axes[i].bar(range(len(model_names)), values, alpha=0.7, color=colors)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylim(0, 1)
        axes[i].set_xticks(range(len(model_names)))
        axes[i].set_xticklabels(model_names, rotation=45, ha='right')

        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightblue', label='Zero-shot'),
                       Patch(facecolor='lightcoral', label='Fine-tuned')]
    axes[-1].legend(handles=legend_elements, loc='center')
    axes[-1].axis('off')

    plt.tight_layout()
    plt.show()


def plot_roc_curves(results_list):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(12, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for i, result in enumerate(results_list):
        true_labels = result['true_labels']
        pred_probs = result['probabilities'][:, 1]

        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        auc_score = result['roc_auc']

        model_type = result['model_type']
        linestyle = '--' if model_type == 'zero-shot' else '-'

        plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {auc_score:.3f})",
                 color=colors[i % len(colors)], linestyle=linestyle, linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================================================================
# SECTION 5: DEFINE MODELS TO EVALUATE
# =============================================================================

models_to_evaluate = [
    # fine-tuned models
    {'path': 'fine-tuning/combined/combined_model_complete/model', 'name': 'Combined-FineTuned', 'zero_shot': False},
    {'path': 'fine-tuning/safe guard/safeguard_model_complete/model', 'name': 'SafeGuard-FineTuned',
     'zero_shot': False},
    {'path': 'fine-tuning/jailbreak/jailbreak_model_complete/model', 'name': 'Jailbreak-FineTuned', 'zero_shot': False},
    {'path': 'fine-tuning/xxz224/xxz224_prompt_injection_model_complete/model', 'name': 'xxz224-FineTuned',
     'zero_shot': False},

    # zero-shot model
    {'path': None, 'name': 'DistilBERT-ZeroShot', 'zero_shot': True},
]

for model in models_to_evaluate:
    model_type = "Zero-shot" if model['zero_shot'] else "Fine-tuned"
    path_str = model['path'] if model['path'] else "Pre-trained DistilBERT"
    print(f"  - {model['name']} ({model_type}): {path_str}")

# =============================================================================
# SECTION 6: EVALUATE ALL MODELS
# =============================================================================

all_results = []
successful_evaluations = []

for model_config in models_to_evaluate:
    print(f"EVALUATING: {model_config['name']}")

    evaluator = ModelEvaluator(
        model_config['path'],
        model_config['name'],
        is_zero_shot=model_config['zero_shot']
    )

    if evaluator.load_model():
        try:
            results = evaluator.evaluate(unified_df)
            all_results.append(results)
            successful_evaluations.append(model_config['name'])

            plt.figure(figsize=(8, 6))
            plot_confusion_matrix(results['confusion_matrix'], results['model_name'])
            plt.show()

        except Exception as e:
            print(f"Error during evaluation of {model_config['name']}: {str(e)}")
            import traceback

            traceback.print_exc()
    else:
        print(f"Skipping {model_config['name']} due to loading issues")

print(f"\nSuccessfully evaluated {len(successful_evaluations)} models")
if successful_evaluations:
    print("Successfully evaluated models:", ', '.join(successful_evaluations))

# =============================================================================
# SECTION 7: COMPARATIVE ANALYSIS AND VISUALIZATIONS
# =============================================================================

if len(all_results) > 0:
    summary_df = pd.DataFrame({
        'Model': [r['model_name'] for r in all_results],
        'Type': [r['model_type'] for r in all_results],
        'Accuracy': [r['accuracy'] for r in all_results],
        'Precision': [r['precision'] for r in all_results],
        'Recall': [r['recall'] for r in all_results],
        'F1-Score': [r['f1_score'] for r in all_results],
        'ROC AUC': [r['roc_auc'] for r in all_results]
    })

    print("\nSUMMARY TABLE:")
    print(summary_df.round(4).to_string(index=False))

    # separate fine-tuned and zero-shot results
    finetuned_results = [r for r in all_results if r['model_type'] == 'fine-tuned']
    zeroshot_results = [r for r in all_results if r['model_type'] == 'zero-shot']

    if finetuned_results:
        print(f"\nFINE-TUNED MODELS:")
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']:
            metric_key = metric.lower().replace('-', '_').replace(' ', '_')
            best_idx = max(range(len(finetuned_results)), key=lambda i: finetuned_results[i][metric_key])
            best_model = finetuned_results[best_idx]['model_name']
            best_score = finetuned_results[best_idx][metric_key]
            print(f"  Best {metric}: {best_model} ({best_score:.4f})")

    if zeroshot_results:
        print(f"\nZERO-SHOT MODEL:")
        zs_result = zeroshot_results[0]
        print(f"  {zs_result['model_name']}:")
        print(f"    Accuracy: {zs_result['accuracy']:.4f}")
        print(f"    F1-Score: {zs_result['f1_score']:.4f}")

    if finetuned_results and zeroshot_results:
        best_ft = max(finetuned_results, key=lambda x: x['f1_score'])
        zs = zeroshot_results[0]
        improvement = best_ft['f1_score'] - zs['f1_score']
        print(f"\nFINE-TUNING IMPROVEMENT:")
        print(f"  Best fine-tuned F1: {best_ft['f1_score']:.4f} ({best_ft['model_name']})")
        print(f"  Zero-shot F1: {zs['f1_score']:.4f}")
        print(f"  Improvement: +{improvement:.4f} ({improvement / zs['f1_score'] * 100:.1f}%)")

    print(f"\nGenerating comparison visualizations...")

    plot_metrics_comparison(all_results)
    plot_roc_curves(all_results)

    if len(all_results) > 1:
        n_models = len(all_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()

        for i, result in enumerate(all_results):
            ax = axes[i]
            plot_confusion_matrix(result['confusion_matrix'], result['model_name'], ax)

        for j in range(len(all_results), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

else:
    print("No models were successfully evaluated. Please check model paths and files.")

# =============================================================================
# SECTION 8: ERROR ANALYSIS
# =============================================================================

if len(all_results) > 0:
    for result in all_results:
        print(f"\n--- Error Analysis for {result['model_name']} ({result['model_type']}) ---")

        true_labels = np.array(result['true_labels'])
        pred_labels = result['predictions']
        probabilities = result['probabilities']

        misclassified = true_labels != pred_labels

        if np.sum(misclassified) > 0:
            print(f"Total misclassified: {np.sum(misclassified)} out of {len(true_labels)}")

            false_positives = (true_labels == 0) & (pred_labels == 1)
            fp_count = np.sum(false_positives)
            print(f"False Positives (Safe → Injection): {fp_count}")

            false_negatives = (true_labels == 1) & (pred_labels == 0)
            fn_count = np.sum(false_negatives)
            print(f"False Negatives (Injection → Safe): {fn_count}")

            # show some examples of misclassified samples
            if fp_count > 0:
                print(f"\nExample False Positives (showing up to 2):")
                fp_indices = np.where(false_positives)[0][:2]
                for i, idx in enumerate(fp_indices):
                    text = unified_df.iloc[idx]['text'][:150] + "..."
                    confidence = probabilities[idx][1]
                    print(f"  {i + 1}. Text: {text}")
                    print(f"     Confidence: {confidence:.3f}")

            if fn_count > 0:
                print(f"\nExample False Negatives (showing up to 2):")
                fn_indices = np.where(false_negatives)[0][:2]
                for i, idx in enumerate(fn_indices):
                    text = unified_df.iloc[idx]['text'][:150] + "..."
                    confidence = probabilities[idx][0]
                    print(f"  {i + 1}. Text: {text}")
                    print(f"     Confidence: {confidence:.3f}")

# =============================================================================
# SECTION 9: SAVE RESULTS
# =============================================================================

if len(all_results) > 0:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"evaluation_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    summary_df.to_csv(f"{results_dir}/summary_metrics.csv", index=False)
    print(f"Saved summary metrics to {results_dir}/summary_metrics.csv")

    # save detailed results for each model
    for i, result in enumerate(all_results):
        model_name = result['model_name'].replace('/', '_').replace(' ', '_').replace('-', '_')

        predictions_df = pd.DataFrame({
            'text': unified_df['text'],
            'true_label': result['true_labels'],
            'predicted_label': result['predictions'],
            'prob_safe': result['probabilities'][:, 0],
            'prob_injection': result['probabilities'][:, 1],
            'source': unified_df['source'] if 'source' in unified_df.columns else 'unknown'
        })
        predictions_df.to_csv(f"{results_dir}/{model_name}_predictions.csv", index=False)

        metrics_dict = {
            'model_name': result['model_name'],
            'model_type': result['model_type'],
            'evaluation_timestamp': timestamp,
            'dataset_size': len(unified_df),
            'metrics': {
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1_score': float(result['f1_score']),
                'roc_auc': float(result['roc_auc'])
            },
            'confusion_matrix': result['confusion_matrix'].tolist()
        }

        with open(f"{results_dir}/{model_name}_metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2)

    print(f"Saved detailed results to {results_dir}/")
    print(f"Results directory: {results_dir}")

print(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if len(all_results) > 0:
    # best model
    best_f1_idx = np.argmax([r['f1_score'] for r in all_results])
    best_model = all_results[best_f1_idx]['model_name']
    best_f1 = all_results[best_f1_idx]['f1_score']
    best_type = all_results[best_f1_idx]['model_type']
    print(f"Overall best model (by F1-Score): {best_model} ({best_type}) - F1: {best_f1:.4f}")

    finetuned_results = [r for r in all_results if r['model_type'] == 'fine-tuned']
    zeroshot_results = [r for r in all_results if r['model_type'] == 'zero-shot']

    if finetuned_results and zeroshot_results:
        best_ft = max(finetuned_results, key=lambda x: x['f1_score'])
        zs = zeroshot_results[0]
        print(f"Fine-tuning vs Zero-shot:")
        print(f"   Best Fine-tuned: {best_ft['model_name']} - F1: {best_ft['f1_score']:.4f}")
        print(f"   Zero-shot: {zs['model_name']} - F1: {zs['f1_score']:.4f}")
        print(f"   Improvement: +{best_ft['f1_score'] - zs['f1_score']:.4f}")

else:
    print("No models were successfully evaluated.")
