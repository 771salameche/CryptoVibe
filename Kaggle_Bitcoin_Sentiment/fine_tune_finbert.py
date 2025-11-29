import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================
# PART 2: TOKENIZATION
# ============================================

def tokenize_data(train_df, val_df, test_df, model_name):
    """
    Tokenize les données pour le modèle
    
    Args:
        model_name: "ProsusAI/finbert"
    """
    logging.info(f"Tokenizing data for {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )
    
    # Convertir en Dataset Hugging Face
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
    
    # Tokenizer
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Format pour PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return train_dataset, val_dataset, test_dataset, tokenizer

# ============================================
# PART 3: FINE-TUNING
# ============================================

def fine_tune_model(train_dataset, val_dataset, model_name, output_dir):
    """
    Fine-tune un modèle
    
    Args:
        model_name: nom du modèle Hugging Face
        output_dir: dossier pour sauvegarder le modèle
    """
    logging.info(f"{'='*60}")
    logging.info(f"FINE-TUNING: {model_name}")
    logging.info(f"{'='*60}")
    
    # Charger modèle pré-entraîné
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # positive, negative
        ignore_mismatched_sizes=True
    )
    
    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,  # Réduire à 8 si mémoire insuffisante
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,  # Garder seulement 2 checkpoints
        report_to="none"  # Désactiver wandb
    )
    
    # Métrique d'évaluation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy
        }
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Entraînement
    logging.info("Starting training...")
    trainer.train()
    
    # Sauvegarder
    logging.info(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    
    return trainer

# ============================================
# PART 4: ÉVALUATION
# ============================================

def evaluate_model(trainer, test_dataset, model_name):
    """Évalue le modèle sur test set"""
    
    logging.info(f"{'='*60}")
    logging.info(f"EVALUATION: {model_name}")
    logging.info(f"{'='*60}")
    
    # Prédictions
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    
    # Metrics
    accuracy = accuracy_score(labels, preds)
    logging.info(f"Accuracy: {accuracy:.4f}")
    
    logging.info("Classification Report:")
    logging.info(classification_report(labels, preds, target_names=['Negative', 'Positive']))
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{model_name.replace("/", "_")}_confusion_matrix.png')
    
    return {
        'accuracy': accuracy,
        'predictions': preds,
        'labels': labels
    }

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    
    # Charger données
    train_df = pd.read_csv("train_data.csv")
    val_df = pd.read_csv("validation_data.csv")
    test_df = pd.read_csv("test_data.csv")
    
    # Fine-tune FinBERT
    logging.info("="*60)
    logging.info("PHASE 1: Fine-tuning FinBERT")
    logging.info("="*60)
    
    train_dataset_fb, val_dataset_fb, test_dataset_fb, tokenizer_fb = tokenize_data(
        train_df, val_df, test_df,
        model_name="ProsusAI/finbert"
    )
    
    trainer_fb = fine_tune_model(
        train_dataset_fb,
        val_dataset_fb,
        model_name="ProsusAI/finbert",
        output_dir="./finbert_finetuned"
    )
    
    results_fb = evaluate_model(trainer_fb, test_dataset_fb, "FinBERT (fine-tuned)")
    
    logging.info("FinBERT fine-tuning complete!")
    logging.info("Generated files:")
    logging.info("  - ./finbert_finetuned/ (fine-tuned FinBERT)")
    logging.info("  - FinBERT (fine-tuned)_confusion_matrix.png")
