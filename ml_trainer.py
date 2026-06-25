"""
AI Vision System - Face Recognition ML Training
Trains face recognition models using the LFW (Labeled Faces in the Wild) dataset.
Uses PCA + SVM, KNN, Random Forest, and Neural Network classifiers.
"""

import numpy as np
import os
import pickle
import time
from datetime import datetime

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
import joblib


TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_output")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")


class FaceMLTrainer:
    def __init__(self, min_faces_per_person=70, resize=0.5):
        self.min_faces_per_person = min_faces_per_person
        self.resize = resize
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_names = None
        self.pca = None
        self.scaler = None
        self.label_encoder = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

        os.makedirs(TRAINING_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

    def load_dataset(self):
        print("=" * 70)
        print("  LOADING LFW DATASET")
        print("=" * 70)
        print(f"  Min faces per person: {self.min_faces_per_person}")
        print(f"  Resize factor: {self.resize}")

        self.dataset = fetch_lfw_people(
            min_faces_per_person=self.min_faces_per_person,
            resize=self.resize,
            download_if_missing=True
        )

        X = self.dataset.data
        y = self.dataset.target
        self.target_names = self.dataset.target_names
        n_classes = len(self.target_names)

        print(f"\n  Dataset Statistics:")
        print(f"    Total images:    {X.shape[0]}")
        print(f"    Image size:      {self.dataset.images.shape[1]}x{self.dataset.images.shape[2]}")
        print(f"    Features:        {X.shape[1]}")
        print(f"    People:          {n_classes}")
        print(f"    Classes:         {list(self.target_names)}")

        print(f"\n  Images per person:")
        unique, counts = np.unique(y, return_counts=True)
        for idx, count in zip(unique, counts):
            print(f"    {self.target_names[idx]:30s}: {count} images")

        return X, y

    def preprocess(self, X, y):
        print("\n" + "=" * 70)
        print("  PREPROCESSING")
        print("=" * 70)

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
        )

        print(f"  Train set: {self.X_train.shape[0]} images")
        print(f"  Test set:  {self.X_test.shape[0]} images")

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"  Data normalized (StandardScaler)")

        n_components = min(150, self.X_train.shape[0], self.X_train.shape[1])
        self.pca = PCA(n_components=n_components, whiten=True, random_state=42)
        self.X_train = self.pca.fit_transform(self.X_train)
        self.X_test = self.pca.transform(self.X_test)

        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"  PCA applied: {n_components} components ({explained_var:.1%} variance retained)")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(self):
        print("\n" + "=" * 70)
        print("  TRAINING MODELS")
        print("=" * 70)

        models_to_train = {
            "SVM (RBF Kernel)": SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True),
            "SVM (Linear Kernel)": SVC(kernel='linear', C=1, random_state=42, probability=True),
            "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'),
            "KNN (k=3)": KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean'),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
            "Neural Network (MLP)": MLPClassifier(
                hidden_layer_sizes=(256, 128, 64), activation='relu',
                solver='adam', max_iter=500, random_state=42,
                early_stopping=True, validation_fraction=0.1
            ),
        }

        for name, model in models_to_train.items():
            print(f"\n  Training {name}...")
            start_time = time.time()

            model.fit(self.X_train, self.y_train)
            train_time = time.time() - start_time

            y_pred = model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)

            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')

            self.models[name] = model
            self.results[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "train_time": train_time,
            }

            print(f"    Accuracy:       {accuracy:.4f}")
            print(f"    Precision:      {precision:.4f}")
            print(f"    Recall:         {recall:.4f}")
            print(f"    F1-Score:       {f1:.4f}")
            print(f"    CV Accuracy:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"    Training time:  {train_time:.2f}s")

        return self.results

    def hyperparameter_tuning(self):
        print("\n" + "=" * 70)
        print("  HYPERPARAMETER TUNING (SVM GridSearch)")
        print("=" * 70)

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }

        svm = SVC(random_state=42, probability=True)
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=0, refit=True
        )

        print("  Running GridSearchCV...")
        start_time = time.time()
        grid_search.fit(self.X_train, self.y_train)
        tuning_time = time.time() - start_time

        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV score:   {grid_search.best_score_:.4f}")
        print(f"  Tuning time:     {tuning_time:.2f}s")

        y_pred = grid_search.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"  Test accuracy:   {accuracy:.4f}")

        self.models["SVM (Tuned)"] = grid_search.best_estimator_
        self.results["SVM (Tuned)"] = {
            "accuracy": accuracy,
            "precision": precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
            "cv_mean": grid_search.best_score_,
            "cv_std": 0,
            "train_time": tuning_time,
            "best_params": grid_search.best_params_,
        }

        return grid_search.best_estimator_

    def select_best_model(self):
        print("\n" + "=" * 70)
        print("  MODEL COMPARISON")
        print("=" * 70)

        print(f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'CV Mean':>10}")
        print("  " + "-" * 75)

        best_score = 0
        for name, res in self.results.items():
            print(f"  {name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} {res['recall']:>10.4f} {res['f1']:>10.4f} {res['cv_mean']:>10.4f}")
            if res['accuracy'] > best_score:
                best_score = res['accuracy']
                self.best_model = self.models[name]
                self.best_model_name = name

        print("  " + "-" * 75)
        print(f"\n  BEST MODEL: {self.best_model_name} (Accuracy: {best_score:.4f})")

    def detailed_evaluation(self):
        print("\n" + "=" * 70)
        print(f"  DETAILED EVALUATION - {self.best_model_name}")
        print("=" * 70)

        y_pred = self.best_model.predict(self.X_test)

        print("\n  Classification Report:")
        print("  " + "-" * 50)
        report = classification_report(
            self.y_test, y_pred,
            target_names=self.label_encoder.classes_.astype(str),
            zero_division=0
        )
        for line in report.split('\n'):
            print(f"  {line}")

        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\n  Confusion Matrix Shape: {cm.shape}")
        print(f"  Correct predictions:   {np.trace(cm)}/{len(self.y_test)}")
        print(f"  Error rate:            {1 - np.trace(cm)/len(self.y_test):.4f}")

    def save_model(self):
        print("\n" + "=" * 70)
        print("  SAVING TRAINED MODEL")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_data = {
            "model": self.best_model,
            "pca": self.pca,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "target_names": self.target_names,
            "best_model_name": self.best_model_name,
            "results": self.results,
            "training_date": timestamp,
            "min_faces_per_person": self.min_faces_per_person,
            "n_components": self.pca.n_components_,
            "image_shape": self.dataset.images.shape[1:],
        }

        main_path = os.path.join(MODELS_DIR, "face_recognition_model.pkl")
        with open(main_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"  Saved main model: {main_path}")

        timestamped_path = os.path.join(MODELS_DIR, f"face_model_{timestamp}.pkl")
        with open(timestamped_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"  Saved timestamped: {timestamped_path}")

        joblib.dump(self.best_model, os.path.join(MODELS_DIR, "best_classifier.joblib"))
        joblib.dump(self.pca, os.path.join(MODELS_DIR, "pca_transform.joblib"))
        joblib.dump(self.scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
        print(f"  Saved individual components (.joblib)")

        report_path = os.path.join(TRAINING_DIR, f"training_report_{timestamp}.txt")
        with open(report_path, "w") as f:
            f.write("AI VISION SYSTEM - Training Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Date: {timestamp}\n")
            f.write(f"Dataset: LFW (min_faces_per_person={self.min_faces_per_person})\n")
            f.write(f"Best Model: {self.best_model_name}\n\n")
            f.write("All Results:\n")
            for name, res in self.results.items():
                f.write(f"\n  {name}:\n")
                for k, v in res.items():
                    if k != "best_params":
                        f.write(f"    {k}: {v}\n")
            f.write(f"\nClassification Report ({self.best_model_name}):\n")
            y_pred = self.best_model.predict(self.X_test)
            f.write(classification_report(
                self.y_test, y_pred,
                target_names=self.label_encoder.classes_.astype(str),
                zero_division=0
            ))
        print(f"  Saved report: {report_path}")

        print("\n  Model files:")
        for f in os.listdir(MODELS_DIR):
            size = os.path.getsize(os.path.join(MODELS_DIR, f))
            print(f"    {f}: {size/1024:.1f} KB")

    def run_full_training(self):
        print("\n" + "#" * 70)
        print("#" + " " * 68 + "#")
        print("#    AI VISION SYSTEM - FACE RECOGNITION ML TRAINING    #")
        print("#" + " " * 68 + "#")
        print("#" * 70)

        total_start = time.time()

        X, y = self.load_dataset()
        self.preprocess(X, y)
        self.train_models()
        self.hyperparameter_tuning()
        self.select_best_model()
        self.detailed_evaluation()
        self.save_model()

        total_time = time.time() - total_start

        print("\n" + "#" * 70)
        print(f"  TRAINING COMPLETE!")
        print(f"  Total time: {total_time:.1f} seconds")
        print(f"  Best model: {self.best_model_name}")
        print(f"  Accuracy:   {self.results[self.best_model_name]['accuracy']:.4f}")
        print(f"  Model saved to: {MODELS_DIR}")
        print("#" * 70)

        return self.best_model, self.results


def quick_train():
    trainer = FaceMLTrainer(min_faces_per_person=70, resize=0.5)
    return trainer.run_full_training()


def detailed_train():
    trainer = FaceMLTrainer(min_faces_per_person=40, resize=0.6)
    return trainer.run_full_training()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "detailed":
        detailed_train()
    else:
        quick_train()
