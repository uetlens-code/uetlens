import pandas as pd
import numpy as np
import joblib
import torch
import gc
import os
import sys
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

current_dir = os.path.dirname(os.path.abspath(__file__))
opensae_src_path = os.path.join(current_dir, "OpenSAE", "src")
if opensae_src_path not in sys.path:
    sys.path.insert(0, opensae_src_path)

from opensae.transformer_with_sae import TransformerWithSae
from transformers import AutoTokenizer, AutoModelForCausalLM

class EventTypeSVMClassifier:
    def __init__(self, model_path, sae_path_template, cuda_devices="4,5,6,7", event_types=None, layer_vector_dict=None):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        self.model_path = model_path
        self.sae_path_template = sae_path_template
        self.tokenizer = None
        self.svm_model = None
        self.feature_info = None
        self.feature_columns = None
        self.label_encoder = LabelEncoder()
        self.event_types = event_types if event_types else [
            'Attack', 'Transport', 'Die', 'Injure', 'Meet', 'Elect', 'Trial'
        ]
        self.layer_vector_dict = layer_vector_dict if layer_vector_dict else {
            0: [147991],
            8: [218025],
            15: [246929],
            24: [194013],
            30: [179528]
        }
        
    def setup_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer
    
    def process_layer(self, layer_idx, vector_ids, sentences):
        base_transformer = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="balanced",
            torch_dtype=torch.float16,
            local_files_only=True
        )
        sae_path = self.sae_path_template.format(layer_idx)
        model = TransformerWithSae(
            transformer=base_transformer,
            sae=sae_path,
            use_multi_gpu=True
        )
        layer_features = []
        for sentence in sentences:
            layer_results = model.get_specific_activations(
                texts=[sentence],
                layer_vector_dict={layer_idx: vector_ids},
                tokenizer=self.tokenizer
            )
            layer_activations = layer_results["layer_activations"][layer_idx]
            sentence_features = []
            for vector_id in vector_ids:
                activation = layer_activations.get(vector_id, 0.0)
                sentence_features.append(activation)
            layer_features.append(sentence_features)
        del model
        del base_transformer
        gc.collect()
        torch.cuda.empty_cache()
        return layer_features
    
    def extract_features(self, texts):
        self.setup_tokenizer()
        all_features = [[] for _ in range(len(texts))]
        for layer_idx, vector_ids in self.layer_vector_dict.items():
            layer_features = self.process_layer(layer_idx, vector_ids, texts)
            for i, sentence_features in enumerate(layer_features):
                all_features[i].extend(sentence_features)
        return all_features
    
    def train_svm(self, training_data, output_dir="out/svm"):
        train_texts = [item[0] for item in training_data]
        train_labels = [item[1] for item in training_data]
        
        self.label_encoder.fit(self.event_types)
        encoded_labels = self.label_encoder.transform(train_labels)
        
        all_features = self.extract_features(train_texts)
        
        feature_columns = []
        for layer_idx, vector_ids in self.layer_vector_dict.items():
            for vector_id in vector_ids:
                feature_columns.append(f"feature_layer{layer_idx}_vec{vector_id}")
        
        self.feature_columns = feature_columns
        
        data_dicts = []
        for i, features in enumerate(all_features):
            row_dict = {}
            for col_idx, feature_value in enumerate(features):
                row_dict[feature_columns[col_idx]] = feature_value
            row_dict['label'] = encoded_labels[i]
            row_dict['text'] = train_texts[i]
            row_dict['event_type'] = train_labels[i]
            data_dicts.append(row_dict)
        
        df = pd.DataFrame(data_dicts)
        
        os.makedirs(output_dir, exist_ok=True)
        dataset_path = os.path.join(output_dir, 'event_type_features_dataset.csv')
        df.to_csv(dataset_path, index=False)
        
        X = df[feature_columns]
        y = df['label']
        
        self.svm_model = svm.SVC(kernel='linear', random_state=42, probability=True)
        self.svm_model.fit(X, y)
        
        self.feature_info = {
            'feature_names': list(X.columns),
            'num_features': X.shape[1]
        }
        
        model_path = os.path.join(output_dir, 'event_type_classifier.pkl')
        joblib.dump(self.svm_model, model_path)
        
        label_encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, label_encoder_path)
        
        feature_info_path = os.path.join(output_dir, 'feature_info.pkl')
        feature_data = {
            'feature_names': list(X.columns),
            'num_features': X.shape[1],
            'feature_columns': feature_columns,
            'event_types': self.event_types
        }
        joblib.dump(feature_data, feature_info_path)
        
        train_predictions = self.svm_model.predict(X)
        train_accuracy = accuracy_score(y, train_predictions)
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        return self.svm_model, self.feature_info
    
    def load_model(self, output_dir="out/svm"):
        svm_path = os.path.join(output_dir, 'event_type_classifier.pkl')
        feature_info_path = os.path.join(output_dir, 'feature_info.pkl')
        label_encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
        
        if (os.path.exists(svm_path) and 
            os.path.exists(feature_info_path) and 
            os.path.exists(label_encoder_path)):
            
            self.svm_model = joblib.load(svm_path)
            feature_data = joblib.load(feature_info_path)
            self.feature_info = feature_data
            self.feature_columns = feature_data.get('feature_columns', [])
            self.label_encoder = joblib.load(label_encoder_path)
            self.event_types = feature_data.get('event_types', self.event_types)
            return True
        return False
    
    def predict(self, texts, true_labels=None, output_dir="out/svm"):
        if self.svm_model is None:
            if not self.load_model(output_dir):
                return None, None
        
        all_features = self.extract_features(texts)
        predictions = []
        probabilities = []
        
        for i, features in enumerate(all_features):
            features = np.array(features)
            if len(features) != self.feature_info['num_features']:
                continue
            features_df = pd.DataFrame([features], columns=self.feature_columns)
            prediction = self.svm_model.predict(features_df)[0]
            predictions.append(prediction)
            prob = self.svm_model.predict_proba(features_df)[0]
            probabilities.append(prob)
        
        decoded_predictions = self.label_encoder.inverse_transform(predictions)
        
        if true_labels is not None and len(true_labels) == len(predictions):
            encoded_true = self.label_encoder.transform(true_labels)
            accuracy = accuracy_score(encoded_true, predictions)
            print(f"Total Accuracy: {accuracy:.4f}")
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                encoded_true, predictions, average='weighted'
            )
            print("\nPerformance Metrics:")
            print("| Metric | Score |")
            print("|--------|-------|")
            print(f"| Precision | {precision:.4f} |")
            print(f"| Recall | {recall:.4f} |")
            print(f"| F1-Score | {f1:.4f} |")
            
            per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
                encoded_true, predictions, average=None
            )
            print("\nPer-Class Performance:")
            print("| Event Type | Precision | Recall | F1-Score |")
            print("|------------|-----------|--------|----------|")
            for i, event_type in enumerate(self.event_types):
                print(f"| {event_type} | {per_class_precision[i]:.4f} | {per_class_recall[i]:.4f} | {per_class_f1[i]:.4f} |")
            
            cm = confusion_matrix(encoded_true, predictions)
            cm_df = pd.DataFrame(cm, index=self.event_types, columns=self.event_types)
            print("\nConfusion Matrix:")
            print(cm_df)
        
        print("\nDetailed Results:")
        for i in range(len(texts)):
            print(f"{i+1}. {texts[i][:40]}...")
            print(f"   True: {true_labels[i]}, Pred: {decoded_predictions[i]}, {'✓' if true_labels[i]==decoded_predictions[i] else '✗'}")
            print(f"   Confidence: {probabilities[i][self.label_encoder.transform([decoded_predictions[i]])[0]]:.4f}")
            
        return decoded_predictions, probabilities
    
    def get_class_probabilities(self, text, output_dir="out/svm"):
        features = self.extract_features([text])[0]
        features_df = pd.DataFrame([features], columns=self.feature_columns)
        
        if self.svm_model is None:
            if not self.load_model(output_dir):
                return None
        
        probabilities = self.svm_model.predict_proba(features_df)[0]
        result = {}
        for i, event_type in enumerate(self.event_types):
            result[event_type] = probabilities[i]
        
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))