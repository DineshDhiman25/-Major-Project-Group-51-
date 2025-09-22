# LIME and SHAP Integration with HRVLLMWrapper
# This code shows how to use the wrapper with explainability tools

import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class HRVExplainabilityAnalyzer:
    """
    Complete integration of LIME and SHAP with the paper's LLM wrapper
    Provides explanations for human rights violation detection
    """

    def __init__(self,
                 model_wrapper: HRVLLMWrapper,
                 prompt_style: str = 'zero_shot',
                 prompt_language: str = 'english'):
        """
        Initialize explainability analyzer

        Args:
            model_wrapper: Instance of HRVLLMWrapper
            prompt_style: 'zero_shot' or 'few_shot'
            prompt_language: 'english' or 'russian'
        """
        self.wrapper = model_wrapper
        self.prompt_style = prompt_style
        self.prompt_language = prompt_language

        # Initialize LIME explainer
        self.lime_explainer = LimeTextExplainer(
            class_names=['No HRV', 'HRV Present'],
            feature_selection='auto',
            verbose=True
        )

        # Cache for explanations
        self.explanation_cache = {}

    def _create_prediction_function(self):
        """Create a prediction function for LIME/SHAP compatibility"""
        def predict_fn(texts):
            if isinstance(texts, str):
                texts = [texts]
            return self.wrapper.predict_proba_lime_compatible(
                texts, self.prompt_style, self.prompt_language
            )
        return predict_fn

    def explain_with_lime(self,
                         text: str,
                         num_features: int = 15,
                         num_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single text

        Args:
            text: Text to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME to generate

        Returns:
            Dictionary containing explanation results
        """
        # Check cache first
        cache_key = f"lime_{hash(text)}_{self.prompt_style}_{self.prompt_language}"
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]

        predict_fn = self._create_prediction_function()

        # Generate LIME explanation
        explanation = self.lime_explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )

        # Extract explanation data
        explanation_data = {
            'text': text,
            'prediction_proba': predict_fn([text])[0],
            'local_prediction': explanation.local_pred[1],  # HRV class probability
            'local_r2_score': explanation.score,  # Local model fit
            'feature_weights': explanation.as_list(),  # [(feature, weight), ...]
            'intercept': explanation.intercept[1],
            'explanation_obj': explanation  # Full LIME explanation object
        }

        # Cache the result
        self.explanation_cache[cache_key] = explanation_data

        return explanation_data

    def explain_with_shap(self,
                         texts: List[str],
                         max_evals: int = 200,
                         background_size: int = 50) -> Dict[str, Any]:
        """
        Generate SHAP explanations for multiple texts

        Args:
            texts: List of texts to explain
            max_evals: Maximum number of model evaluations
            background_size: Size of background dataset for SHAP

        Returns:
            Dictionary containing SHAP results
        """
        # Create SHAP-compatible prediction function
        def predict_fn(texts):
            return self.wrapper.predict_shap_compatible(
                texts, self.prompt_style, self.prompt_language
            )

        # Create background dataset (sample from input texts)
        background_texts = texts[:min(background_size, len(texts))]

        # Initialize SHAP explainer
        explainer = shap.Explainer(
            predict_fn,
            shap.maskers.Text(tokenizer=r"\\W+")
        )

        # Limit texts to process based on max_evals
        texts_to_process = texts[:min(len(texts), max_evals // 20)]

        # Generate SHAP values
        shap_values = explainer(texts_to_process)

        return {
            'texts': texts_to_process,
            'shap_values': shap_values.values,
            'base_values': shap_values.base_values,
            'data': shap_values.data,
            'explainer': explainer,
            'shap_explanation': shap_values
        }

    def analyze_predictions_with_explanations(self,
                                           dataset: pd.DataFrame,
                                           text_column: str = 'text',
                                           label_column: str = 'label',
                                           sample_size: int = 50) -> Dict[str, Any]:
        """
        Comprehensive analysis combining predictions and explanations

        Args:
            dataset: DataFrame with texts and labels
            text_column: Name of text column
            label_column: Name of label column
            sample_size: Number of samples to analyze

        Returns:
            Dictionary with analysis results
        """
        # Sample data for analysis
        sample_data = dataset.sample(n=min(sample_size, len(dataset)), random_state=42)

        results = {
            'predictions': [],
            'lime_explanations': [],
            'performance_metrics': {},
            'error_analysis': {},
            'feature_importance': {}
        }

        # Get predictions and explanations
        for idx, row in sample_data.iterrows():
            text = row[text_column]
            true_label = row[label_column]

            try:
                # Get prediction
                pred_result = self.wrapper.predict_single(
                    text, self.prompt_style, self.prompt_language
                )

                # Get LIME explanation
                lime_exp = self.explain_with_lime(text)

                # Combine results
                combined_result = {
                    'text': text,
                    'true_label': true_label,
                    'predicted_proba': pred_result['hrv_present'],
                    'predicted_label': 1 if pred_result['hrv_present'] > 0.5 else 0,
                    'correct_prediction': (pred_result['hrv_present'] > 0.5) == (true_label == 1),
                    'lime_score': lime_exp['local_r2_score'],
                    'lime_prediction': lime_exp['local_prediction'],
                    'feature_weights': lime_exp['feature_weights'],
                    'raw_response': pred_result['raw_response']
                }

                results['predictions'].append(combined_result)
                results['lime_explanations'].append(lime_exp)

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        # Calculate performance metrics
        if results['predictions']:
            correct_preds = [p for p in results['predictions'] if p['correct_prediction']]
            results['performance_metrics'] = {
                'accuracy': len(correct_preds) / len(results['predictions']),
                'avg_lime_score': np.mean([p['lime_score'] for p in results['predictions']]),
                'avg_confidence': np.mean([p['predicted_proba'] for p in results['predictions']]),
                'total_samples': len(results['predictions'])
            }

        # Error analysis
        correct_preds = [p for p in results['predictions'] if p['correct_prediction']]
        incorrect_preds = [p for p in results['predictions'] if not p['correct_prediction']]

        results['error_analysis'] = {
            'correct_predictions': len(correct_preds),
            'incorrect_predictions': len(incorrect_preds),
            'avg_lime_score_correct': np.mean([p['lime_score'] for p in correct_preds]) if correct_preds else 0,
            'avg_lime_score_incorrect': np.mean([p['lime_score'] for p in incorrect_preds]) if incorrect_preds else 0
        }

        return results

    def compare_human_disagreement_cases(self,
                                       dataset: pd.DataFrame,
                                       agreement_column: str = 'human_agreement',
                                       text_column: str = 'text',
                                       label_column: str = 'label') -> Dict[str, Any]:
        """
        Analyze model explanations on cases where humans disagreed
        This directly implements the paper's analysis of the 184 disagreement cases
        """
        # Separate agreed and disagreed cases
        agreed_cases = dataset[dataset[agreement_column] == True].sample(n=30, random_state=42)
        disagreed_cases = dataset[dataset[agreement_column] == False].sample(n=30, random_state=42)

        results = {
            'agreed_cases': {},
            'disagreed_cases': {},
            'comparison': {}
        }

        # Analyze agreed cases
        agreed_analysis = self.analyze_predictions_with_explanations(
            agreed_cases, text_column, label_column, sample_size=30
        )
        results['agreed_cases'] = agreed_analysis

        # Analyze disagreed cases
        disagreed_analysis = self.analyze_predictions_with_explanations(
            disagreed_cases, text_column, label_column, sample_size=30
        )
        results['disagreed_cases'] = disagreed_analysis

        # Compare results
        results['comparison'] = {
            'accuracy_agreed': agreed_analysis['performance_metrics']['accuracy'],
            'accuracy_disagreed': disagreed_analysis['performance_metrics']['accuracy'],
            'lime_score_agreed': agreed_analysis['performance_metrics']['avg_lime_score'],
            'lime_score_disagreed': disagreed_analysis['performance_metrics']['avg_lime_score'],
            'confidence_agreed': agreed_analysis['performance_metrics']['avg_confidence'],
            'confidence_disagreed': disagreed_analysis['performance_metrics']['avg_confidence']
        }

        return results

    def visualize_explanations(self, explanation_results: Dict[str, Any], save_plots: bool = True):
        """
        Create visualizations for explanation analysis
        """
        if not explanation_results['predictions']:
            print("No predictions to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Accuracy vs LIME Score
        predictions = explanation_results['predictions']
        lime_scores = [p['lime_score'] for p in predictions]
        accuracies = [1 if p['correct_prediction'] else 0 for p in predictions]

        axes[0, 0].scatter(lime_scores, accuracies, alpha=0.6)
        axes[0, 0].set_xlabel('LIME Local Model R² Score')
        axes[0, 0].set_ylabel('Prediction Accuracy')
        axes[0, 0].set_title('Prediction Accuracy vs LIME Model Quality')

        # 2. Confidence distribution
        confidences = [p['predicted_proba'] for p in predictions]
        axes[0, 1].hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Model Confidence (HRV Probability)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Model Confidence')

        # 3. Feature importance (top features)
        all_features = []
        for p in predictions:
            for feature, weight in p['feature_weights']:
                all_features.append((feature, abs(weight)))

        # Get top features by absolute weight
        feature_df = pd.DataFrame(all_features, columns=['Feature', 'Weight'])
        top_features = feature_df.groupby('Feature')['Weight'].mean().sort_values(ascending=False).head(10)

        axes[1, 0].barh(range(len(top_features)), top_features.values)
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features.index)
        axes[1, 0].set_xlabel('Average Absolute Feature Weight')
        axes[1, 0].set_title('Top 10 Most Important Features')

        # 4. Correct vs Incorrect predictions comparison
        correct_conf = [p['predicted_proba'] for p in predictions if p['correct_prediction']]
        incorrect_conf = [p['predicted_proba'] for p in predictions if not p['correct_prediction']]

        axes[1, 1].boxplot([correct_conf, incorrect_conf], labels=['Correct', 'Incorrect'])
        axes[1, 1].set_ylabel('Model Confidence')
        axes[1, 1].set_title('Confidence Distribution: Correct vs Incorrect')

        plt.tight_layout()

        if save_plots:
            plt.savefig('hrv_explanation_analysis.png', dpi=300, bbox_inches='tight')

        plt.show()

    def generate_explanation_report(self,
                                  analysis_results: Dict[str, Any],
                                  model_name: str) -> str:
        """
        Generate a comprehensive report of explanation analysis
        """
        report = []
        report.append("="*80)
        report.append(f"EXPLAINABILITY ANALYSIS REPORT: {model_name.upper()}")
        report.append(f"Configuration: {self.prompt_style} prompting, {self.prompt_language} language")
        report.append("="*80)
        report.append("")

        # Performance summary
        metrics = analysis_results['performance_metrics']
        report.append("PERFORMANCE SUMMARY:")
        report.append(f"  Total samples analyzed: {metrics['total_samples']}")
        report.append(f"  Model accuracy: {metrics['accuracy']:.3f}")
        report.append(f"  Average LIME score: {metrics['avg_lime_score']:.3f}")
        report.append(f"  Average confidence: {metrics['avg_confidence']:.3f}")
        report.append("")

        # Error analysis
        error_analysis = analysis_results['error_analysis']
        report.append("ERROR ANALYSIS:")
        report.append(f"  Correct predictions: {error_analysis['correct_predictions']}")
        report.append(f"  Incorrect predictions: {error_analysis['incorrect_predictions']}")
        report.append(f"  LIME score (correct): {error_analysis['avg_lime_score_correct']:.3f}")
        report.append(f"  LIME score (incorrect): {error_analysis['avg_lime_score_incorrect']:.3f}")
        report.append("")

        # Feature importance insights
        all_features = []
        for p in analysis_results['predictions']:
            for feature, weight in p['feature_weights']:
                all_features.append((feature, weight))

        if all_features:
            feature_df = pd.DataFrame(all_features, columns=['Feature', 'Weight'])
            top_positive = feature_df[feature_df['Weight'] > 0].groupby('Feature')['Weight'].mean().sort_values(ascending=False).head(5)
            top_negative = feature_df[feature_df['Weight'] < 0].groupby('Feature')['Weight'].mean().sort_values().head(5)

            report.append("TOP FEATURES INDICATING HRV:")
            for feature, weight in top_positive.items():
                report.append(f"  '{feature}': {weight:.3f}")

            report.append("\\nTOP FEATURES INDICATING NO HRV:")
            for feature, weight in top_negative.items():
                report.append(f"  '{feature}': {weight:.3f}")

        return "\\n".join(report)

# Usage example integrating with the paper's methodology
def main_analysis_example():
    """
    Example of how to integrate this with the paper's exact methodology
    """

    # 1. Initialize the wrapper with paper's configuration
    wrapper = HRVLLMWrapper(
        model_name='gpt-4',
        api_key='your-api-key-here',
        temperature=0.1,  # Paper's setting
        max_tokens=10     # Paper's setting
    )

    # 2. Load the paper's dataset (1000 samples)
    # dataset = pd.read_csv('hrv_dataset_1000.csv')
    # For demo, create sample data
    sample_data = pd.DataFrame({
        'text': [
            "Civilians targeted in bombing attacks",
            "Weather forecast sunny today",
            "Journalists detained without charges",
            "Restaurant opens new location"
        ],
        'label': [1, 0, 1, 0],
        'human_agreement': [True, True, False, True]
    })

    # 3. Initialize explainability analyzer
    analyzer = HRVExplainabilityAnalyzer(
        model_wrapper=wrapper,
        prompt_style='zero_shot',    # Paper's configuration
        prompt_language='english'    # Can also use 'russian'
    )

    # 4. Run comprehensive analysis
    results = analyzer.analyze_predictions_with_explanations(
        sample_data,
        sample_size=4
    )

    # 5. Analyze human disagreement cases (paper's 184 samples)
    disagreement_analysis = analyzer.compare_human_disagreement_cases(sample_data)

    # 6. Generate visualizations
    analyzer.visualize_explanations(results)

    # 7. Generate report
    report = analyzer.generate_explanation_report(results, 'GPT-4')
    print(report)

    return results, disagreement_analysis

if __name__ == "__main__":
    # Run the analysis
    results, disagreement = main_analysis_example()
