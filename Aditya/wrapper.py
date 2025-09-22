import numpy as np
import pandas as pd
import openai
import anthropic
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional, Union
import re
import time
import logging
from functools import wraps
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HRVLLMWrapper:
    """
    Complete wrapper for LLM models used in Human Rights Violation detection
    Implements exact methodology from the paper with LIME/SHAP compatibility
    """

    def __init__(self,
                 model_name: str,
                 api_key: Optional[str] = None,
                 temperature: float = 0.1,
                 max_tokens: int = 10,
                 cache_predictions: bool = True):
        """
        Initialize the LLM wrapper

        Args:
            model_name: Name of the model (gpt-4, gpt-3.5-turbo, claude-2, llama-3.2-1b, mistral-7b)
            api_key: API key for proprietary models
            temperature: Temperature for generation (paper uses 0.1)
            max_tokens: Max tokens for response (paper uses short responses)
            cache_predictions: Whether to cache predictions to avoid redundant API calls
        """
        self.model_name = model_name.lower()
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_predictions = cache_predictions
        self.prediction_cache = {}

        # Initialize model-specific clients
        self._initialize_model()

        # Exact prompts from the paper
        self.prompts = self._get_paper_prompts()

    def _initialize_model(self):
        """Initialize the specific model client/tokenizer"""
        try:
            if 'gpt' in self.model_name:
                openai.api_key = self.api_key
                self.model_type = 'openai'

            elif 'claude' in self.model_name:
                self.anthropic_client = anthropic.Anthropic(api_key=self.api_key)
                self.model_type = 'anthropic'

            elif 'llama' in self.model_name or 'mistral' in self.model_name:
                # For open-source models, load locally
                model_map = {
                    'llama-3.2-1b': 'meta-llama/Llama-3.2-1B',
                    'mistral-7b': 'mistralai/Mistral-7B-v0.1'
                }

                model_id = model_map.get(self.model_name, self.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.model_type = 'huggingface'

                # Set pad token if not exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            logger.error(f"Failed to initialize {self.model_name}: {e}")
            raise

    def _get_paper_prompts(self) -> Dict[str, Dict[str, str]]:
        """
        Exact prompts used in the paper for different configurations
        Based on the paper's methodology section
        """
        prompts = {
            # Zero-shot prompts
            'zero_shot': {
                'english': """Analyze the following social media post and determine whether it contains references to human rights violations.

Human rights violations include but are not limited to: unlawful killings, torture, arbitrary detention, forced displacement, restrictions on freedom of expression, discrimination, and violations of civilian protections during conflict.

Post: {text}

Answer with only "Yes" if the post contains references to human rights violations, or "No" if it does not.""",

                'russian': """Проанализируйте следующий пост в социальных сетях и определите, содержит ли он ссылки на нарушения прав человека.

Нарушения прав человека включают, но не ограничиваются: незаконные убийства, пытки, произвольное задержание, принудительное перемещение, ограничения свободы выражения мнений, дискриминацию и нарушения защиты гражданских лиц во время конфликта.

Пост: {text}

Ответьте только "Да", если пост содержит ссылки на нарушения прав человека, или "Нет", если не содержит."""
            },

            # Few-shot prompts with examples (as used in the paper)
            'few_shot': {
                'english': """Analyze the following social media posts and determine whether they contain references to human rights violations.

Human rights violations include but are not limited to: unlawful killings, torture, arbitrary detention, forced displacement, restrictions on freedom of expression, discrimination, and violations of civilian protections during conflict.

Examples:
Post: "Civilians were targeted in the bombing of residential areas yesterday"
Answer: Yes

Post: "Weather forecast shows rain for the weekend"
Answer: No

Post: "Journalists were detained without charges for reporting on protests"
Answer: Yes

Post: "New restaurant opened in downtown area"
Answer: No

Post: {text}
Answer with only "Yes" or "No":""",

                'russian': """Проанализируйте следующие посты в социальных сетях и определите, содержат ли они ссылки на нарушения прав человека.

Нарушения прав человека включают, но не ограничиваются: незаконные убийства, пытки, произвольное задержание, принудительное перемещение, ограничения свободы выражения мнений, дискриминацию и нарушения защиты гражданских лиц во время конфликта.

Примеры:
Пост: "Гражданские лица стали мишенью при бомбардировке жилых районов вчера"
Ответ: Да

Пост: "Прогноз погоды показывает дождь на выходные"
Ответ: Нет

Пост: "Журналистов задержали без предъявления обвинений за освещение протестов"
Ответ: Да

Пост: "В центре города открылся новый ресторан"
Ответ: Нет

Пост: {text}
Ответьте только "Да" или "Нет":"""
            }
        }
        return prompts

    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API (GPT-3.5, GPT-4)"""
        try:
            # Use the newer client format
            client = openai.OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "Error"

    def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API (Claude)"""
        try:
            response = self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return "Error"

    def _call_huggingface_model(self, prompt: str) -> str:
        """Call local Hugging Face model (LLaMA, Mistral)"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode only the new tokens
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            logger.error(f"Hugging Face model error: {e}")
            return "Error"

    def _get_single_prediction(self, text: str, prompt_template: str) -> str:
        """Get a single prediction from the model"""
        full_prompt = prompt_template.format(text=text)

        # Check cache first
        if self.cache_predictions:
            cache_key = f"{self.model_name}_{hash(full_prompt)}"
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]

        # Call appropriate API
        if self.model_type == 'openai':
            response = self._call_openai_api(full_prompt)
        elif self.model_type == 'anthropic':
            response = self._call_anthropic_api(full_prompt)
        elif self.model_type == 'huggingface':
            response = self._call_huggingface_model(full_prompt)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Cache the response
        if self.cache_predictions:
            self.prediction_cache[cache_key] = response

        return response

    def _convert_response_to_probability(self, response: str) -> float:
        """
        Convert text response to probability score
        Based on the paper's binary classification approach
        """
        response_lower = response.lower().strip()

        # Positive indicators (HRV present)
        positive_indicators = ['yes', 'да', 'positive', 'violation', 'нарушение', '1', 'true']
        # Negative indicators (No HRV)
        negative_indicators = ['no', 'нет', 'negative', 'none', 'отсутствует', '0', 'false']

        # Check for positive indicators
        if any(indicator in response_lower for indicator in positive_indicators):
            return 0.85  # High confidence for positive
        # Check for negative indicators
        elif any(indicator in response_lower for indicator in negative_indicators):
            return 0.15  # Low confidence for positive (high confidence for negative)
        # Ambiguous response
        else:
            logger.warning(f"Ambiguous response: {response}")
            return 0.5  # Neutral probability

    def predict_single(self,
                      text: str,
                      prompt_style: str = 'zero_shot',
                      prompt_language: str = 'english') -> Dict[str, float]:
        """
        Predict for a single text

        Args:
            text: Input text to classify
            prompt_style: 'zero_shot' or 'few_shot'
            prompt_language: 'english' or 'russian'

        Returns:
            Dictionary with probabilities for both classes
        """
        prompt_template = self.prompts[prompt_style][prompt_language]
        response = self._get_single_prediction(text, prompt_template)
        prob_positive = self._convert_response_to_probability(response)

        return {
            'no_hrv': 1 - prob_positive,
            'hrv_present': prob_positive,
            'raw_response': response
        }

    def predict_batch(self,
                     texts: List[str],
                     prompt_style: str = 'zero_shot',
                     prompt_language: str = 'english',
                     batch_delay: float = 0.1) -> List[Dict[str, float]]:
        """
        Predict for a batch of texts with rate limiting

        Args:
            texts: List of input texts
            prompt_style: 'zero_shot' or 'few_shot'
            prompt_language: 'english' or 'russian'
            batch_delay: Delay between API calls to avoid rate limits

        Returns:
            List of prediction dictionaries
        """
        predictions = []

        for i, text in enumerate(texts):
            if i > 0 and batch_delay > 0:
                time.sleep(batch_delay)

            try:
                pred = self.predict_single(text, prompt_style, prompt_language)
                predictions.append(pred)

            except Exception as e:
                logger.error(f"Error predicting text {i}: {e}")
                # Return neutral prediction on error
                predictions.append({
                    'no_hrv': 0.5,
                    'hrv_present': 0.5,
                    'raw_response': 'Error'
                })

        return predictions

    def predict_proba_lime_compatible(self,
                                    texts: List[str],
                                    prompt_style: str = 'zero_shot',
                                    prompt_language: str = 'english') -> np.ndarray:
        """
        LIME-compatible prediction function
        Returns probabilities in sklearn format: [prob_class0, prob_class1]

        Args:
            texts: List of input texts
            prompt_style: 'zero_shot' or 'few_shot'
            prompt_language: 'english' or 'russian'

        Returns:
            numpy array of shape (n_samples, 2) with probabilities
        """
        predictions = self.predict_batch(texts, prompt_style, prompt_language)
        probas = np.array([[pred['no_hrv'], pred['hrv_present']] for pred in predictions])
        return probas

    def predict_shap_compatible(self,
                              texts: List[str],
                              prompt_style: str = 'zero_shot',
                              prompt_language: str = 'english') -> np.ndarray:
        """
        SHAP-compatible prediction function
        Returns only positive class probabilities for binary classification

        Args:
            texts: List of input texts
            prompt_style: 'zero_shot' or 'few_shot'
            prompt_language: 'english' or 'russian'

        Returns:
            numpy array of positive class probabilities
        """
        predictions = self.predict_batch(texts, prompt_style, prompt_language)
        probas = np.array([pred['hrv_present'] for pred in predictions])
        return probas

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model configuration"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'cache_enabled': self.cache_predictions,
            'cached_predictions': len(self.prediction_cache)
        }

    def clear_cache(self):
        """Clear the prediction cache"""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")

    def save_cache(self, filepath: str):
        """Save prediction cache to file"""
        with open(filepath, 'w') as f:
            json.dump(self.prediction_cache, f)
        logger.info(f"Cache saved to {filepath}")

    def load_cache(self, filepath: str):
        """Load prediction cache from file"""
        try:
            with open(filepath, 'r') as f:
                self.prediction_cache = json.load(f)
            logger.info(f"Cache loaded from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Cache file {filepath} not found")

# Example usage and testing
def test_wrapper():
    """Test the wrapper with sample data"""

    sample_texts = [
        "Civilians were targeted in bombing attacks yesterday",
        "Weather forecast shows sunny skies",
        "Journalists detained without charges for reporting",
        "New coffee shop opened downtown"
    ]

    # Test with GPT-4 (requires API key)
    try:
        wrapper = HRVLLMWrapper('gpt-4', api_key='your-api-key-here')

        # Test single prediction
        result = wrapper.predict_single(sample_texts[0], 'zero_shot', 'english')
        print(f"Single prediction: {result}")

        # Test batch prediction
        batch_results = wrapper.predict_batch(sample_texts[:2], 'zero_shot', 'english')
        print(f"Batch predictions: {batch_results}")

        # Test LIME compatibility
        lime_probas = wrapper.predict_proba_lime_compatible(sample_texts[:2])
        print(f"LIME-compatible probabilities: {lime_probas}")

        # Test SHAP compatibility
        shap_probas = wrapper.predict_shap_compatible(sample_texts[:2])
        print(f"SHAP-compatible probabilities: {shap_probas}")

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_wrapper()
