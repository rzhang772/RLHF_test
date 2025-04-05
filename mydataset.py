import datasets
import torch

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

class MyDataset():
    def __init__(self, dataset_name: str):
        self.dataset = datasets.load_dataset(dataset_name, 'main')['train']
        print(f"Loaded {len(self.dataset)} examples from {dataset_name}")
        self.formated_data = []

    def _build_prompt(self, messages: str):
        return "\n".join([msg["content"].strip() for msg in messages])
    
    def _extract_answer_from_dataset(self, answer_text: str):
        if "####" not in answer_text:
            return None
        return answer_text.split("####")[1].strip()
    
    def _formate_data(self):
        for chat in self.dataset:
            prompt_str = self._build_prompt([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chat["question"]}
            ])
            formated_chat = {
                "prompt": prompt_str,
                "answer": self._extract_answer_from_dataset(chat["answer"])
            }
            self.formated_data.append(formated_chat)

    def prepare_data(self):
        self._formate_data()
        return self.formated_data