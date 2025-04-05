from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import random
import copy
import re
import wandb
import time
import os

class MyGRPOTrainer:
    def __init__(self, model, tokenizer, device_ids, device):
        self.tokenizer = tokenizer
        self.model = model
        self.num_gpus = len(device_ids)
        self.device_ids = device_ids
        self.device = device # device to generate completions
        self.config_model()
        self.reward_function = self.combined_reward
    
    def config_model(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

    def optimize_model_memory(self):
        self.model.train()
        self.model.config.use_cache = False
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        self.model.gradient_checkpointing_enable()
    
    def train_with_GRPO(self, train_data, eval_data, num_iterations=1, num_steps=500, batch_size=4, num_generations=4, max_completion_length=128, beta=0.1,
                              learning_rate=5e-6, mu=3, epsilon=0.2, checkpoint_path=None, checkpoint_interval = 50):
        assert self.device_ids is not None, "device_ids must be provided"
        assert len(self.device_ids) > 1, "device_ids must have more than one device"

        self.optimize_model_memory()

        self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        print(f"Model wrapped with nn.DataParallel on devices: {self.device_ids}")

        for iteration in range(num_iterations):
            print(f"\nIteration {iteration}/{num_iterations}")
            ref_model = copy.deepcopy(self.model.module)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
            print("Reference model updated")
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

            for step in range(num_steps):
                batch_samples = random.sample(train_data, batch_size)
                with torch.no_grad():
                    roullout_data = self.generate_rollout_data(ref_model, batch_samples, num_generations, max_completion_length)

                for grpo_iter in range(mu):
                    loss, avg_reward = self.grpo_loss(roullout_data, self.reward_function, beta, epsilon)
                    self.model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    wandb.log(
                        {
                            "iteration": iteration,
                            "step": step,
                            "grpo_iter": grpo_iter,
                            "loss": loss.item(),
                            "avg_reward": avg_reward
                        }
                    )

                    print(f"Iteration {iteration+1}/{num_iterations}, Step {step+1}/{num_steps}, "
                      f"GRPO iter {grpo_iter+1}/{mu}, loss: {loss.item():.4f}")
                if (step+1) % checkpoint_interval == 0 and step > 0:
                    temp_path = os.path.join(checkpoint_path, f"checkpoint-iteration{iteration}-step{step}")
                    self.model.module.save_pretrained(temp_path)
                    self.tokenizer.save_pretrained(temp_path)
                    wandb.log({
                        "Train Accuracy": self.evaluate(self.model.module, self.tokenizer, random.sample(train_data, 30), self.device),
                        "Eval Accuracy": self.evaluate(self.model.module, self.tokenizer, eval_data, self.device),
                    })
                    print(f"Checkpoint saved at {temp_path}")
            

                    
        # Unwrap the model and claer the cache
        self.model = self.model.module
        self.model.config.use_cache = True
        self.model.gradient_checkpointing_disable()
        self.model.eval()
        torch.cuda.empty_cache()
        print("Model unwrapped and cache cleared")
        return self.model, self.tokenizer

    def grpo_loss(self, rollout_data, reward_function, beta=0.01, epsilon=0.2):
        '''
        Compute the GRPO loss
        '''

        input_ids = rollout_data["input_ids"]
        attention_mask = rollout_data["attention_mask"]
        completion_mask = rollout_data["completion_mask"]
        logits_to_keep = rollout_data["logits_to_keep"]
        old_log_probs = rollout_data["old_log_probs"]
        ref_log_probs = rollout_data["ref_log_probs"]

        log_probs = self.compute_log_prob(self.model.module, input_ids, attention_mask, logits_to_keep)
        ratio = torch.exp(log_probs - old_log_probs)

        rewards = torch.tensor(
            reward_function(prompts=rollout_data["repeated_prompts"], completions=rollout_data["formatted_completions"], answer=rollout_data["repeated_answers"]),
            dtype=torch.float32,
            device=self.device
        ) # shape (batch_size * num_generations)

        batch_size = rollout_data["batch_size"]
        num_generations = rollout_data["num_generations"]
        rewards = rewards.view(batch_size, num_generations)
        avg_reward = rewards.mean().item()

        mean_reward = rewards.mean(dim=1).repeat_interleave(num_generations)  # num_generations is the group num. shape (batch_size * num_generations)
        std_reward = rewards.std(dim=1).repeat_interleave(num_generations)
        advantages = ((rewards.view(-1)-mean_reward)/(std_reward+1e-4)).unsqueeze(1)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
        surrogate_loss = torch.min(surr1, surr2)

        kl = torch.exp(ref_log_probs-log_probs) - (ref_log_probs - log_probs) - 1
        per_token_loss = surrogate_loss - beta * kl
        loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        return loss, avg_reward

    
    def generate_rollout_data(self, ref_model, batch_samples, num_generations, max_completion_length):
        '''
        concatenate the prompt with the completion at each step

        self.model.module: the original model without nn.DataParallel
        '''
        prompts = [sample["prompt"] for sample in batch_samples]
        answers = [sample["answer"] for sample in batch_samples]

        with torch.no_grad():
            prompt_ids, prompt_mask, completion_ids, completion_mask = self.generate_completions(prompts, num_generations, max_completion_length)

            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

            logits_to_keep = completion_ids.size(1)

            old_log_probs = self.compute_log_prob(self.model.module, input_ids, attention_mask, logits_to_keep)
            ref_log_probs = self.compute_log_prob(ref_model, input_ids, attention_mask, logits_to_keep)

        formatted_completions = [[{'content': self.tokenizer.decode(sample, skip_special_tokens=True)}] for sample in completion_ids]
        repeated_prompts = [p for p in prompts for _ in range(num_generations)]
        repeated_answers = [a for a in answers for _ in range(num_generations)]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "logits_to_keep": logits_to_keep,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "formatted_completions": formatted_completions,
            "repeated_prompts": repeated_prompts,
            "repeated_answers": repeated_answers,
            "batch_size": len(batch_samples),
            "num_generations": num_generations
        }


    def generate_completions(self, prompt, num_generations, max_completion_length):
        '''
        generate multiple completions for the same prompt
        '''
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        prompt_ids = inputs["input_ids"].to(self.device)
        prompt_mask = inputs["attention_mask"].to(self.device)
        prompt_length = prompt_ids.size(1)

        #
        prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

        outputs = self.model.module.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_length= max_completion_length,
            do_sample=True,
            temperature=1.0,
            # num_return_sequences=num_generations,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # return_dict_in_generate=True
            early_stopping=False,
        )

        completion_ids = outputs[:, prompt_length:]
        completion_mask = self.create_completion_mask(completion_ids)

        return prompt_ids, prompt_mask, completion_ids, completion_mask
        
    def create_completion_mask(self, completion_ids):
        is_eos = completion_ids == self.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
        mask_exists = is_eos.any(dim=1)
        eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
        sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        return (sequence_indices <= eos_idx.unsqueeze(1)).int()

    def compute_log_prob(self, model, input_ids, attention_mask, logits_to_keep):
        # print(f"logits_to_keep: {logits_to_keep}, type: {logits_to_keep.dtype}")
        logits = model(input_ids, attention_mask=attention_mask).logits[:,:-1,:]
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        return log_probs
    
    def correctness_reward(self, completions, answer):
        responses = [completion[0]['content'] for completion in completions]
        extracted_answers = [self._extract_answer_from_completion(response) for response in responses]
        rewards = []
        for r, a in zip(extracted_answers, answer):
            if r is not None and r == a:
                rewards.append(2.0)
            else:
                r_num = self._extract_single_number(str(r))
                a_num = self._extract_single_number(str(a))
                if r_num is not None and a_num is not None and r_num == a_num:
                    rewards.append(1.5)
                else:
                    rewards.append(0.0)
        return rewards
    
    def format_reward(self, completions):
        responses = [completion[0]['content'] for completion in completions]
        rewards = []
        format_scores = []
        for response in responses:
            score = 0.0
            if "<reasoning>" in response: score += 0.2
            if "</reasoning>" in response: score += 0.2
            if "<answer>" in response: score += 0.2
            if "</answer>" in response: score += 0.2
            rewards.append(score)
            format_scores.append(score)
        return rewards
    def combined_reward(self, prompts, completions, answer):
        correctness_scores = self.correctness_reward(completions=completions, answer=answer)
        format_scores = self.format_reward(completions)

        # Combine rewards - correctness is weighted more heavily
        combined_rewards = []
        for c_score, f_score in zip(correctness_scores, format_scores):
            # Correctness score range: 0.0 to 2.0
            # Format score range: 0.0 to 0.8
            # Total range: 0.0 to 2.8
            combined_rewards.append(c_score + f_score)

        return combined_rewards
    
    def _extract_answer_from_completion(self, completion):
        parts = completion.split("<answer>")
        if len(parts) < 2:  # No <answer> tag found
            return None
        last_part = parts[-1]

        # Extract content up to </answer>
        if "</answer>" not in last_part:
            return None
        answer = last_part.split("</answer>")[0].strip()
        return None if answer == "..." else answer
    
    def _extract_single_number(self, text):
        numbers = re.findall(r'-?\d*\.?\d+', text)
        return float(numbers[0]) if len(numbers) == 1 else None
    
    def _extract_last_number(self, text):
        numbers = re.findall(r'\d+(?:\.\d+)?', text)  # Matches integers and decimals
        return numbers[-1] if numbers else None
    
    def evaluate(self, model, tokenizer, eval_data, device):
        model.eval()
        correct = 0
        total = len(eval_data)

        for i, example in enumerate(eval_data):
            question = example['prompt']
            answer = example['answer']
            try:
                input_ids = tokenizer.encode(question, return_tensors="pt").to(device)
                # print(f"Input IDs: {input_ids}")
                output = model.generate(
                    input_ids, 
                    max_length=512,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    forced_eos_token_id=tokenizer.eos_token_id, 
                    early_stopping=False, 
                    )
                # print(f"Output IDs: {output}")
                response_all = tokenizer.decode(output[0], skip_special_tokens=True)
                # print(f"Response: {response_all}")
                predicted_number = self._extract_answer_from_completion(response_all)
                # print(f"Response Answer: {response_answer}")
                # exit(0)

                if predicted_number is not None and predicted_number == answer:
                    is_correct = True
                else:
                    predicted_number = self._extract_single_number(str(response_all))
                    actual_number = self._extract_single_number(str(answer))
                    if predicted_number is not None and actual_number is not None and predicted_number == actual_number:
                        is_correct = True
                    else:
                        predicted_number = self._extract_last_number(str(response_all))
                        actual_number = self._extract_last_number(str(answer))
                        is_correct = (predicted_number is not None and actual_number is not None and predicted_number == actual_number)
                if is_correct:
                    correct += 1
                
                # print(f"\nQuestion: {question}")
                # print(f"\nExcepted Answer: {answer}")
                # print(f"\nPredicted Answer: {predicted_number}")
                # print("\nCorrect:", "yes" if is_correct else "no")
                # print("------------------------------------------------------------\n")
            except Exception as e:
                print(f"Error: {e}")
        
        accuracy = (correct / total) * 100
        # print(f"Accuracy: {accuracy:.2f}%")
        # print(f"================>>> Evaluation completed <<<================")
        return accuracy