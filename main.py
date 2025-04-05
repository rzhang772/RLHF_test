import pandas as pd
import numpy as np
import random
import copy
import os
import re
import wandb
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset

from mydataset import MyDataset
from mygrpotrainer import MyGRPOTrainer

def set_seed(seed: int = 42):
    """
    Set seed for reproducibility
    """

    # Set seed for random module
    random.seed(seed)
    # Set seed for numpy module
    np.random.seed(seed)
    # Set seed for pytorch module
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, tokenizer, eval_data, device):
    """
    Evaluate the model on the eval_data
    """
    model.eval()
    correct = 0
    total = len(eval_data)
    print(f"================>>> Evaluating on {total} examples... <<<================")

    for i, example in enumerate(eval_data):
        question = example['prompt']
        answer = example['answer']
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>Example {i+1}/{total}")
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
            print(f"Response: {response_all}")
            predicted_number = extract_answer_from_model_output(response_all)
            # print(f"Response Answer: {response_answer}")
            # exit(0)

            if predicted_number is not None and predicted_number == answer:
                is_correct = True
            else:
                predicted_number = extract_single_number(str(response_all))
                actual_number = extract_single_number(str(answer))
                if predicted_number is not None and actual_number is not None and predicted_number == actual_number:
                    is_correct = True
                else:
                    predicted_number = extract_last_number(str(response_all))
                    actual_number = extract_last_number(str(answer))
                    is_correct = (predicted_number is not None and actual_number is not None and predicted_number == actual_number)
            if is_correct:
                correct += 1
            
            print(f"\nQuestion: {question}")
            print(f"\nExcepted Answer: {answer}")
            print(f"\nPredicted Answer: {predicted_number}")
            print("\nCorrect:", "yes" if is_correct else "no")
            print("------------------------------------------------------------\n")
        except Exception as e:
            print(f"Error: {e}")
    
    accuracy = (correct / total) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"================>>> Evaluation completed <<<================")
    return accuracy

def extract_answer_from_model_output(text):
   # Split on <answer> and take everything after the last occurrence
   parts = text.split("<answer>")
   if len(parts) < 2:  # No <answer> tag found
       return None
   last_part = parts[-1]

   # Extract content up to </answer>
   if "</answer>" not in last_part:
       return None
   answer = last_part.split("</answer>")[0].strip()
   return None if answer == "..." else answer

def extract_single_number(text):
   numbers = re.findall(r'-?\d*\.?\d+', text)
   return float(numbers[0]) if len(numbers) == 1 else None

def extract_last_number(text):
    # text = text.replace('$', '').replace('%', '')
    # pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
    # match = re.search(pattern, text)
    # return float(match.group(1)) if match else None
    numbers = re.findall(r'\d+(?:\.\d+)?', text)  # Matches integers and decimals
    return numbers[-1] if numbers else None

def main():
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    set_seed()

    # prepare data
    mydataset = MyDataset("openai/gsm8k")
    data = mydataset.prepare_data()
    print(f"Total data: {len(data)}")
    # print(f"First example: {data[0]}")
    # for i in range(5):
    #     print(f"Example {i+1}: {data[i]}")
    # exit(0)

    random.shuffle(data)
    print(f"shuffled data: {len(data)}")
    size_eval_data = 30
    train_data = data[size_eval_data:]
    eval_data = data[:size_eval_data]

    print(f"Train data: {len(train_data)}")
    print(f"Eval data: {len(eval_data)}")

    # load model
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir = "math_solver_model"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = "left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")
    device_ids = list(range(num_gpus)) if num_gpus > 1 else None

    # print("\nInitial model evaluation before finetuning:")
    # pre_grpo_accuracy = evaluate(model, tokenizer, eval_data, default_device)
    # print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")
    # exit(0)

    print("\nStarting RL fine-tuning using GRPO...")
    # This config was tested on a 4xA6000 48G
    training_config = {
        'num_iterations': 1,
        'num_steps': 500,
        'batch_size': 4, # reduce if you have fewer GPUs
        'num_generations': 4, # reduce if you have GPUs with less VRAM
        'max_completion_length': 400, # reduce if you have GPUs with less VRAM
        'beta': 0.04,
        'learning_rate': 5e-6,
        'mu': 1,
        'epsilon': 0.1,
        'checkpoint_path': "./grpo_checkpoint",
        'checkpoint_interval': 100,
    }
    # Initialize Weights & Biases
    os.environ["WANDB_API_KEY"] = "10f4fe980c70d8a9936f254f5e1c37d6a214fa99"
    os.environ["WANDB_PROJECT"] = "GRPO-Qwen-1.5-Instruct-Multi-GPU"
    wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True, config=training_config)
    print("Weights & Biases initialized.")

    trainer = MyGRPOTrainer(model, tokenizer, [0,1,2,3], default_device)
    GRPOmodel, GRPOtokenizer = trainer.train_with_GRPO(train_data, eval_data, **training_config)

    wandb.finish()
    print("Training completed and wandb run finished.")

    print("\nFinal model evaluation after GRPO RL fine-tuning:")
    post_grpo_accuracy = evaluate(GRPOmodel, tokenizer, eval_data, default_device)
    print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")

    print("\nSaving GRPO fine-tuned model...")
    GRPOmodel.save_pretrained("grpo_finetuned_model")
    GRPOtokenizer.save_pretrained("grpo_finetuned_model")

if __name__ == "__main__":
    main()
