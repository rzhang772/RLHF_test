from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from mydataset import MyDataset
import re
import os
import warnings
import absl.logging



def extract_last_number(text):
    # text = text.replace('$', '').replace('%', '')
    # pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
    # match = re.search(pattern, text)
    # return float(match.group(1)) if match else None
    numbers = re.findall(r'\d+(?:\.\d+)?', text)  # Matches integers and decimals
    return numbers[-1] if numbers else None

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

def main():
    warnings.filterwarnings("ignore")
    absl.logging.set_verbosity(absl.logging.ERROR)
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model and tokenizer
    model_path = "grpo_checkpoint/checkpoint-0-99"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    mydataset = MyDataset("openai/gsm8k")
    data = mydataset.prepare_data()
    print(f"Total data: {len(data)}")

    evaluate(model, tokenizer, data, device)


if main() == "__main__":
    main()
