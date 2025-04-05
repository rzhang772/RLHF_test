import torch
import os
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed
from trl.core import LengthSampler

# 设置 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# 配置训练参数
config = PPOConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",  # 使用 Qwen2.5 1.5B Instruct 版本
    learning_rate=1.41e-5,
    batch_size=8,  # 根据显存适当调整
    mini_batch_size=4,
    gradient_accumulation_steps=4,
    early_stopping=True,
    target_kl=0.1,
    adap_kl_ctrl=True,
    ppo_epochs=4,
    optimize_device_cache=True,
    remove_unused_columns=False,
    log_with="wandb",
    project_name="ppo-qwen-gsm8k"
)

# **加载模型**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = AutoModelForCausalLM.from_pretrained(
    config.model_name, 
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    use_cache=False  # PPO 训练时禁用缓存
)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# **包装成 PPO 需要的 Value Head**
model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
model = model.to(device)

# **加载数据集 GSM8K**
dataset = load_dataset("gsm8k", "main")

# **定义奖励模型（Reward Function）**
reward_model_name = "OpenAssistant/reward-model-deberta-v3-large"
reward_model = pipeline("text-classification", model=reward_model_name, device=0)

# **定义数据采样器**
input_size = LengthSampler(5, 30)

# **训练函数**
def collate_fn(samples):
    """
    处理 GSM8K 数据，使其符合 LLM 训练格式。
    """
    queries = []
    for sample in samples:
        question = sample["question"]
        query = f"Q: {question}\nA: "
        queries.append(query)
    return queries

# **定义 PPO 训练器**
ppo_trainer = PPOTrainer(config, model, ref_model=None, tokenizer=tokenizer)

# **开始训练**
set_seed(42)  # 设定随机种子
wandb.init(project="ppo-qwen-gsm8k")

for epoch in range(3):  # 训练 3 轮
    for batch in dataset["train"].shuffle(seed=epoch).select(range(2000)):  # 选择部分数据
        queries = collate_fn([batch])
        
        # 让模型生成文本
        input_tensors = tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to(device)
        response_tensors = model.generate(**input_tensors, max_new_tokens=128)
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # 计算奖励
        rewards = []
        for response in responses:
            reward_output = reward_model(response)
            reward_score = reward_output[0]["score"]  # 获取奖励分数
            rewards.append(reward_score)

        # **PPO 更新**
        train_stats = ppo_trainer.step(input_tensors["input_ids"], response_tensors, rewards)
        wandb.log(train_stats)

    print(f"Epoch {epoch + 1} completed!")

# **保存模型**
model.save_pretrained("./ppo_qwen2.5_gsm8k")
tokenizer.save_pretrained("./ppo_qwen2.5_gsm8k")
wandb.finish()