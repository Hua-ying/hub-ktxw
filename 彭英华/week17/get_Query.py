import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
from peft import LoraConfig,TaskType,get_peft_model
class Makedfdataset():
    def __init__(self,movies_path,ratings_path):
      
        #读取所有电影
        self.movies_data = pd.read_csv(movies_path,sep="::",header=None,engine='python',encoding="gbk")
        self.movies_data.columns = ['movie_id','movie_title','movie_type']
        self.movie_titles = self.movies_data['movie_title'].tolist()
        self.movie_ids = self.movies_data['movie_id'].tolist()
      
        # 电影名称与索引对应
        self.idx2movie = {}
        for id,movie_title in zip(self.movie_ids,self.movie_titles):
            self.idx2movie[id] = movie_title

        # 获取评分数据
        self.ratings_data = pd.read_csv(ratings_path,sep="::",header=None,engine="python")
        self.ratings_data.columns = ["user_id", "movie_id", "rating", "timestamp"]
        self.ratings_data = self.ratings_data[self.ratings_data['rating']>=3].reset_index(drop=True)
        self.ratings_data = self.ratings_data.sort_values(['user_id','timestamp'])

        # 构建交互序列
        self.user_movie = {}
        for user,group in self.ratings_data.groupby('user_id'):
            items = set(group['movie_id'].tolist())
            if (len(items)>=15):
                items = list(items)[:15]
            items = list(items)
            self.user_movie[user] = items
        self.user_ids = list(self.user_movie.keys())
    
    # 划分训练集，验证集和测试集
    def getdataset(self):
        train_users, test_users = train_test_split(
            self.user_ids,
            test_size=0.2,
            random_state=42
        )
        valid_users, test_users = train_test_split(
            test_users,
            test_size=0.5,
            random_state=42
        )
        train_df = self.get_df(train_users)
        valid_df = self.get_df(valid_users)
        test_df = self.get_df(test_users)
        return train_df,valid_df,test_df

    # 将数据集用dataframe存储
    def get_df(self,users):
        user_movie_ids = [self.user_movie[user] for user in users]
        movie_titles = []
        for seq in user_movie_ids:
            movies =[]
            for id in seq:
                movie_title = self.idx2movie[id]
                movies.append(movie_title)
            movie_titles.append(movies)
        prompts =[]
        for seq in movie_titles:
            prompt = "之前，用户已经看过这些电影：" + '\n'
            item_titles = "，".join(seq[:-1])
            prompt = prompt+item_titles +'\n未来，用户可能去看这部电影：'
            prompts.append({'prompt':prompt,'output':seq[-1]})
        df = pd.DataFrame(prompts)
        return df
      
def load_and_preprocess_data(train_data):
    """加载和预处理数据"""

    # 重命名列并添加输入列
    train_data.columns = ["instruction", "output"]

    # 转换为Hugging Face Dataset
    ds = Dataset.from_pandas(train_data)

    return ds
  
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16  # 使用半精度减少内存占用
    )

    return tokenizer, model
  
def process_func(example, tokenizer, max_length=384):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建指令部分
    instruction_text = f"<|im_start|>system\n现在开始进行电影推荐，根据用户已经观看过的电影，预测用户未来可能观看的一部电影名称，只输出一部电影的英文名称，不要输出其他内容。<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    # 组合输入ID和注意力掩码
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    # 构建标签（指令部分用-100忽略，只计算响应部分的损失）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

    # 截断超过最大长度的序列
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
  
def setup_lora(model):
    """设置LoRA配置并应用到模型"""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model
  
def setup_training_args():
    """设置训练参数"""
    return TrainingArguments(
        output_dir="./output_Qwen",
        per_device_train_batch_size=6,
        gradient_accumulation_steps=4,
        logging_steps=100,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=5,
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none"  # 禁用wandb等报告工具
    )
# 进行批量预测或单样本预测
def predict(prompts,model,tokenizer,device,beam_size):
    if isinstance(prompts,str):
        prompts = [prompts]
    predictions = []
    for prompt in prompts:
        formatted_text = f"<|im_start|>system\n现在开始进行电影推荐，根据用户已经观看过的电影，预测用户未来可能观看的一部电影名称，只输出一部电影的英文名称，不要输出其他内容。<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        # print(formatted_text)
        model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

       #生成结果
        with torch.no_grad():
            outputs = model.generate(
                model_inputs.input_ids,
                max_new_tokens = 20,
                num_beams = beam_size,
                num_return_sequences = 5,
                do_sample = False,
                early_stopping=True,  # 所有束都遇到EOS则停止，节省计算
                no_repeat_ngram_size=0,  # 避免在单个查询内出现重复n-gram
                length_penalty=1.0,  # 稍鼓励短查询（值<1.0），推荐场景常设
                repetition_penalty=1.2,  # 全局重复惩罚，避免不同查询间过度相似
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.1
            )
            queries = []
            input_length = model_inputs.input_ids.shape[1]   #原始输入长度
            for seq in outputs:
                query = tokenizer.decode(seq[input_length:],skip_special_tokens=True).strip()  #获取模型输出
                queries.append(query)
        predictions.append(queries)
    return predictions
if __name__ == "__main__":
    train_df,valid_df,test_df = Makedfdataset("./M_ML-100K/movies.dat",'./M_ML-100K/ratings.dat').getdataset()
    model_path = "./models/Qwen/Qwen3-0.6B"
    tokenizer,model = initialize_model_and_tokenizer(model_path)
    train_ds = load_and_preprocess_data(train_df)
    valid_ds = load_and_preprocess_data(valid_df)
    process_func_with_tokenizer = lambda example: process_func(example,tokenizer)
    train_tokenized = train_ds.map(process_func_with_tokenizer,remove_columns=train_ds.column_names)
    valid_tokenized = valid_ds.map(process_func_with_tokenizer,remove_columns=valid_ds.column_names)
    print(tokenizer.decode(train_tokenized[1]['input_ids']))
    print("设置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)
    print("配置训练参数...")
    training_args = setup_training_args()
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
        data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding=True,
                pad_to_multiple_of=8  # 优化GPU内存使用
            ),
    )
    trainer.train()
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./output_Qwen")
    print("测试")
    predictions = predict(train_df.iloc[0]['instruction'], model, tokenizer, device="cuda", beam_size=5)
    print(predictions)

