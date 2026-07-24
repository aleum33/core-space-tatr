import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import gc

# ==========================================
# 1. 하이퍼파라미터 및 환경 설정
# ==========================================
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASETS_TO_TRAIN = ["snli", "mnli"]

MAX_SEQ_LEN = 128
BATCH_SIZE = 8
GRAD_ACCUMULATION = 4
LEARNING_RATE = 2e-4
EPOCHS = 3
EMA_DECAY = 0.999

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


# ==========================================
# 2. EMA 클래스 정의
# ==========================================
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


# ==========================================
# 3. 데이터셋별 자동 반복 루프 시작
# ==========================================
DATASET_CONFIGS = {
    "snli": {"path": "snli", "name": None, "p_col": "premise", "h_col": "hypothesis"},
    "mnli": {"path": "glue", "name": "mnli", "p_col": "premise", "h_col": "hypothesis"}
}

for dataset_name in DATASETS_TO_TRAIN:
    print(f"\n{'=' * 60}\n🚀 [학습 시작] {dataset_name.upper()} 데이터셋 전면 재학습 (MLP 포함)\n{'=' * 60}")

    # 3.1 매번 깨끗한 모델과 LoRA 세팅 (독립성 보장)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16
    )


    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    ema = EMA(model, EMA_DECAY)

    # 3.2 데이터 전처리 (스마트 맵핑)
    config = DATASET_CONFIGS[dataset_name]
    if config["name"]:
        dataset = load_dataset(config["path"], config["name"], split="train")
    else:
        dataset = load_dataset(config["path"], split="train")


    def tokenize_function(examples):
        texts = []
        p_col = config["p_col"]
        h_col = config["h_col"]

        for p, h, l in zip(examples[p_col], examples[h_col], examples['label']):
            # 라벨이 문자로 되어있는 경우 방어
            if isinstance(l, int):
                ans = LABEL_MAP.get(l, "neutral")
            else:
                ans = str(l).lower()
            texts.append(f"Premise: {p}\nHypothesis: {h}\nAnswer: {ans}{tokenizer.eos_token}")

        return tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_SEQ_LEN)


    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    tokenized_dataset.set_format("torch")

    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3.3 옵티마이저 및 스케줄러 세팅
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    model.train()

    # 3.4 대망의 학습 루프
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"{dataset_name.upper()} - Epoch {epoch}")
        for step, batch in enumerate(pbar):

            inputs = {k: v.to(model.device) for k, v in batch.items()}

            labels = inputs['input_ids'].clone()
            labels[inputs['attention_mask'] == 0] = -100
            inputs['labels'] = labels

            outputs = model(**inputs)
            loss = outputs.loss / GRAD_ACCUMULATION
            loss.backward()

            if (step + 1) % GRAD_ACCUMULATION == 0:
                optimizer.step()
                scheduler.step()
                ema.update()
                optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item() * GRAD_ACCUMULATION)

    # 3.5 모델 저장 및 캐시 청소
    ema.apply_shadow()
    output_dir = f"./output_{dataset_name}_full_mlp"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ {dataset_name.upper()} 저장 완료: {output_dir}")
    del model, optimizer, scheduler, dataloader, dataset, ema
    gc.collect()
    torch.cuda.empty_cache()

print("🎉 24GB 서버의 모든 학습 임무가 종료되었습니다!")