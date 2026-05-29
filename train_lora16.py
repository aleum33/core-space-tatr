import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import gc
import warnings
import torch
from pathlib import Path
from tqdm import tqdm
import datasets
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore")

task_to_keys = {
    "sick": ("sentence_A", "sentence_B"),
    "qnli": ("text1", "text2"),
    "rte": ("text1", "text2"),
    "scitail": ("premise", "hypothesis"),
}

task_ids = {
    "sick": "sick",
    "qnli": "SetFit/qnli",
    "rte": "SetFit/rte",
    "scitail": "allenai/scitail",
}

task_masks = {
    "sick": None,
    "qnli": 1,
    "rte": 1,
    "scitail": 2,
}

MODEL_NAME_OR_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASETS_TO_TRAIN = ["sick", "qnli", "rte", "scitail"]
MAX_NUM_EPOCHS = 3
LR = 3e-5

BATCH_SIZE = 4
GRAD_ACCUMULATION = 8

NUM_WORKERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, padding_side="right")
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


for TASK in DATASETS_TO_TRAIN:
    print(f"\n{'=' * 60}\n🚀 [RTX 3090 초고속 bf16] {TASK.upper()} (Attention + MLP) 시작!\n{'=' * 60}")

    # 4.1 데이터셋 다운로드 및 오리지널 전처리 복제
    if TASK == "scitail":
        dataset = datasets.load_dataset(task_ids[TASK], 'tsv_format', trust_remote_code=True)
    else:
        dataset = datasets.load_dataset(task_ids[TASK], trust_remote_code=True)

    sentence1_key, sentence2_key = task_to_keys[TASK]


    def tokenize_function(examples):
        args = (examples[sentence1_key], examples[sentence2_key])
        outputs = tokenizer(*args, truncation=True, max_length=1000 if TASK == "scitail" else 2000)

        # SciTail 문자열 및 기타 라벨 정제 시스템
        mapped_labels = []
        for l in examples["label"]:
            if isinstance(l, str):
                if l.lower() in ['entailment', 'entails']:
                    mapped_labels.append(0)
                elif l.lower() in ['neutral']:
                    mapped_labels.append(1)
                else:
                    mapped_labels.append(2)
            else:
                mapped_labels.append(l)
        outputs["labels"] = mapped_labels
        return outputs


    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset['train'].column_names)
    tokenized_datasets = tokenized_datasets.filter(lambda x: x["labels"] in [0, 1, 2])

    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn,
                                  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    val_key = "validation" if "validation" in tokenized_datasets else "test"
    val_dataloader = DataLoader(tokenized_datasets[val_key], shuffle=False, collate_fn=collate_fn,
                                batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # 매 에포크가 끝날 때마다 정확하게 검증하도록 평가 주기 계산
    EVAL_AFTER_STEPS = len(train_dataloader) // GRAD_ACCUMULATION

    # 4.2 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME_OR_PATH, return_dict=True, num_labels=3, torch_dtype=torch.bfloat16, device_map={"": 0}
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model.gradient_checkpointing_enable()

    # LoRA 아키텍처 구성 (Attention 전체 + MLP 전체 통합 타겟 지정)
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())

    # 4.3 옵티마이저 및 스케줄러 세팅
    mask_class = task_masks[TASK]
    criterion = CrossEntropyLoss()
    optimizer = AdamW(params=model.parameters(), lr=LR)

    total_training_steps = (len(train_dataloader) * MAX_NUM_EPOCHS) // GRAD_ACCUMULATION
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.06 * total_training_steps),
        num_training_steps=total_training_steps,
    )

    total_steps = 0
    max_acc = 0
    best_state_dict = None

    # 4.4 오리지널 저자들의 정석 트레인 루프 시스템 가동
    for epoch in range(MAX_NUM_EPOCHS):
        model.train()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(pbar):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(DEVICE)

            outputs = model(**inputs)
            logits = outputs.logits

            # 2지선다형 태스크 저자 특수 마스킹 트릭 복제
            if mask_class is not None:
                logits[:, mask_class] = -1e10

            loss = criterion(logits, labels) / GRAD_ACCUMULATION
            loss.backward()

            if (step + 1) % GRAD_ACCUMULATION == 0:
                total_steps += 1
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # 에포크 단위 정기 검증 주기 돌입
                if total_steps % EVAL_AFTER_STEPS == 0:
                    model.eval()
                    correct, total = 0, 0
                    with torch.no_grad():
                        for v_batch in val_dataloader:
                            v_inputs = {k: v.to(DEVICE) for k, v in v_batch.items() if k != 'labels'}
                            v_outputs = model(**v_inputs)
                            v_logits = v_outputs.logits
                            if mask_class is not None:
                                v_logits[:, mask_class] = -1e10
                            preds = v_logits.argmax(dim=-1)
                            correct += (preds == v_batch['labels'].to(DEVICE)).sum().item()
                            total += v_batch['labels'].size(0)

                    acc = correct / total
                    print(f"\n 📊 [Epoch {epoch + 1} / Step {total_steps}] Validation 정확도 : {acc:.3%}")

                    # 최고 성능 시점 캐치 및 백업
                    if acc > max_acc:
                        max_acc = acc
                        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    model.train()

            pbar.set_postfix(loss=loss.item() * GRAD_ACCUMULATION)

    # 4.5 학습 종료 후 가장 완벽했던 가중치를 복원하여 로컬 서버에 최종 세이브
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    save_path = f"./output_{TASK}_final"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ {TASK.upper()} 3 에포크 완료! 최고 체크포인트 저장 성공! (Best Val: {max_acc:.3%})")

    # 메모리 완전 세척 후 다음 태스크로 이동
    del model, optimizer, lr_scheduler, train_dataloader, val_dataloader, best_state_dict
    gc.collect()
    torch.cuda.empty_cache()

print("🎉 서버용 4개 데이터셋의 학습 임무가 100% 안전하게 종료되었습니다!")