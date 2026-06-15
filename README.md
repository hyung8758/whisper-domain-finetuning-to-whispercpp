# Whisper Domain Fine-Tuning to whisper.cpp

도메인 데이터로 Whisper를 LoRA fine-tuning하고, merge한 모델을 whisper.cpp용 GGML/quantized 모델로 변환하는 파이프라인이다.

## 1. 바로 실행하는 명령어

### 1.1 환경 준비

```bash
conda create -n whisper-domain-finetuning python=3.10 -y
conda activate whisper-domain-finetuning
pip install -r requirements.txt
```

### 1.2 데이터 준비

`data/download` 아래에 raw corpus를 준비한 뒤 실행한다.

```bash
python scripts/prepare_data.py \
  --data_root ./data/download \
  --output_root ./data/whisper_small_lora
```

### 1.3 LoRA 학습

```bash
python scripts/train_whisper_lora.py \
  --run_name whisper_small_lora_epoch3_v2
```

### 1.4 LoRA merge

```bash
python scripts/merge_whisper_lora.py \
  --train_dir ./exp/train/whisper_small_lora_epoch3_v2 \
  --output_dir ./exp/merged/whisper_small_lora_epoch3_v2
```

### 1.5 Merge 모델 디코딩 평가

```bash
python scripts/run_merged_whisper_decoding.py \
  --model_dir ./exp/merged/whisper_small_lora_epoch3_v2 \
  --manifest_path ./data/whisper_small_lora/eval.jsonl \
  --output_dir ./exp/results/whisper_small_lora_epoch3_v2_beam1_float16 \
  --device cuda:0 \
  --beam_size 1 \
  --precision float16
```

### 1.6 whisper.cpp 모델 변환 및 quantization

- **1.6.1 whisper.cpp 준비**

  이미 submodule을 받아둔 상태라면 이 단계는 스킵한다.

  ```bash
  git submodule update --init --recursive
  ```

- **1.6.2 whisper.cpp 빌드**

  이미 `third_party/whisper.cpp/build/bin/whisper-quantize`가 있으면 이 단계는 스킵한다. 기본은 CUDA build다.

  ```bash
  cmake -S third_party/whisper.cpp \
    -B third_party/whisper.cpp/build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release

  cmake --build third_party/whisper.cpp/build \
    --config Release \
    -j "$(nproc)"
  ```

  빌드 확인:

  ```bash
  ls third_party/whisper.cpp/build/bin/whisper-quantize
  ```

- **1.6.3 모델 변환 및 quantization**

  ```bash
  python scripts/convert_merged_lora_whisper_model_to_whisper_cpp.py \
    --model_dir ./exp/merged/whisper_small_lora_epoch3_v2 \
    --output_dir ./exp/converted_model/whisper_small_lora_epoch3_v2
  ```

### 1.7 whisper.cpp 로딩 확인

```bash
LD_LIBRARY_PATH=third_party/whisper.cpp/build/src:third_party/whisper.cpp/build/ggml/src:third_party/whisper.cpp/build/ggml/src/ggml-cuda \
third_party/whisper.cpp/build/bin/whisper-cli \
  -m ./exp/converted_model/whisper_small_lora_epoch3_v2/ggml-model-q5_0.bin \
  -f data/sample/asr_kcsc_001.wav \
  -l ko
```

### 1.8 whisper.cpp 디코딩 평가

변환한 ggml 모델로 `data/whisper_small_lora/eval.jsonl`을 디코딩하고 `metrics.json`을 만든다.

```bash
python scripts/run_whisper_cpp_decoding.py \
  --model_path ./exp/converted_model/whisper_small_lora_epoch3_v2/ggml-model-q5_0.bin \
  --manifest_path ./data/whisper_small_lora/eval.jsonl \
  --output_dir ./exp/results/whisper_small_lora_epoch3_v2_whisper_cpp_q5_0 \
  --language ko \
  --beam_size 1
```

결과는 아래 경로에 저장된다.

```text
exp/results/whisper_small_lora_epoch3_v2_whisper_cpp_q5_0/
  predictions.jsonl
  errors.jsonl
  metrics.json
  run_config.json
  logs/run.log
```

## 2. 추가 사용법

### CPU-only whisper.cpp 빌드

CUDA 없이 CPU만 사용할 경우에는 `-DGGML_CUDA=ON`을 빼고 build한다.

```bash
cmake -S third_party/whisper.cpp \
  -B third_party/whisper.cpp/build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build third_party/whisper.cpp/build \
  --config Release \
  -j "$(nproc)"
```

### 데이터 검증만 실행

```bash
python scripts/validate_data.py \
  --output_root ./data/whisper_small_lora
```

### 학습 config 변경

기본 학습 설정은 아래 파일에서 관리한다.

```text
config/whisper_small_lora.yaml
```

주로 바꾸는 값:

```text
base_model_name_or_path
lora.r
lora.alpha
training.epoch
training.learning_rate
training.per_device_train_batch_size
training.gradient_accumulation_steps
training.fp16
```

### 학습 재개

```bash
python scripts/train_whisper_lora.py \
  --run_name whisper_small_lora_epoch3_v2_resume \
  --resume_from_checkpoint ./exp/train/whisper_small_lora_epoch3_v2/epochs/epoch_002
```

### 특정 checkpoint merge

```bash
python scripts/merge_whisper_lora.py \
  --train_dir ./exp/train/whisper_small_lora_epoch3_v2 \
  --checkpoint_path ./exp/train/whisper_small_lora_epoch3_v2/epochs/epoch_002 \
  --output_dir ./exp/merged/whisper_small_lora_epoch3_v2_epoch_002
```

### 기존 출력 덮어쓰기

이미 같은 출력 디렉토리가 있을 때는 `--overwrite`를 붙인다.

```bash
python scripts/merge_whisper_lora.py \
  --train_dir ./exp/train/whisper_small_lora_epoch3_v2 \
  --output_dir ./exp/merged/whisper_small_lora_epoch3_v2 \
  --overwrite
```

```bash
python scripts/convert_merged_lora_whisper_model_to_whisper_cpp.py \
  --model_dir ./exp/merged/whisper_small_lora_epoch3_v2 \
  --output_dir ./exp/converted_model/whisper_small_lora_epoch3_v2 \
  --overwrite
```

### 변환 quantization 종류 변경

```bash
python scripts/convert_merged_lora_whisper_model_to_whisper_cpp.py \
  --model_dir ./exp/merged/whisper_small_lora_epoch3_v2 \
  --output_dir ./exp/converted_model/whisper_small_lora_epoch3_v2 \
  --quantizations q8_0 q5_0
```

## 3. 참고 정보

### 디렉토리 구조

```text
config/       설정 파일
data/         raw corpus, split cache, train/dev/eval manifest
scripts/      실행 스크립트
src/          내부 구현 코드
exp/          학습, merge, 변환, 디코딩 산출물
third_party/  whisper.cpp submodule
```

### 데이터 manifest 형식

학습/검증/eval JSONL은 아래 필드를 사용한다.

```json
{
  "id": "sample_id",
  "audio": "/absolute/path/to/audio.wav",
  "text": "정규화된 정답 문장",
  "duration": 3.21,
  "dataset": "zeroth",
  "bucket": "short",
  "split": "train"
}
```

### 주요 산출물

```text
data/whisper_small_lora/
  train.jsonl
  dev.jsonl
  eval.jsonl

exp/train/<train_name>/
  run_config.json
  checkpoints/
  epochs/
  adapter/
  processor/
  training_summary.json

exp/merged/<train_name>/
  config.json
  generation_config.json
  model.safetensors
  tokenizer.json
  merge_summary.json

exp/converted_model/<train_name>/
  ggml-model.bin
  ggml-model-q8_0.bin
  ggml-model-q5_0.bin
  conversion_summary.json
  convert.log

exp/results/<result_name>/
  predictions.jsonl
  errors.jsonl
  metrics.json
  run_config.json
  logs/run.log
```

### 결과 확인

디코딩 평가 결과는 아래 파일을 본다.

```text
exp/results/<result_name>/metrics.json
```

`metrics.json`에는 전체 CER/WER와 dataset별, bucket별 결과가 들어간다.
