# Whisper Domain Fine-Tuning to whisper.cpp

도메인 데이터로 Whisper base model을 LoRA fine-tuning한 뒤, LoRA를 merge하고 whisper.cpp용 GGML 모델과 quantized 모델로 변환하는 파이프라인이다.

구조는 아래 기준으로 나눈다.

```text
config/  설정 파일
data/    raw corpus, split cache, 학습/검증/eval manifest
scripts/ 사용자가 직접 실행하는 스크립트
src/     내부 구현 코드
exp/     학습, merge, 변환, 디코딩 산출물
```

명령어는 repository root에서 실행한다.

### 환경 준비

Python 3.10 기준 conda 환경을 만든다.

```bash
conda create -n whisper-domain-finetuning python=3.10 -y
conda activate whisper-domain-finetuning
pip install -r requirements.txt
```

CUDA 버전이 다르면 `requirements.txt`의 PyTorch/torchaudio wheel을 현재 서버 환경에 맞게 바꾼 뒤 설치한다.

현재 로컬 작업 디렉토리에서는 `data/download`가 기존 benchmark repository의 `data/download`를 가리키는 symlink다. 새 환경에서는 `data/download` 아래에 raw archive를 두고, 압축 해제 결과는 `data/download/extracted`, 긴 음성 split cache는 `data/download/splits` 아래에 생성한다.

### 데이터 준비

먼저 raw corpus에서 train/dev/eval manifest를 만든다.

```bash
python scripts/prepare_data.py \
  --data_root ./data/download \
  --output_root ./data/whisper_small_lora \
  --sample_rate 16000 \
  --seed 42
```

기존 구조에서 생성한 manifest는 오디오 경로 기준이 다를 수 있으므로, 디렉토리 재정리 후에는 데이터를 한 번 다시 생성하는 것을 권장한다.

생성이 끝나면 `train.jsonl`, `dev.jsonl`, `eval.jsonl` 구조와 오디오 경로를 자동으로 검증한다. 기본 split은 train/dev/eval = 8:1:1이다. `dev.jsonl`은 학습 중 validation에 쓰고, `eval.jsonl`은 학습 완료 후 최종 평가에 쓴다.

학습용 jsonl은 학습 후 디코딩/평가까지 그대로 쓰기 위해 아래 7개 필드로 고정한다.

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

새 데이터에 다른 metadata가 있더라도 기본 manifest에는 위 필드만 저장한다.

### 학습 실행

설정은 `config/whisper_small_lora.yaml`에서 관리한다.

```bash
python scripts/train_whisper_lora.py
```

출력 위치를 명시하려면 아래처럼 실행한다.

```bash
python scripts/train_whisper_lora.py \
  --run_name whisper_small_lora_epoch3_v2 \
  --output_root ./exp/train
```

출력은 `exp/train/<train_name>/` 아래에 저장된다.

```text
exp/train/<train_name>/
  run_config.json
  checkpoints/
  epochs/
  adapter/
  processor/
  training_summary.json
```

### 주요 설정

- `base_model_name_or_path`: 기본값 `openai/whisper-small`
- `language`: 기본값 `Korean`
- `lora.r`, `lora.alpha`, `lora.target_modules`: LoRA 설정
- `training.per_device_train_batch_size`: GPU 1장 기준 batch size
- `training.gradient_accumulation_steps`: effective batch size 조절
- `training.fp16`: fp16 학습 사용 여부
- `training.gradient_checkpointing`: 기본값 `false`. Whisper small LoRA 1차 학습은 안정성을 우선해서 끈다.
- validation과 모델 저장은 epoch 단위로 실행한다.
- `epochs/epoch_001`, `epochs/epoch_002`처럼 epoch 기준 경로가 생성된다.
- 학습이 끝난 뒤 `adapter/`에는 dev loss 기준 best epoch 모델이 저장된다.

학습 재개가 필요하면 저장된 epoch checkpoint를 지정한다.

```bash
python scripts/train_whisper_lora.py \
  --resume_from_checkpoint ./exp/train/<train_name>/epochs/epoch_002
```

### LoRA merge

학습이 끝나면 best epoch LoRA adapter를 base model에 merge해서 일반 Hugging Face Whisper checkpoint 형태로 저장한다.

```bash
python scripts/merge_whisper_lora.py \
  --train_dir ./exp/train/<train_name> \
  --output_dir ./exp/merged/<train_name>
```

`--checkpoint_path`를 직접 지정하지 않으면 trainer state의 best epoch checkpoint를 먼저 사용한다. best epoch 정보를 찾을 수 없는 경우에만 `adapter/`, 최신 epoch 저장본 순서로 내려간다.

출력은 기본적으로 아래에 저장된다.

```text
exp/merged/<train_name>/
  config.json
  generation_config.json
  model.safetensors
  preprocessor_config.json
  tokenizer.json
  merge_summary.json
```

특정 epoch 저장본을 바로 merge하고 싶으면 경로를 직접 지정한다.

```bash
python scripts/merge_whisper_lora.py \
  --train_dir ./exp/train/<train_name> \
  --checkpoint_path ./exp/train/<train_name>/epochs/epoch_002 \
  --output_dir ./exp/merged/<train_name>_epoch_002
```

이미 같은 출력 디렉토리가 있으면 기본적으로 중단된다. 덮어쓰려면 `--overwrite`를 붙인다.

### Merge 모델 디코딩

Merge된 모델은 fine-tuning eval manifest로 바로 디코딩할 수 있다.

```bash
python scripts/run_merged_whisper_decoding.py \
  --model_dir ./exp/merged/<train_name> \
  --manifest_path ./data/whisper_small_lora/eval.jsonl \
  --output_dir ./exp/results/<train_name>_beam1_float16 \
  --device cuda:0 \
  --beam_size 1 \
  --precision float16
```

출력은 기존 benchmark 결과와 같은 구조다.

```text
exp/results/<train_name>_beam1_float16/
  predictions.jsonl
  errors.jsonl
  metrics.json
  run_config.json
  logs/run.log
```

`metrics.json`에는 전체 CER/WER뿐 아니라 dataset별, bucket별 결과도 함께 저장된다.

### whisper.cpp 변환

이 repository 안에 whisper.cpp를 준비한다.

```bash
git submodule update --init --recursive
```

submodule이 아직 등록되지 않은 로컬 개발 상태라면 아래 위치에 whisper.cpp를 clone하거나 submodule로 추가한다.

```bash
git submodule add https://github.com/ggml-org/whisper.cpp.git third_party/whisper.cpp
git submodule update --init --recursive
```

whisper.cpp는 CUDA 사용 여부에 맞춰 먼저 build해야 한다.

Merge된 Hugging Face 모델은 whisper.cpp용 `ggml` 모델로 변환할 수 있다. 변환 스크립트는 f16 ggml 모델과 `q8_0`, `q5_0` quantized 모델을 함께 만든다.

```bash
python scripts/convert_merged_lora_whisper_model_to_whisper_cpp.py \
  --model_dir ./exp/merged/<train_name> \
  --output_dir ./exp/converted_model/<train_name>
```

출력:

```text
exp/converted_model/<train_name>/
  ggml-model.bin
  ggml-model-q8_0.bin
  ggml-model-q5_0.bin
  conversion_summary.json
  convert.log
```

이미 생성된 파일은 기본적으로 재사용한다. 다시 만들려면 `--overwrite`를 붙인다.

변환 후에는 짧은 음성 하나로 whisper.cpp가 모델을 읽는지 확인한다.

```bash
LD_LIBRARY_PATH=third_party/whisper.cpp/build/src:third_party/whisper.cpp/build/ggml/src:third_party/whisper.cpp/build/ggml/src/ggml-cuda \
third_party/whisper.cpp/build/bin/whisper-cli \
  -m ./exp/converted_model/<train_name>/ggml-model-q5_0.bin \
  -f <test_audio.wav> \
  -l ko
```

`ggml-model.bin`, `ggml-model-q8_0.bin`, `ggml-model-q5_0.bin` 중 어떤 모델을 쓸지는 사용자가 직접 짧은 샘플과 eval set으로 확인한 뒤 선택한다. 보통은 정확도와 속도를 함께 보고 하나를 고른다.

### 다음 단계

변환 모델이 정상 로딩되면 같은 `eval.jsonl` 기준으로 whisper.cpp server 디코딩 결과를 만든다.

```bash
python scripts/run_whisper_cpp_server_decoding.py \
  --model_path ./exp/converted_model/<train_name>/ggml-model-q5_0.bin \
  --manifest_path ./data/whisper_small_lora/eval.jsonl \
  --result_dir ./exp/results/<train_name>_whisper_cpp_q5_0_beam1 \
  --server_binary_path ./third_party/whisper.cpp/build/bin/whisper-server \
  --device cuda \
  --device_index 0 \
  --port 8100 \
  --beam_size 1
```

출력:

```text
exp/results/<train_name>_whisper_cpp_q5_0_beam1/
  predictions.jsonl
  errors.jsonl
  metrics.json
  run_config.json
  logs/
```

이 결과를 merge된 Hugging Face 모델 결과와 비교한다.

```text
exp/results/<train_name>_beam1_float16/
exp/results/<train_name>_whisper_cpp_q5_0_beam1/
```

필요하면 metrics만 다시 생성한다.

```bash
python scripts/evaluate_predictions.py \
  --manifest_path ./data/whisper_small_lora/eval.jsonl \
  --result_dir ./exp/results/<train_name>_whisper_cpp_q5_0_beam1
```

`ggml-model.bin`, `ggml-model-q8_0.bin`, `ggml-model-q5_0.bin` 중 어느 모델을 쓸지는 정확도와 속도를 확인해서 선택한다. 선택한 모델은 이후 실제 서비스나 데모 서버의 whisper.cpp model path에 연결한다.
