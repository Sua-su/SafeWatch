# 작업 인수인계 (Claude용)

> 새 컴퓨터에서 이 저장소를 받은 뒤 Claude에게 "HANDOFF.md 읽고 이어서 작업해줘"라고 하면 된다.
> 작업이 끝나면 이 파일의 "지금까지 한 것"과 "해야 할 것"을 최신 상태로 고쳐서 같이 커밋할 것.

## 프로젝트
- 라즈베리파이 + 웹캠 + YOLO로 만드는 안전관리 워쳐 프로그램 (GitHub: Sua-su/SafeWatch, 브랜치 main).
- 감지 두 가지:
  1. **사람 안전**: 안전모·안전조끼 착용 여부
     - 모델: `ppe_n_ncnn_model/` (또는 `ppe_n.pt`)
     - 클래스: Person / Hardhat / Safety Vest / NO-Hardhat / NO-Safety Vest
  2. **공장 안전**: 화재·연기 (라이터 불 정도까지)
     - 모델: `fire_n_ncnn_model/` (또는 `fire_n.pt`)
     - 클래스: 0=smoke, 1=fire
- 위반 시 GPIO 17 릴레이(`pinctrl`)로 경고등 ON, `violations/`에 사진과 `log.csv` 기록.
- **실제 실행 환경은 라즈베리파이(64bit Raspberry Pi OS)**. 개발 PC는 코드 수정·모델 학습용.
- `requirements.txt`는 라즈베리파이용. `requirements-mac.txt`는 이전 Intel Mac 전용(torch 2.2.2 고정)이니 다른 컴퓨터에서는 쓰지 말 것.
- 모델 가중치(`.pt`, `*_ncnn_model/`)와 `runs/`는 git에 올리는 방침 (.gitignore에서 일부러 뺐음).

## 파일
| 파일 | 내용 |
|---|---|
| `ppl.py` | 메인 프로그램. PPE + 화재 감지, 경보 2개(PPE 5프레임 ON/15프레임 OFF, 화재 3프레임 ON/30프레임 OFF), 릴레이, 기록 |
| `training/train_fire.py` | Kaggle D-Fire 데이터셋을 kagglehub로 받아 `yolo26n.pt`로 학습 → 저장소 루트에 `fire_n.pt` + `fire_n_ncnn_model/` 생성 |
| `training/train_fire_colab.ipynb` | GPU가 없을 때 Google Colab에서 학습하는 노트북 |

## 지금까지 한 것
- `ppl.py`에 화재/연기 감지 추가 완료. 화재 모델이 없으면 PPE 감지만 동작하고, `pinctrl`이 없는 PC에서는 릴레이 제어만 건너뜀.
- 데이터셋: D-Fire (`sayedgamal99/smoke-fire-detection-yolo`, 2.84GB, CC0). train 14,122 / val 3,099 / test 4,306장. Kaggle 로그인 없이 받아짐.
- 이전 Intel Mac(i7-7700HQ, GPU 없음)에서 빠른 테스트 실행:
  `python training/train_fire.py --fraction 0.01 --epochs 1 --imgsz 320 --name fire_test`
  - 데이터 다운로드, 학습, val/test 평가: 성공
  - **NCNN 변환: 실패**
    - 학습한 yolo26n 모델: `exporter.py`에서 `KeyError: 'feats'`
      (직전 경고: "NCNN export does not support end2end models, disabling end2end branch")
    - 기본 `yolo26n.pt` / `yolo11n.pt`: pnnx 단계에서 `model_pnnx.py` 없음 (`FileNotFoundError`)
    - 이전 Mac이 torch 2.2.2 + ultralytics 8.4.123이라 환경 문제일 가능성이 큼 (미확인).
  - `train_fire.py`는 변환이 실패해도 `fire_n.pt`는 남기고, 다시 변환하는 명령을 출력하게 해둠.
- 이전 Mac CPU로는 전체 학습이 5일 이상 걸려서 GPU에서 학습하기로 함.

## 해야 할 것
1. 이 컴퓨터 환경 확인 (OS, Python 버전, NVIDIA GPU/CUDA 또는 Apple Silicon 여부).
2. 가상환경(`venv`) 만들고 `ultralytics`, `ncnn`, `kagglehub`, `opencv-python` 설치. GPU가 있으면 그에 맞는 torch가 깔렸는지 확인.
3. 빠른 테스트로 전체 흐름 확인. **특히 NCNN 변환이 되는지 꼭 확인**:
   `python training/train_fire.py --fraction 0.01 --epochs 1 --imgsz 320 --name fire_test`
   - 또 실패하면 원인을 찾아서 고칠 것. yolo26 문제면 `--model yolo11n.pt`로 바꾸는 것도 검토.
   - 끝나면 테스트 결과물(`fire_test.pt`, `fire_test_ncnn_model/`, `runs/detect/fire_test`, `yolo26n.pt`)은 지울 것.
4. **학습 시작 전에 이 컴퓨터 환경과 예상 학습 시간을 사용자에게 먼저 알릴 것.**
5. GPU가 있으면 본 학습 실행 (오래 걸리니 백그라운드로 돌리고 진행 상황 보고):
   `python training/train_fire.py --epochs 60 --imgsz 640`
   GPU가 없으면 Colab 노트북으로 학습하는 방법을 안내.
6. 학습이 끝나면 `ppl.py`에서 `fire_n_ncnn_model`이 정상 로드되는지 확인하고, 이 파일을 최신 상태로 고친 뒤 커밋·push.

## 아직 없는 것
- 사람 안전 모델 `ppe_n_ncnn_model/` (또는 `ppe_n.pt`): 저장소에 없음. 사용자가 원래 만든 곳에서 복사해 와야 함.
