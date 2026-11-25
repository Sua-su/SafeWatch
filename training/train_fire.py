"""
화재/연기 감지 모델 학습 스크립트 (SafeWatch 공장 안전용)

- 데이터셋: D-Fire (Kaggle: sayedgamal99/smoke-fire-detection-yolo, 약 3.1GB, CC0)
  클래스: 0 = smoke, 1 = fire
- 결과물: 저장소 루트에 fire_n.pt 와 fire_n_ncnn_model/ 을 만든다.
  (ppl.py 의 FIRE_MODEL_PATH 가 바로 이 이름을 사용)

GPU가 있는 곳(Google Colab 등)에서 실행하는 것을 권장.
CPU만 있는 PC나 라즈베리파이에서는 전체 학습에 며칠이 걸린다.

사용 예:
    python training/train_fire.py                         # 기본 학습 (GPU 자동 사용)
    python training/train_fire.py --epochs 100 --batch 32
    python training/train_fire.py --fraction 0.01 --epochs 1 --imgsz 320   # 동작 확인용 빠른 테스트
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # 저장소 루트
KAGGLE_DATASET = "sayedgamal99/smoke-fire-detection-yolo"
CLASS_NAMES = ["smoke", "fire"]                 # D-Fire 클래스 순서 그대로


def download_dataset() -> Path:
    """Kaggle에서 데이터셋을 받고, train/images 가 들어 있는 폴더를 찾아 반환"""
    import kagglehub

    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    for train_dir in sorted(path.rglob("train")):
        if (train_dir / "images").is_dir():
            return train_dir.parent
    raise FileNotFoundError(f"{path} 안에서 train/images 폴더를 찾지 못했습니다.")


def write_data_yaml(data_root: Path) -> Path:
    yaml_path = ROOT / "training" / "fire_data.yaml"
    lines = [
        f"path: {data_root}",
        "train: train/images",
        "val: val/images",
    ]
    if (data_root / "test" / "images").is_dir():
        lines.append("test: test/images")
    lines.append("names:")
    lines += [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main():
    ap = argparse.ArgumentParser(description="화재/연기 YOLO 모델 학습")
    ap.add_argument("--model", default="yolo26n.pt", help="시작 모델 (yolo26n.pt, yolo11n.pt 등)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1이면 GPU 메모리에 맞춰 자동")
    ap.add_argument("--device", default=None, help="0(GPU), cpu, mps 등. 생략하면 자동")
    ap.add_argument("--fraction", type=float, default=1.0, help="학습 데이터 사용 비율 (테스트용)")
    ap.add_argument("--name", default="fire_n", help="결과 이름 (runs/detect/<name>)")
    ap.add_argument("--data", default=None, help="이미 받은 데이터셋 폴더 (train/val/test 상위)")
    args = ap.parse_args()

    from ultralytics import YOLO

    data_root = Path(args.data) if args.data else download_dataset()
    data_yaml = write_data_yaml(data_root)
    print(f"데이터셋: {data_root}\n설정 파일: {data_yaml}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        fraction=args.fraction,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
        patience=15,          # 15 epoch 동안 나아지지 않으면 조기 종료
        plots=True,
    )

    best = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    out_pt = ROOT / f"{args.name}.pt"
    shutil.copy(best, out_pt)
    print(f"모델 저장: {out_pt}")

    # 라즈베리파이에서 빠르게 돌리기 위해 NCNN 으로 변환
    best_model = YOLO(str(out_pt))
    best_model.val(data=str(data_yaml), imgsz=args.imgsz, split="test"
                   if (data_root / "test" / "images").is_dir() else "val")
    try:
        ncnn_dir = Path(best_model.export(format="ncnn", imgsz=args.imgsz))
    except Exception as e:
        # 학습 결과(.pt)는 이미 저장됐으니 변환만 다시 하면 된다
        print(f"NCNN 변환 실패: {type(e).__name__}: {e}")
        print(f"{out_pt.name} 은 저장됐습니다. 라즈베리파이나 최신 torch 환경에서 다시 변환하세요:")
        print(f'  python -c "from ultralytics import YOLO; YOLO(\'{out_pt.name}\').export(format=\'ncnn\', imgsz={args.imgsz})"')
        return
    print(f"NCNN 변환 완료: {ncnn_dir}")
    print("ppl.py 와 같은 폴더에 fire_n_ncnn_model/ 이 있으면 화재 감지가 켜집니다.")


if __name__ == "__main__":
    main()
