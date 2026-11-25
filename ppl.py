"""
SafeWatch - 작업자 안전 + 공장 화재 감지 / 경고등(릴레이) 제어 / 위반 기록
1) 사람 안전: 안전모 '와' 안전조끼를 둘 다 착용해야 정상(OK).
   둘 중 하나라도 없으면 경고등을 켠다.
2) 공장 안전: 불꽃(라이터 불 포함)이나 연기가 보이면 경고등을 켠다.
   화재 모델(FIRE_MODEL_PATH)이 없으면 사람 안전 감지만 동작한다.
- 경고가 켜질 때 사진과 시간을 violations/ 폴더에 저장한다.
- 처음 설치 (라즈베리파이):
        git clone https://github.com/Sua-su/SafeWatch.git ~/SafeWatch
        cd ~/SafeWatch && python3 -m venv venv && source venv/bin/activate
        pip install -r requirements.txt
- 실행: cd ~/SafeWatch && source venv/bin/activate
        DISPLAY=:0 QT_QPA_PLATFORM=xcb python ppl.py
- 종료: 영상 창에서 q 키, 또는 터미널에서 Ctrl + C
"""
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# ===== 설정 =====
CAMERA = 0                        # 웹캠 번호 (/dev/video0)
IMGSZ = 640                       # 입력 크기

# --- 사람 안전 (PPE) ---
PPE_MODEL_PATH = "ppe_n_ncnn_model"   # 학습한 모델 (NCNN 변환본). ppe_n.pt / ppe_s.pt도 가능

# 클래스별 최소 확신도 (착용 장비는 조금 낮게 잡아야 놓치지 않음)
CONF_TABLE = {
    "Person": 0.35,
    "Hardhat": 0.30,
    "Safety Vest": 0.30,
    "NO-Hardhat": 0.45,
    "NO-Safety Vest": 0.45,
}

PPE_ON_FRAMES = 5                 # 연속 몇 프레임 위반이면 경고등 ON
PPE_OFF_FRAMES = 15               # 연속 몇 프레임 정상이면 경고등 OFF

# --- 공장 안전 (화재/연기) ---
FIRE_MODEL_PATH = "fire_n_ncnn_model" # 화재/연기 모델 (NCNN 변환본). fire_n.pt도 가능

# 모델의 클래스 이름(소문자) -> 경고 이유. 모델마다 이름이 다르면 여기에 추가
FIRE_CLASSES = {
    "fire": "화재 감지",
    "flame": "화재 감지",
    "smoke": "연기 감지",
}
FIRE_CONF = {"화재 감지": 0.40, "연기 감지": 0.45}  # 이유별 최소 확신도

FIRE_ON_FRAMES = 3                # 화재는 위험하니 더 빨리 경고
FIRE_OFF_FRAMES = 30              # 불이 꺼진 뒤 충분히 확인하고 경고 해제
FIRE_EVERY = 1                    # 화재 모델을 몇 프레임마다 돌릴지 (느리면 2~3으로)

# --- 경고등 릴레이 ---
RELAY_PIN = 17                    # PPE 경고 릴레이 GPIO 번호 (11번 핀)
FIRE_RELAY_PIN = 17               # 화재 경고 릴레이 GPIO 번호 (같은 경고등이면 RELAY_PIN과 동일)
RELAY_ON_HIGH = True              # 'pinctrl set 17 op dh'에서 켜졌으면 True

LOG_DIR = Path("violations")      # 위반 사진/기록 저장 폴더
SNAPSHOT_INTERVAL = 10            # 경고가 계속될 때 몇 초마다 사진을 추가로 저장할지
# =================

# 화면 표시용 영어 문구 (OpenCV는 한글을 못 그림)
REASON_EN = {
    "안전모 미착용": "NO HELMET",
    "안전조끼 미착용": "NO VEST",
    "화재 감지": "FIRE",
    "연기 감지": "SMOKE",
}

_relay_warned = False


def set_relay(pin: int, on: bool):
    global _relay_warned
    level = "dh" if (on == RELAY_ON_HIGH) else "dl"
    try:
        subprocess.run(["pinctrl", "set", str(pin), "op", level], check=False)
    except FileNotFoundError:
        # 라즈베리파이가 아닌 PC에서 테스트할 때
        if not _relay_warned:
            print("pinctrl 명령이 없어 경고등 제어를 건너뜁니다 (라즈베리파이가 아님).")
            _relay_warned = True


class Alarm:
    """연속 프레임 수로 경고 ON/OFF를 판단하는 경보 하나"""

    def __init__(self, kind, on_frames, off_frames):
        self.kind = kind
        self.on_frames = on_frames
        self.off_frames = off_frames
        self.on = False
        self.bad = self.good = 0
        self.start = self.last_snap = 0.0
        self.reasons = []

    def update(self, violation, reasons):
        """이번 프레임 결과를 반영하고 'ON' / 'OFF' / 'SNAP' / None 중 하나를 반환"""
        if violation:
            self.bad += 1
            self.good = 0
            self.reasons = reasons
        else:
            self.good += 1
            self.bad = 0

        now = time.time()
        if not self.on and self.bad >= self.on_frames:
            self.on = True
            self.start = self.last_snap = now
            return "ON"
        if self.on and self.good >= self.off_frames:
            self.on = False
            return "OFF"
        if self.on and now - self.last_snap >= SNAPSHOT_INTERVAL:
            self.last_snap = now
            return "SNAP"
        return None


def center_in(box, area):
    """box의 중심점이 area 안에 있는지"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return area[0] <= cx <= area[2] and area[1] <= cy <= area[3]


def check_violation(dets):
    """
    dets: [(클래스명, [x1,y1,x2,y2]), ...]
    반환: (위반 여부, 이유 리스트)
    """
    persons = [b for n, b in dets if n == "Person"]
    helmets = [b for n, b in dets if n == "Hardhat"]
    vests = [b for n, b in dets if n == "Safety Vest"]
    names = {n for n, _ in dets}
    reasons = []

    # 1) 미착용이 직접 감지되면 바로 위반
    if "NO-Hardhat" in names:
        reasons.append("안전모 미착용")
    if "NO-Safety Vest" in names:
        reasons.append("안전조끼 미착용")

    # 2) 사람마다 안전모와 조끼가 둘 다 있는지 확인
    for p in persons:
        h = p[3] - p[1]
        upper = [p[0], p[1] - h * 0.15, p[2], p[1] + h * 0.45]   # 머리 쪽 영역
        if not any(center_in(b, upper) for b in helmets):
            reasons.append("안전모 미착용")
        if not any(center_in(b, p) for b in vests):
            reasons.append("안전조끼 미착용")

    # 3) 사람 박스는 못 찾았지만 장비 일부만 보이는 경우
    if not persons and names & {"Hardhat", "Safety Vest"}:
        if not helmets:
            reasons.append("안전모 미착용")
        if not vests:
            reasons.append("안전조끼 미착용")

    reasons = sorted(set(reasons))
    return bool(reasons), reasons


def check_fire(result, names):
    """화재 모델 결과에서 불/연기를 찾는다. 반환: (위반 여부, 이유 리스트)"""
    reasons = set()
    for c, cf in zip(result.boxes.cls, result.boxes.conf):
        reason = FIRE_CLASSES.get(names[int(c)].lower())
        if reason and float(cf) >= FIRE_CONF[reason]:
            reasons.add(reason)
    reasons = sorted(reasons)
    return bool(reasons), reasons


def load_fire_model():
    if not Path(FIRE_MODEL_PATH).exists():
        print(f"화재 모델({FIRE_MODEL_PATH})이 없어 사람 안전 감지만 실행합니다.")
        return None, []
    model = YOLO(FIRE_MODEL_PATH)
    ids = [i for i, n in model.names.items() if n.lower() in FIRE_CLASSES]
    if not ids:
        print(f"화재 모델 클래스 {list(model.names.values())} 중 FIRE_CLASSES와 맞는 이름이 없습니다.")
        return None, []
    return model, ids


def save_snapshot(view, kind, reasons, writer, f):
    now = datetime.now()
    img_path = LOG_DIR / f"{now:%Y%m%d_%H%M%S}_{kind}.jpg"
    cv2.imwrite(str(img_path), view)
    writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), kind, "ON",
                     " / ".join(reasons), img_path.name])
    f.flush()
    print(f"  사진 저장: {img_path}")


def main():
    ppe_model = YOLO(PPE_MODEL_PATH)
    ppe_names = ppe_model.names
    ppe_ids = [i for i, n in ppe_names.items() if n in CONF_TABLE]
    ppe_min_conf = min(CONF_TABLE.values())

    fire_model, fire_ids = load_fire_model()
    fire_min_conf = min(FIRE_CONF.values())

    cap = cv2.VideoCapture(CAMERA)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. CAMERA 번호를 확인하세요.")
        return

    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / "log.csv"
    new_file = not log_file.exists()
    f = open(log_file, "a", newline="", encoding="utf-8-sig")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["시간", "종류", "상태", "이유", "사진"])

    ppe_alarm = Alarm("PPE", PPE_ON_FRAMES, PPE_OFF_FRAMES)
    fire_alarm = Alarm("FIRE", FIRE_ON_FRAMES, FIRE_OFF_FRAMES)
    relay_of = {ppe_alarm: RELAY_PIN, fire_alarm: FIRE_RELAY_PIN}
    pins = set(relay_of.values())
    for pin in pins:
        set_relay(pin, False)

    frame_no = 0
    fire_result = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임을 읽지 못했습니다.")
                break
            frame_no += 1

            t0 = time.time()

            # --- 사람 안전 ---
            ppe_result = ppe_model.predict(frame, imgsz=IMGSZ, conf=ppe_min_conf,
                                           classes=ppe_ids, verbose=False)[0]
            dets = []
            for c, cf, xyxy in zip(ppe_result.boxes.cls, ppe_result.boxes.conf,
                                   ppe_result.boxes.xyxy):
                n = ppe_names[int(c)]
                if float(cf) >= CONF_TABLE[n]:
                    dets.append((n, xyxy.tolist()))
            events = [(ppe_alarm, ppe_alarm.update(*check_violation(dets)))]

            # --- 공장 안전 (화재/연기) ---
            if fire_model is not None and frame_no % FIRE_EVERY == 0:
                fire_result = fire_model.predict(frame, imgsz=IMGSZ, conf=fire_min_conf,
                                                 classes=fire_ids, verbose=False)[0]
                fire_state = check_fire(fire_result, fire_model.names)
                events.append((fire_alarm, fire_alarm.update(*fire_state)))

            ms = (time.time() - t0) * 1000

            view = ppe_result.plot()
            if fire_result is not None:
                view = fire_result.plot(img=view)

            # --- 경보 처리 ---
            for alarm, event in events:
                if event == "ON":
                    print(f"[{alarm.kind} 경고 ON] {', '.join(alarm.reasons)}")
                    save_snapshot(view, alarm.kind, alarm.reasons, writer, f)
                elif event == "OFF":
                    dur = time.time() - alarm.start
                    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                     alarm.kind, "OFF", f"경고 {dur:.0f}초 지속", ""])
                    f.flush()
                    print(f"[{alarm.kind} 경고 OFF] 정상 (경고 {dur:.0f}초)")
                elif event == "SNAP":
                    save_snapshot(view, alarm.kind, alarm.reasons, writer, f)

            # 같은 핀을 쓰는 경보 중 하나라도 켜져 있으면 릴레이 ON
            if any(e in ("ON", "OFF") for _, e in events):
                for pin in pins:
                    set_relay(pin, any(a.on for a, p in relay_of.items() if p == pin))

            # --- 화면 표시 ---
            active = [r for a in (fire_alarm, ppe_alarm) if a.on for r in a.reasons]
            if active:
                text = "WARNING: " + ", ".join(dict.fromkeys(REASON_EN[r] for r in active))
                color = (0, 0, 255)
            else:
                text, color = "OK", (0, 200, 0)
            cv2.putText(view, f"{text}  {ms:.0f}ms", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("SafeWatch", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        for pin in pins:
            set_relay(pin, False)   # 종료할 때 경고등은 반드시 끄기
        cap.release()
        cv2.destroyAllWindows()
        f.close()
        print("종료했습니다. 경고등 OFF")


if __name__ == "__main__":
    main()
