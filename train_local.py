"""날짜 특화 경량 텍스트 인식기 (Date-OCR) 로컬 CPU/GPU 원클릭 학습 스크립트.

실행 방법:
    python train_local.py          # 빠른 고속 학습 (5,000장 샘플링, 약 3~4분 소요)
    python train_local.py --full   # 전체 데이터셋 학습
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# -------------------------------------------------------------
# 1. 모델 아키텍처 (초경량 Date-CRNN: 47만 파라미터 / 1.8MB)
# -------------------------------------------------------------
class DateCRNN(nn.Module):
    def __init__(self, num_classes=49):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H: 32->16, W: W->W/2

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H: 16->8, W: W/2->W/4

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),  # H: 8->4, W: W/4->W/4

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),  # H: 4->2, W: W/4->W/4

            nn.Conv2d(128, 128, (2, 1), 1, 0),  # H: 2->1, W: W/4->W/4
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.rnn = nn.LSTM(128, 64, bidirectional=True, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.cnn(x).squeeze(2).permute(0, 2, 1)  # (B, W/4, 128)
        recurrent, _ = self.rnn(feat)                    # (B, W/4, 128)
        logits = self.fc(recurrent)                      # (B, W/4, num_classes)
        return logits.log_softmax(2)


# -------------------------------------------------------------
# 2. 데이터셋 및 전처리
# -------------------------------------------------------------
class DateDataset(Dataset):
    def __init__(self, list_file, char2idx, data_dir=None, img_h=32, max_w=256, is_train=True, max_samples=None):
        self.char2idx = char2idx
        self.data_dir = Path(data_dir) if data_dir else Path(list_file).parent
        self.img_h = img_h
        self.max_w = max_w
        self.is_train = is_train
        self.samples = []

        lines = Path(list_file).read_text(encoding="utf-8").splitlines()
        for line in lines:
            if "\t" in line:
                p, lbl = line.split("\t", 1)
                self.samples.append((p.strip(), lbl.strip()))

        if max_samples and len(self.samples) > max_samples:
            # Real 샘플(Date-Real)은 무조건 포함하고 Synth만 샘플링
            real_s = [s for s in self.samples if "Date-Real" in s[0]]
            synth_s = [s for s in self.samples if "Date-Real" not in s[0]]
            num_synth = max(0, max_samples - len(real_s))
            random.seed(42)
            synth_sampled = random.sample(synth_s, min(num_synth, len(synth_s)))
            self.samples = real_s + synth_sampled
            random.shuffle(self.samples)

        self.aug = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]) if is_train else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_str, label = self.samples[idx]
        img_p = Path(path_str)
        if not img_p.exists():
            for cand in [
                self.data_dir / path_str,
                self.data_dir / "images" / img_p.name,
                Path("images") / img_p.name,
                self.data_dir.parent / path_str,
            ]:
                if cand.exists():
                    img_p = cand
                    break
                    
        if cv2 is not None:
            img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((self.img_h, 64), dtype=np.uint8)
            if self.is_train and random.random() < 0.2:
                kernel = np.ones((2, 2), np.uint8)
                img = cv2.erode(img, kernel, iterations=1)
            h, w = img.shape
            new_w = max(16, min(self.max_w, int(w * (self.img_h / max(h, 1)))))
            img = cv2.resize(img, (new_w, self.img_h), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        else:
            try:
                pil_img = Image.open(str(img_p)).convert("L")
                w, h = pil_img.size
                new_w = max(16, min(self.max_w, int(w * (self.img_h / max(h, 1)))))
                pil_img = pil_img.resize((new_w, self.img_h), Image.Resampling.BILINEAR)
                arr = np.array(pil_img, dtype=np.uint8)
            except Exception:
                arr = np.zeros((self.img_h, 64), dtype=np.uint8)
            tensor = torch.from_numpy(arr).float().unsqueeze(0) / 255.0

        if self.aug:
            tensor = self.aug(tensor)
        tensor = (tensor - 0.5) / 0.5

        target = [self.char2idx[c] for c in label if c in self.char2idx]
        return tensor, torch.tensor(target, dtype=torch.long), label


def collate_fn(batch):
    max_w = max(item[0].shape[2] for item in batch)
    max_w = max(64, ((max_w + 31) // 32) * 32)

    tensors, targets, lengths, labels = [], [], [], []
    for t, tgt, lbl in batch:
        w = t.shape[2]
        padded = F.pad(t, (0, max_w - w, 0, 0), value=-1.0)
        tensors.append(padded)
        targets.extend(tgt.tolist())
        lengths.append(len(tgt))
        labels.append(lbl)
    return torch.stack(tensors), torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long), labels


def decode_prediction(logits, idx2char):
    preds = logits.argmax(dim=2).cpu().numpy()
    decoded_texts = []
    for row in preds:
        chars = []
        prev = 0
        for idx in row:
            if idx != 0 and idx != prev:
                chars.append(idx2char.get(idx, ""))
            prev = idx
        decoded_texts.append("".join(chars))
    return decoded_texts


# -------------------------------------------------------------
# 3. 메인 학습 루틴
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="날짜 전용 OCR 모델 로컬 학습")
    parser.add_argument("--full", action="store_true", help="전체 128,450장 학습 (지정하지 않으면 빠른 5,000장 모드)")
    parser.add_argument("--epochs", type=int, default=10, help="학습 에폭 수 (기본: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기 (기본: 32)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "dataset" / "rec",
        script_dir / "data",
        script_dir / "rec",
        script_dir,
        Path(r"C:\Users\User\Desktop\ITDA_26\dataset\rec"),
    ]
    rec_dir = next((c for c in candidates if (c / "train.txt").exists()), script_dir)
    weights_dir = script_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 작업 디렉터리: {script_dir}")
    print(f"📊 데이터셋 디렉터리: {rec_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 실행 디바이스: {device} ({os.cpu_count()} CPU 코어 감지)")

    # 1) 문자 사전 로드
    dict_file = rec_dir / "dict.txt"
    if dict_file.exists():
        chars = [l for l in dict_file.read_text(encoding="utf-8").splitlines() if l != ""]
    else:
        chars = [
            " ", "-", ".", "/",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":",
            "A", "B", "C", "D", "E", "F", "G", "J", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "Y",
            "a", "b", "c", "e", "g", "l", "n", "o", "p", "r", "t", "u", "v", "y",
        ]
    char2idx = {c: i + 1 for i, c in enumerate(chars)}
    idx2char = {i + 1: c for i, c in enumerate(chars)}
    num_classes = len(chars) + 1
    print(f"✅ 날짜 전용 사전: {len(chars)}개 토큰 (CTC Blank 0번 포함 {num_classes}개 클래스)")

    # 2) 데이터셋 구성
    max_train_samples = None if args.full else 5000
    train_file = rec_dir / "train.txt"
    val_file = rec_dir / "val.txt"

    train_ds = DateDataset(train_file, char2idx, data_dir=rec_dir, is_train=True, max_samples=max_train_samples)
    val_ds = DateDataset(val_file, char2idx, data_dir=rec_dir, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    mode_name = "전체 128,450장 모드" if args.full else "고속 최적화 모드 (약 3~4분)"
    print(f"📊 학습 모드: {mode_name} | 학습 샘플: {len(train_ds):,}장 | 검증(Real): {len(val_ds)}장")

    # 3) 모델 초기화
    model = DateCRNN(num_classes=num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 모델 생성 완료: 파라미터 {num_params:,}개 ({num_params * 4 / (1024 * 1024):.2f} MB)")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_pth = weights_dir / "date_rec_best.pth"
    t_start = time.time()

    print(f"\n=== 🏁 학습 시작 ({args.epochs} Epochs) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        ep_t0 = time.time()

        for tensors, targets, target_lengths, _ in train_loader:
            tensors = tensors.to(device)
            log_probs = model(tensors)
            log_probs_t = log_probs.permute(1, 0, 2)
            input_lengths = torch.full((tensors.size(0),), log_probs.size(1), dtype=torch.long, device=device)

            loss = criterion(log_probs_t, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)
        ep_elapsed = time.time() - ep_t0

        # 검증
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for tensors, _, _, labels in val_loader:
                tensors = tensors.to(device)
                log_probs = model(tensors)
                preds = decode_prediction(log_probs, idx2char)
                for p, g in zip(preds, labels):
                    if p == g:
                        correct += 1
                    total += 1

        acc = correct / max(total, 1) * 100
        print(f"Epoch {epoch:02d}/{args.epochs:02d} ({ep_elapsed:.1f}s) | Loss: {train_loss:.4f} | Val Real Acc: {acc:.1f}% ({correct}/{total})")

        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), str(best_pth))
            print(f"  🌟 최고 성능 저장! (Acc: {acc:.1f}%) -> {best_pth.name}")

    total_time = time.time() - t_start
    print(f"\n🎉 학습 완료! 총 소요 시간: {total_time:.1f}초 ({total_time / 60:.1f}분) | 최고 정확도: {best_acc:.1f}%")

    # 4) ONNX 내보내기
    print("\n📦 ONNX 모델 변환 시작...")
    model.load_state_dict(torch.load(str(best_pth), map_location="cpu"))
    model.eval().to("cpu")

    dummy_input = torch.randn(1, 1, 32, 192, dtype=torch.float32)
    onnx_path = weights_dir / "date_rec_custom.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 3: "width"},
            "output": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"✅ ONNX 변환 성공: {onnx_path} ({size_mb:.2f} MB)")

    # 5) ONNX Runtime 추론 지연시간 측정
    print("\n⚡ ONNX Runtime CPU 추론 벤치마크...")
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        dummy_np = np.random.randn(1, 1, 32, 192).astype(np.float32)

        for _ in range(5):
            _ = sess.run(None, {"input": dummy_np})

        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = sess.run(None, {"input": dummy_np})
            latencies.append((time.perf_counter() - t0) * 1000)

        p50 = np.median(latencies)
        p90 = np.percentile(latencies, 90)
        print(f"⚡ CPU 단일 추론 지연시간: 중간값 {p50:.2f} ms | p90 {p90:.2f} ms")

        # 샘플 5개 추론 시연
        print("\n🔍 실제 Real 검증 샘플 예측 테스트:")
        for path_str, label in val_ds.samples[:5]:
            if cv2 is not None:
                img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                h, w = img.shape
                new_w = max(16, min(256, int(w * (32 / max(h, 1)))))
                new_w = ((new_w + 31) // 32) * 32
                resized = cv2.resize(img, (new_w, 32), interpolation=cv2.INTER_LINEAR)
                inp = ((resized.astype(np.float32) / 255.0 - 0.5) / 0.5)[np.newaxis, np.newaxis, :, :]
            else:
                try:
                    pil_img = Image.open(path_str).convert("L")
                    w, h = pil_img.size
                    new_w = max(16, min(256, int(w * (32 / max(h, 1)))))
                    new_w = ((new_w + 31) // 32) * 32
                    resized = pil_img.resize((new_w, 32), Image.Resampling.BILINEAR)
                    arr = np.array(resized, dtype=np.float32)
                    inp = ((arr / 255.0 - 0.5) / 0.5)[np.newaxis, np.newaxis, :, :]
                except Exception:
                    continue
                out = sess.run(None, {"input": inp})[0]
                preds = out[0].argmax(axis=-1)
                text = []
                prev = 0
                for idx in preds:
                    if idx != 0 and idx != prev:
                        text.append(idx2char.get(idx, ""))
                    prev = idx
                pred_str = "".join(text)
                mark = "✅" if pred_str == label else "❌"
                print(f"  {mark} 정답: {label:<15} | 예측: {pred_str:<15} | 파일: {Path(path_str).name}")

    except Exception as e:
        print(f"ONNX 검증 스킵: {e}")

    print("\n✨ 모든 과정이 성공적으로 완료되었습니다!")
    print(f"생성된 모델: {onnx_path}")


if __name__ == "__main__":
    main()
