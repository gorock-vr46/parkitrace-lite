"""ParkiTrace-Lite: EfficientNet-B0 (CNN branch) + tiny Vision Transformer (ViT branch), fused, Grad-CAM-ready.
Trains on dataset/healthy + dataset/parkinson, then exports model/parkitrace.onnx + model/meta.json for the browser.
Usage:  python train_light.py --data dataset --epochs 15"""
import argparse, copy, json, numpy as np, torch, torch.nn as nn
from PIL import ImageFilter
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

CLASSES, MEAN, STD = ["healthy", "parkinson"], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.n1, s.qkv, s.o, s.n2 = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d), nn.LayerNorm(d)
        s.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
    def forward(s, x):
        B, N, D = x.shape
        q, k, v = s.qkv(s.n1(x)).reshape(B, N, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-2, -1) / (D // s.h) ** 0.5).softmax(-1)
        x = x + s.o((a @ v).transpose(1, 2).reshape(B, N, D))
        return x + s.mlp(s.n2(x))

class TinyViT(nn.Module):                      # 16x16 patches -> 196 tokens -> 3 transformer blocks
    def __init__(s, d=128, depth=3, heads=4):
        super().__init__()
        s.patch = nn.Conv2d(3, d, 16, 16); s.cls = nn.Parameter(torch.zeros(1, 1, d))
        s.pos = nn.Parameter(torch.randn(1, 197, d) * 0.02)
        s.blocks = nn.Sequential(*[Block(d, heads) for _ in range(depth)]); s.norm = nn.LayerNorm(d)
    def forward(s, x):
        t = s.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([s.cls.expand(len(t), -1, -1), t], 1) + s.pos
        return s.norm(s.blocks(t))[:, 0]

class ParkiTraceLite(nn.Module):
    def __init__(s, pretrained=True):
        super().__init__()
        s.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None).features
        s.vit, s.drop, s.head = TinyViT(), nn.Dropout(0.3), nn.Linear(1280 + 128, 2)
    def forward(s, x):
        f = s.cnn(x)                                              # B x 1280 x 7 x 7 (used for Grad-CAM in the browser)
        return s.head(s.drop(torch.cat([f.mean((2, 3)), s.vit(x)], 1))), f   # feature fusion

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset"); p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs", type=int, default=16); p.add_argument("--no-pretrained", action="store_true")
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    pre = [T.Resize((224, 224)), T.Lambda(lambda im: im.convert("RGB").filter(ImageFilter.MedianFilter(3)))]  # resize + denoise
    fin = [T.ToTensor(), T.Normalize(MEAN, STD)]
    aug = T.Compose(pre + [T.RandomRotation(15), T.RandomHorizontalFlip(), T.ColorJitter(brightness=0.3)] + fin)
    plain = T.Compose(pre + fin)
    full = datasets.ImageFolder(a.data); assert full.classes == CLASSES, f"Need folders {CLASSES}, found {full.classes}"
    tr_i, va_i = train_test_split(range(len(full)), test_size=0.2, stratify=full.targets, random_state=42)
    tr = DataLoader(Subset(datasets.ImageFolder(a.data, aug), tr_i), a.bs, shuffle=True)
    va = DataLoader(Subset(datasets.ImageFolder(a.data, plain), va_i), a.bs)
    net = ParkiTraceLite(not a.no_pretrained).to(dev)
    opt = torch.optim.AdamW([{"params": net.cnn.parameters(), "lr": 1e-4},
                             {"params": list(net.vit.parameters()) + list(net.head.parameters()), "lr": 1e-3}], weight_decay=1e-4)
    lossf, best = nn.CrossEntropyLoss(), -1
    for ep in range(a.epochs):
        net.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(); lossf(net(x)[0], y).backward(); opt.step()
        net.eval(); P, Y = [], []
        with torch.no_grad():
            for x, y in va: P += net(x.to(dev))[0].argmax(1).cpu().tolist(); Y += y.tolist()
        acc = float(np.mean(np.array(P) == np.array(Y))); print(f"epoch {ep+1}/{a.epochs}  val_acc={acc:.3f}")
        if acc >= best: best, bestP, state = acc, list(P), copy.deepcopy(net.state_dict())
    net.load_state_dict(state); net.eval().cpu()
    rep = classification_report(Y, bestP, target_names=CLASSES, output_dict=True, zero_division=0)
    print(classification_report(Y, bestP, target_names=CLASSES, zero_division=0))
    dummy = torch.randn(1, 3, 224, 224)
    kw = dict(input_names=["input"], output_names=["logits", "fmap"], opset_version=17)
    try: torch.onnx.export(net, (dummy,), "model/parkitrace.onnx", dynamo=False, **kw)
    except TypeError: torch.onnx.export(net, (dummy,), "model/parkitrace.onnx", **kw)
    W = net.head.weight[:, :1280].detach().numpy().round(5).tolist()   # CNN-branch weights -> exact Grad-CAM in browser
    json.dump({"classes": CLASSES, "mean": MEAN, "std": STD, "head": W, "accuracy": best, "report": rep,
               "confusion_matrix": confusion_matrix(Y, bestP).tolist()}, open("model/meta.json", "w"))
    try:
        import onnxruntime as ort
        o = ort.InferenceSession("model/parkitrace.onnx").run(None, {"input": dummy.numpy()})[0]
        print("ONNX check, max difference vs PyTorch:", float(np.abs(o - net(dummy)[0].detach().numpy()).max()))
    except Exception as e: print("ONNX check skipped:", e)
    import os; print(f"Saved model/parkitrace.onnx ({os.path.getsize('model/parkitrace.onnx')/1e6:.1f} MB) and model/meta.json")
