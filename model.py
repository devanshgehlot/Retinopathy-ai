"""
DR Model Wrapper — bridges real EfficientNet-B0 backend with Streamlit frontend.
Falls back to mock if dependencies/weights missing.
"""
import os, random
try:
    from PIL import Image
    import numpy as np
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False

try:
    import torch, torch.nn.functional as F_torch, cv2
    from model_arch import load_model
    from inference import preprocess_image, run_prediction, generate_gradcam
    HAS_TORCH = True
except ImportError as e:
    print(f"[WARNING] {e}")
    HAS_TORCH = False

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "best_model.pth")
USE_MOCK = not (HAS_TORCH and os.path.exists(MODEL_PATH))
print(f"[INFO] {'MOCK' if USE_MOCK else 'REAL'} mode | torch={HAS_TORCH} | weights={os.path.exists(MODEL_PATH)}")

SEVERITY_LABELS = {0:"No DR",1:"Mild NPDR",2:"Moderate NPDR",3:"Severe NPDR",4:"Proliferative DR"}
SEVERITY_DESCRIPTIONS = {
    0:"No signs of diabetic retinopathy detected. Regular annual screening recommended.",
    1:"Mild non-proliferative diabetic retinopathy. Microaneurysms detected. Follow-up in 6-12 months.",
    2:"Moderate non-proliferative diabetic retinopathy. Multiple retinal abnormalities present. Refer to ophthalmologist within 3-6 months.",
    3:"Severe non-proliferative diabetic retinopathy. Extensive retinal damage. Urgent referral to retinal specialist required.",
    4:"Proliferative diabetic retinopathy. Neovascularization detected. Immediate treatment required.",
}
SEVERITY_COLORS = {0:"#2cc985",1:"#60d394",2:"#ffc107",3:"#ff8c42",4:"#ff4757"}

class DRModel:
    def __init__(self):
        self.model = None
        self.device = None
        if not USE_MOCK:
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = load_model(MODEL_PATH, self.device)
                print(f"[INFO] Real model loaded on {self.device}")
            except Exception as e:
                print(f"[WARNING] {e} — falling back to mock")

    def _real_predict(self, image_path):
        pil_image = Image.open(image_path)
        input_tensor, _ = preprocess_image(pil_image)
        result = run_prediction(self.model, input_tensor, self.device)
        severity = result["grade"]
        probs = [result["probabilities"][str(i)] / 100.0 for i in range(5)]
        return {"severity":severity,"label":SEVERITY_LABELS[severity],"confidence":round(probs[severity],4),
                "probabilities":[round(p,4) for p in probs],"description":SEVERITY_DESCRIPTIONS[severity],"color":SEVERITY_COLORS[severity]}

    def _real_gradcam(self, image_path, save_path):
        try:
            import base64, io
            pil = Image.open(image_path)
            inp, rgb = preprocess_image(pil)
            res = run_prediction(self.model, inp, self.device)
            hm_b64 = generate_gradcam(self.model, inp, rgb, res["grade"])
            Image.open(io.BytesIO(base64.b64decode(hm_b64))).save(save_path, quality=90)
            return save_path
        except Exception as e:
            print(f"[ERROR] Grad-CAM: {e}")
            return self._mock_gradcam(image_path, save_path)

    def _mock_predict(self, image_path):
        if HAS_IMAGING:
            try:
                px = np.array(Image.open(image_path).resize((64,64)))
                random.seed(int(np.mean(px[:,:,0])*100+np.mean(px[:,:,1])*50)%1000)
            except: random.seed(None)
        else: random.seed(None)
        severity = random.choices([0,1,2,3,4], weights=[.6,.15,.12,.08,.05], k=1)[0]
        probs = [0.0]*5; mp = random.uniform(.65,.95); probs[severity] = mp
        rem = 1.0-mp
        for i in range(5):
            if i != severity: probs[i] = 1.0/(abs(i-severity)+0.5)
        s = sum(probs[i] for i in range(5) if i!=severity)
        if s>0:
            for i in range(5):
                if i!=severity: probs[i] = (probs[i]/s)*rem
        random.seed(None)
        return {"severity":severity,"label":SEVERITY_LABELS[severity],"confidence":round(probs[severity],4),
                "probabilities":[round(p,4) for p in probs],"description":SEVERITY_DESCRIPTIONS[severity],"color":SEVERITY_COLORS[severity]}

    def _mock_gradcam(self, image_path, save_path):
        if not HAS_IMAGING: return None
        try:
            img = Image.open(image_path).convert("RGB"); w,h = img.size
            y,x = np.ogrid[0:h,0:w]; np.random.seed(int(np.mean(np.array(img.resize((64,64)))[:,:,0]))%100)
            mask = np.zeros((h,w),dtype=np.float32)
            for _ in range(np.random.randint(2,5)):
                cx,cy = np.random.randint(int(w*.2),int(w*.8)), np.random.randint(int(h*.2),int(h*.8))
                rx,ry = np.random.randint(int(w*.1),int(w*.3)), np.random.randint(int(h*.1),int(h*.3))
                mask = np.maximum(mask, np.exp(-(((x-cx)**2)/rx**2+((y-cy)**2)/ry**2)))
            np.random.seed(None)
            hm = np.zeros((h,w,3),dtype=np.uint8); hm[:,:,0]=(mask*255).astype(np.uint8); hm[:,:,1]=(mask*150).astype(np.uint8)
            Image.blend(img, Image.fromarray(hm), alpha=0.45).save(save_path); return save_path
        except Exception as e: print(f"Mock heatmap error: {e}"); return None

    def predict(self, image_path):
        return self._real_predict(image_path) if self.model else self._mock_predict(image_path)

    def generate_heatmap(self, image_path, save_path):
        return self._real_gradcam(image_path, save_path) if self.model else self._mock_gradcam(image_path, save_path)

dr_model = DRModel()
