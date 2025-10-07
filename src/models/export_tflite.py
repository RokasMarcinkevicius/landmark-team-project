# src/models/export_tflite.py — convert FROM CONCRETE FUNCTIONS (no SavedModel), MLIR kapalı
import os, numpy as np, tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.metrics import top_k_accuracy_score

# --- GPU/Metal'i devre dışı bırak; CPU'da dönüştür ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

MODEL_PATH = "models/landmark_mnv3.keras"
IMG_SIZE = (224, 224)
TEST_DIR = "data/test"
EXPORT_DIR, REPORTS_DIR = "exports", "reports"
Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)
Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

print(f"Loading Keras model from: {MODEL_PATH}")
full = keras.models.load_model(MODEL_PATH, compile=False)

# ---- Augment'ı grafikten çıkar: 'aug' sonrasını çekirdek olarak al ----
aug = full.get_layer("aug")  # eğitimde name="aug" verdik
core = keras.Model(inputs=aug.output, outputs=full.output, name="inference_core")
core.trainable = False

# ---- Tüm değişkenlerin yaratıldığından emin olmak için 1 dummy forward ----
_ = core(tf.zeros([1, IMG_SIZE[0], IMG_SIZE[1], 3], dtype=tf.float32))

# ---- Concrete function tanımı (serving) ----
@tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE[0], IMG_SIZE[1], 3], tf.float32, name="input")])
def serving_fn(x):
    return {"pred": core(x, training=False)}

concrete = serving_fn.get_concrete_function()

# =========================
# 1) FP32 TFLite (concrete'tan) — MLIR kapalı + SELECT_TF_OPS açık
# =========================
def tflite_from_concrete(cf):
    conv = tf.lite.TFLiteConverter.from_concrete_functions([cf], core)
    # MLIR/“new converter” kapalı (legacy yol)
    if hasattr(conv, "experimental_new_converter"):
        conv.experimental_new_converter = False
    if hasattr(conv, "_experimental_disable_mlir_converter"):
        conv._experimental_disable_mlir_converter = True
    # Karma op'lar için TF ops'a izin ver
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                      tf.lite.OpsSet.SELECT_TF_OPS]
    return conv

print("Exporting TFLite FP32 from concrete function (legacy + SELECT_TF_OPS)...")
conv_fp32 = tflite_from_concrete(concrete)
tflite_fp32 = conv_fp32.convert()
fp32_path = f"{EXPORT_DIR}/landmark_mnv3_fp32.tflite"
with open(fp32_path, "wb") as f:
    f.write(tflite_fp32)
print(f"✓ Saved: {fp32_path}")

# =========================
# 2) INT8 kuantizasyon — DRQ fallback (tam INT8 bazı sürümlerde sorunlu)
# =========================
def tflite_int8_or_drq(cf):
    # Önce tam INT8 dene (MLIR kapalı)
    conv = tflite_from_concrete(cf)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    # Tam INT8 denemesi (giriş/çıkış INT8); başarısız olursa DRQ yapacağız
    try:
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        blob = conv.convert()
        return blob, "INT8", f"{EXPORT_DIR}/landmark_mnv3_int8.tflite"
    except Exception as e:
        print("Full INT8 failed, falling back to DRQ. Reason:", e)
        # DRQ: giriş/çıkış float kalır; SELECT_TF_OPS'u da açıyoruz
        conv = tflite_from_concrete(cf)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        blob = conv.convert()
        return blob, "INT8 (DRQ fallback)", f"{EXPORT_DIR}/landmark_mnv3_int8_drq.tflite"

print("Exporting TFLite INT8 (legacy; fallback DRQ)...")
int8_blob, int8_label, int8_path = tflite_int8_or_drq(concrete)
with open(int8_path, "wb") as f:
    f.write(int8_blob)
print(f"✓ Saved: {int8_path} [{int8_label}]")

# =========================
# 3) Mini sanity test (Keras core vs FP32 vs INT8/DRQ)
# =========================
def load_test_ds():
    ds = keras.preprocessing.image_dataset_from_directory(
        TEST_DIR, image_size=IMG_SIZE, batch_size=1, shuffle=False, label_mode="categorical"
    )
    return ds.prefetch(tf.data.AUTOTUNE), ds.class_names

def run_keras(m, ds):
    yt, yp, ps = [], [], []
    for bx, by in ds:
        p = m.predict(bx, verbose=0)
        ps.append(p[0])
        yt.append(int(np.argmax(by.numpy(), axis=1)[0]))
        yp.append(int(np.argmax(p, axis=1)[0]))
    return np.array(yt), np.array(yp), np.stack(ps, axis=0)

def run_tflite(path, ds, input_is_int8=False):
    itp = tf.lite.Interpreter(model_path=path)
    itp.allocate_tensors()
    ide, ode = itp.get_input_details()[0], itp.get_output_details()[0]

    in_scale  = ide["quantization_parameters"]["scales"]
    in_zp     = ide["quantization_parameters"]["zero_points"]
    out_scale = ode["quantization_parameters"]["scales"]
    out_zp    = ode["quantization_parameters"]["zero_points"]

    yt, yp, ps = [], [], []
    for bx, by in ds:
        x = tf.cast(bx, tf.float32).numpy()
        if input_is_int8 or ide["dtype"] == np.int8:
            s = in_scale if len(in_scale) > 0 else np.array([1/128], np.float32)
            z = in_zp    if len(in_zp)    > 0 else np.array([0], np.int32)
            x = np.round(x / s + z).astype(np.int8)
        itp.set_tensor(ide["index"], x)
        itp.invoke()
        out = itp.get_tensor(ode["index"])
        if ode["dtype"] == np.int8:
            so = out_scale if len(out_scale) > 0 else np.array([1/128], np.float32)
            zo = out_zp    if len(out_zp)    > 0 else np.array([0], np.int32)
            out = (out.astype(np.float32) - zo) * so
        ps.append(out[0])
        yt.append(int(np.argmax(by.numpy(), axis=1)[0]))
        yp.append(int(np.argmax(out, axis=1)[0]))
    return np.array(yt), np.array(yp), np.stack(ps, axis=0)

# Test verisini yükle
test_ds, class_names = load_test_ds()

# Keras (core) referans
yt_k, yp_k, pk = run_keras(core, test_ds)
top1_k = (yt_k == yp_k).mean()
top3_k = top_k_accuracy_score(yt_k, pk, k=3)

# FP32
yt_f, yp_f, pf = run_tflite(fp32_path, test_ds)
top1_f = (yt_f == yp_f).mean()
top3_f = top_k_accuracy_score(yt_f, pf, k=3)

# INT8/DRQ
yt_i, yp_i, pi = run_tflite(int8_path, test_ds)
top1_i = (yt_i == yp_i).mean()
top3_i = top_k_accuracy_score(yt_i, pi, k=3)

# Rapor
with open(f"{REPORTS_DIR}/tflite_eval.md", "w", encoding="utf-8") as f:
    f.write("# TFLite Eval (concrete function path, MLIR off)\n\n")
    f.write(f"- Keras(core): Top-1 **{top1_k:.3f}**, Top-3 **{top3_k:.3f}**\n")
    f.write(f"- TFLite FP32: Top-1 **{top1_f:.3f}**, Top-3 **{top3_f:.3f}**\n")
    f.write(f"- TFLite {int8_label}: Top-1 **{top1_i:.3f}**, Top-3 **{top3_i:.3f}**\n")

print("====================================================")
print("Eval tamamlandı -> reports/tflite_eval.md")
print(f"Keras(core) Top-1 {top1_k:.3f} | FP32 {top1_f:.3f} | {int8_label} {top1_i:.3f}")
print("====================================================")
