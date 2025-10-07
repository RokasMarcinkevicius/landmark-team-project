import os, pathlib, numpy as np, tensorflow as tf, wandb
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
from wandb.integration.keras import WandbMetricsLogger

# ==== W&B ====
wandb.init(project="landmark-id", name="mnv3small-baseline",
           config={
             "arch":"MobileNetV3Small",
             "img_size":224,
             "batch":32,
             "epochs":25,
             "optimizer":"AdamW",
             "lr":1e-3,
             "weight_decay":1e-4,
             "augment":"flip,zoom,brightness,contrast",
             "seed":42
           })
CFG = wandb.config
tf.keras.utils.set_random_seed(CFG.seed)

IMG_SIZE = (CFG.img_size, CFG.img_size)
BATCH = CFG.batch

train_ds = keras.preprocessing.image_dataset_from_directory(
    "data/train", image_size=IMG_SIZE, batch_size=BATCH, seed=CFG.seed, label_mode="categorical")
val_ds = keras.preprocessing.image_dataset_from_directory(
    "data/val", image_size=IMG_SIZE, batch_size=BATCH, seed=CFG.seed, label_mode="categorical")
test_ds = keras.preprocessing.image_dataset_from_directory(
    "data/test", image_size=IMG_SIZE, batch_size=BATCH, seed=CFG.seed, shuffle=False, label_mode="categorical")

class_names = train_ds.class_names
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# === class_weight (inverse-frequency) ===
train_dir = pathlib.Path("data/train")
counts = {name: len(list((train_dir / name).glob("*"))) for name in class_names}
total = sum(counts.values())
class_weight = {i: total / (len(class_names) * counts[name]) for i, name in enumerate(class_names)}

# ==== Augment ====
data_augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.12),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.10),
], name="aug")

# ==== Model ====
base = keras.applications.MobileNetV3Small(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augment(inputs)
x = keras.applications.mobilenet_v3.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)

opt = keras.optimizers.AdamW(learning_rate=CFG.lr, weight_decay=CFG.weight_decay)
model.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=[
        keras.metrics.TopKCategoricalAccuracy(k=1, name="top1"),
        keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
    ],
)

os.makedirs("models", exist_ok=True)

# ==== Callbacks (genel) ====
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                  monitor="val_top1", mode="max"),
    WandbMetricsLogger(log_freq="epoch"),
    keras.callbacks.ModelCheckpoint(
        filepath="models/ckpt_{epoch:02d}_{val_top1:.3f}.keras",
        monitor="val_top1", mode="max",
        save_best_only=True, save_weights_only=False
    ),
]

# ======= Baş: Başlığı eğit =======
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CFG.epochs,
    callbacks=callbacks,
    verbose=2,
    class_weight=class_weight,  # <-- eklendi
)

# === Fine-tune (daha az layer + BN freeze) ===
base.trainable = True

# BN katmanlarını dondur (stabil)
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Sadece son 15 katman açık
for layer in base.layers[:-15]:
    layer.trainable = False

# Yeni optimizer ile düşük LR (decay kapalı)
opt_ft = keras.optimizers.AdamW(learning_rate=3e-6, weight_decay=0.0)

# Label smoothing ile compile
model.compile(
    optimizer=opt_ft,
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=[
        keras.metrics.TopKCategoricalAccuracy(k=1, name="top1"),
        keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
    ],
)

# Kısa fine-tune (+ ReduceLROnPlateau)
ft_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks + [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=2, min_lr=1e-7, verbose=1
        )
    ],
    verbose=2,
    class_weight=class_weight,  # <-- eklendi
)

# ======= Test değerlendirme =======
test_metrics = model.evaluate(test_ds, verbose=0)
metrics_dict = {f"test_{name}": float(val) for name, val in zip(model.metrics_names, test_metrics)}
wandb.log(metrics_dict)

# ======= Karışıklık matrisi =======
y_true, y_pred = [], []
for bx, by in test_ds:
    preds = model.predict(bx, verbose=0)
    y_true.extend(np.argmax(by.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
cm = confusion_matrix(y_true, y_pred)

# CM'i W&B'ye görsel ve tablo olarak yükle
import matplotlib.pyplot as plt
import seaborn as sns
os.makedirs("reports", exist_ok=True)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Pred"); plt.ylabel("True"); plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=200)
wandb.log({"confusion_matrix_image": wandb.Image("reports/confusion_matrix.png")})

# ======= Sınıf bazlı rapor =======
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
for cls, vals in report.items():
    if isinstance(vals, dict):
        for k, v in vals.items():
            wandb.log({f"cls/{cls}/{k}": v})

# ======= Model kaydet + Artifact =======
model.save("models/landmark_mnv3.keras")

artifact = wandb.Artifact("landmark-mnv3-baseline", type="model",
                          description="MobileNetV3Small baseline + stabilized FT (BN freeze, last15, cw, RLRP)")
artifact.add_file("models/landmark_mnv3.keras")
artifact.add_file("reports/confusion_matrix.png")
wandb.log_artifact(artifact)

wandb.finish()
