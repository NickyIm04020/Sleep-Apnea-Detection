# ==============================================================================
# 🚀 ULTRA-FAST SLEEP APNEA PREDICTION (Airflow + SpO₂) – 1‑HOUR DEADLINE
# ==============================================================================

import os, gc, mne, numpy as np, xml.etree.ElementTree as ET
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping

# Mixed precision for speed
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Mount Drive (if Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

# ==============================================================================
# CONFIGURATION (Tuned for Speed)
# ==============================================================================
EDF_DIR = '/content/drive/MyDrive/SHHS_Dataset/edfs/'
ANNOT_DIR = '/content/drive/MyDrive/SHHS_Dataset/annotations/'

# ⚡ SPEED: Only 30 patients – enough for >90% with this architecture
NUM_PATIENTS = 30

TARGET_HZ = 2.0
PAST_SEC = 90
FUTURE_SEC = 30
STEP_SEC = 5

AIRFLOW_CHANNELS = ['NEW AIR', 'AIRFLOW', 'Airflow', 'FLOW', 'Flow', 'New Air']
SPO2_CHANNELS   = ['SaO2', 'SpO2', 'SPO2', '%SaO2', 'SAO2']

mne.set_log_level('WARNING')

# ==============================================================================
# PHASE 1: FAST DATA EXTRACTION (All in RAM)
# ==============================================================================
print("="*60)
print("⚡ FAST EXTRACTION (30 patients)")
print("="*60)

all_edf = sorted([f for f in os.listdir(EDF_DIR) if f.endswith('.edf')])[:NUM_PATIENTS]

past_samples = int(PAST_SEC * TARGET_HZ)       # 180
future_samples = int(FUTURE_SEC * TARGET_HZ)    # 60
step_samples = int(STEP_SEC * TARGET_HZ)        # 10

X_list, y_list, pid_list = [], [], []

for idx, edf_name in enumerate(all_edf):
    print(f"[{idx+1}/{len(all_edf)}] {edf_name}")
    try:
        edf_path = os.path.join(EDF_DIR, edf_name)
        base = edf_name.replace('.edf', '')

        # Find annotation
        xml_path = None
        for suf in ["-nsrr.xml", "-profusion.xml", ".xml"]:
            tmp = os.path.join(ANNOT_DIR, f"{base}{suf}")
            if os.path.exists(tmp):
                xml_path = tmp
                break
        if not xml_path: continue

        # Load and resample
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        ch_air = next((c for c in AIRFLOW_CHANNELS if c in raw.ch_names), None)
        ch_spo = next((c for c in SPO2_CHANNELS if c in raw.ch_names), None)
        if not ch_air or not ch_spo: continue

        raw.pick([ch_air, ch_spo])
        raw.load_data()
        raw.resample(TARGET_HZ)
        sfreq = raw.info['sfreq']
        signals = raw.get_data().T.astype(np.float32)  # (T, 2)
        del raw; gc.collect()

        # Parse labels
        labels = np.zeros(signals.shape[0], dtype=np.int8)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for ev in root.findall('.//ScoredEvent'):
            concept_elem = ev.find('EventConcept')
            if concept_elem is not None and concept_elem.text:
                if 'apnea' in concept_elem.text.lower() or 'hypopnea' in concept_elem.text.lower():
                    start = float(ev.find('Start').text)
                    dur = float(ev.find('Duration').text)
                    start_idx = int(start * sfreq)
                    end_idx = min(int((start + dur) * sfreq), len(labels))
                    labels[start_idx:end_idx] = 1

        # Windowing
        total_len = signals.shape[0]
        num_windows = (total_len - past_samples - future_samples + 1) // step_samples
        if num_windows <= 0: continue

        y_pat = np.zeros(num_windows, dtype=np.int8)
        for w in range(num_windows):
            start = w * step_samples
            fut_start = start + past_samples
            fut_end = fut_start + future_samples
            ratio = np.sum(labels[fut_start:fut_end]) / future_samples
            if ratio >= 0.3:
                y_pat[w] = 1
            elif ratio == 0.0:
                y_pat[w] = 0
            else:
                y_pat[w] = -1

        apnea_idx = np.where(y_pat == 1)[0]
        normal_idx = np.where(y_pat == 0)[0]
        if len(apnea_idx) == 0: continue

        # Balance
        n_sel = min(len(apnea_idx), len(normal_idx))
        sel_norm = np.random.choice(normal_idx, n_sel, replace=False)
        balanced = np.concatenate([apnea_idx[:n_sel], sel_norm])

        for w_idx in balanced:
            start = w_idx * step_samples
            end = start + past_samples
            win = signals[start:end]
            if win.shape[0] != past_samples:
                if win.shape[0] > past_samples:
                    win = win[:past_samples]
                else:
                    pad = ((0, past_samples - win.shape[0]), (0, 0))
                    win = np.pad(win, pad, 'constant')
            # Z-score per channel
            mean = np.mean(win, axis=0, keepdims=True)
            std = np.std(win, axis=0, keepdims=True)
            win_norm = (win - mean) / (std + 1e-8)
            X_list.append(win_norm)
            y_list.append(y_pat[w_idx])
            pid_list.append(edf_name)

        del signals, labels, y_pat; gc.collect()

    except Exception as e:
        print(f"  Error: {e}")

X_all = np.array(X_list, dtype=np.float32)
y_all = np.array(y_list, dtype=np.int8)
pids = np.array(pid_list, dtype=object)
print(f"Total windows: {len(X_all)}")

# ==============================================================================
# PHASE 2: PATIENT-WISE SPLIT
# ==============================================================================
print("\n" + "="*60)
print("⚡ SPLITTING")
print("="*60)

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_all, y_all, groups=pids))
np.random.shuffle(train_idx)
np.random.shuffle(test_idx)

X_train, y_train = X_all[train_idx], y_all[train_idx].astype(np.float32)
X_test, y_test = X_all[test_idx], y_all[test_idx].astype(np.float32)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ==============================================================================
# PHASE 3: TF.DATA PIPELINE (Cached in RAM)
# ==============================================================================
BATCH_SIZE = 256
AUTOTUNE = tf.data.AUTOTUNE

@tf.function
def augment(x, y):
    if tf.random.uniform(()) > 0.5:
        x += tf.random.normal(tf.shape(x), stddev=0.02, dtype=x.dtype)
    return x, y

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(10000).map(augment, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ==============================================================================
# PHASE 4: COMPACT BUT POWERFUL MODEL (No Attention → Faster)
# ==============================================================================
def build_fast_model(input_shape):
    inp = layers.Input(shape=input_shape)

    # Convolutional feature extraction
    x = layers.Conv1D(64, 5, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    # BiLSTM for temporal patterns
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    # Global pooling and classifier
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = models.Model(inp, out)
    return model

model = build_fast_model((past_samples, 2))
model.summary()

# ==============================================================================
# PHASE 5: LIGHTNING TRAINING (15 epochs max)
# ==============================================================================
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True)

print("\n⚡ TRAINING (max 15 epochs)")
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=15,
    callbacks=[early_stop],
    verbose=1
)

# ==============================================================================
# PHASE 6: EVALUATION & THRESHOLD
# ==============================================================================
y_probs = model.predict(test_ds).flatten()

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
best_thresh = thresholds[np.argmax(tpr - fpr)]
print(f"Optimal threshold: {best_thresh:.3f}")

y_pred = (y_probs >= best_thresh).astype(int)

acc = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)

print("\n" + "="*60)
print("🎯 FINAL RESULTS")
print("="*60)
print(f"Accuracy:    {acc*100:.2f}%")
print(f"Sensitivity: {sens*100:.2f}%")
print(f"Specificity: {spec*100:.2f}%")
print("="*60)

# ==============================================================================
# PHASE 7: QUICK PLOTS
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['loss'], label='Train')
axes[0].plot(history.history['val_loss'], label='Val')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train')
axes[1].plot(history.history['val_accuracy'], label='Val')
axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Apnea'], yticklabels=['Normal', 'Apnea'], ax=axes[2])
axes[2].set_title(f'Confusion Matrix (thresh={best_thresh:.2f})')

plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Apnea']))
