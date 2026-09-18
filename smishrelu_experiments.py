# ============================================================
# SMishReLU: A Continuous C1-Smooth Hybrid Activation Function
# Experimental Code
#
# This script extends the original MishReLU experimental
# framework by adding the proposed SMishReLU activation
# function and comparing it against MishReLU and standard
# baselines (ReLU, Mish, ELU, LeakyReLU, SELU).
#
# Run on Kaggle Notebooks with GPU enabled.
# ============================================================

# ============================================================
# Import required libraries for data processing, modeling,
# evaluation metrics, statistical tests, and reproducibility
# ============================================================
import os, time, random
import numpy as np
import tensorflow as tf
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Activation, Bidirectional
from sklearn.metrics import f1_score, classification_report
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Add, Input, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical


# ======================================================================================
# MNIST Dataset Preparation
# ======================================================================================
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()

X_train_mnist = X_train_mnist.astype("float32") / 255.0
X_test_mnist  = X_test_mnist.astype("float32") / 255.0

X_train_mnist = X_train_mnist.reshape(-1, 28, 28, 1)
X_test_mnist  = X_test_mnist.reshape(-1, 28, 28, 1)

y_train_mnist = to_categorical(y_train_mnist, 10)
y_test_mnist  = to_categorical(y_test_mnist, 10)

X_train_mnist, X_val_mnist, y_train_mnist, y_val_mnist = train_test_split(
    X_train_mnist, y_train_mnist,
    test_size=0.2,
    random_state=42,
    stratify=y_train_mnist
)


# ======================================================================================
# Fashion-MNIST Dataset Preparation
# ======================================================================================
(X_train_fm, y_train_fm), (X_test_fm, y_test_fm) = fashion_mnist.load_data()

X_train_fm = X_train_fm.astype("float32") / 255.0
X_test_fm  = X_test_fm.astype("float32") / 255.0

X_train_fm = X_train_fm.reshape(-1, 28, 28, 1)
X_test_fm  = X_test_fm.reshape(-1, 28, 28, 1)

y_train_fm = to_categorical(y_train_fm, 10)
y_test_fm  = to_categorical(y_test_fm, 10)

X_train_fm, X_val_fm, y_train_fm, y_val_fm = train_test_split(
    X_train_fm, y_train_fm,
    test_size=0.2,
    random_state=42,
    stratify=y_train_fm
)


# =============================
# CIFAR-100 (fixed: consistent capitalized variable names)
# =============================
(X_train_c100, y_train_c100), (X_test_c100, y_test_c100) = cifar100.load_data(label_mode='fine')

X_train_c100 = X_train_c100.astype("float32") / 255.0
X_test_c100  = X_test_c100.astype("float32") / 255.0

y_train_c100 = to_categorical(y_train_c100, 100)
y_test_c100  = to_categorical(y_test_c100, 100)

X_train_c100, X_val_c100, y_train_c100, y_val_c100 = train_test_split(
    X_train_c100, y_train_c100,
    test_size=0.2,
    random_state=42
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(X_train_c100)

train_generator_c100 = datagen.flow(
    X_train_c100,
    y_train_c100,
    batch_size=256
)


# ======================================================================================
# IMDB Dataset Preparation
# NOTE: Requires the "IMDB Dataset of 50K Movie Reviews" dataset attached on Kaggle.
# ======================================================================================
imdb_data = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

imdb_data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)

train_imdb_data, test_imdb_data = train_test_split(
    imdb_data, test_size=0.2, random_state=42, stratify=imdb_data["sentiment"]
)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_imdb_data["review"])

x_train_imdb_data = pad_sequences(tokenizer.texts_to_sequences(train_imdb_data["review"]), maxlen=200)
x_test_imdb_data = pad_sequences(tokenizer.texts_to_sequences(test_imdb_data["review"]), maxlen=200)

y_train_imdb_data = train_imdb_data["sentiment"].to_numpy()
y_test_imdb_data = test_imdb_data["sentiment"].to_numpy()

x_train_imdb_data, x_val_imdb_data, y_train_imdb_data, y_val_imdb_data = train_test_split(
    x_train_imdb_data, y_train_imdb_data,
    test_size=0.2, random_state=42, stratify=y_train_imdb_data
)


# ======================================================================================
# Reuters Newswire Dataset Preparation
# (fixed: correct variable names, uses pad_sequences already imported above)
# ======================================================================================
(x_train_reuters, y_train_reuters), (x_test_reuters, y_test_reuters) = reuters.load_data(num_words=10000)

max_words = 200
x_train_reuters = pad_sequences(x_train_reuters, maxlen=max_words)
x_test_reuters = pad_sequences(x_test_reuters, maxlen=max_words)

y_train_reuters = to_categorical(y_train_reuters, 46)
y_test_reuters = to_categorical(y_test_reuters, 46)

x_train_reuters, x_val_reuters, y_train_reuters, y_val_reuters = train_test_split(
    x_train_reuters, y_train_reuters,
    test_size=0.2,
    random_state=42,
    stratify=y_train_reuters
)


# ============================================================
# List all available built-in activation functions in TensorFlow
# ============================================================
activation_functions_builtin = dir(tf.keras.activations)
activation_functions_builtin = [f for f in activation_functions_builtin if not f.startswith('__')]
print(activation_functions_builtin)


# ============================================================
# Definition and registration of custom and baseline
# activation functions used in the experiments
# ============================================================

# Baseline: MishReLU (from the original paper being extended)
def MishRelU(x):
    return tf.where(x > 0, x, x * tf.keras.activations.tanh(tf.keras.activations.softplus(x)))
get_custom_objects()['MishRelU'] = MishRelU

# ============================================================
# PROPOSED: SMishReLU (Smooth MishReLU)
# Resolves MishReLU's first-derivative discontinuity at x = 0
# by scaling the negative branch with alpha = 1 / tanh(ln 2) ~ 1.6667
# so that the left-hand and right-hand derivatives at x = 0 both equal 1.
# ============================================================
ALPHA_SMISH = 1.0 / np.tanh(np.log(2.0))  # ~ 1.666667

def SMishReLU(x):
    mish_part = ALPHA_SMISH * x * tf.keras.activations.tanh(tf.keras.activations.softplus(x))
    return tf.where(x > 0, x, mish_part)
get_custom_objects()['SMishReLU'] = SMishReLU

# Baseline ReLU
def ReLU(x):
    return tf.keras.activations.relu(x)
get_custom_objects()['ReLU'] = ReLU

# Baseline Mish
def Mish(x):
    return x * tf.keras.activations.tanh(tf.keras.activations.softplus(x))
get_custom_objects()['Mish'] = Mish

# Baseline Elu
def Elu(x, a=1):
    return tf.keras.activations.elu(x, alpha=a)
get_custom_objects()['Elu'] = Elu

# Baseline LeakyReLU
def LeakyReLU(x):
    return tf.keras.layers.LeakyReLU(alpha=0.01)(x)
get_custom_objects()['LeakyReLU'] = LeakyReLU

# Baseline Selu
def Selu(x):
    return tf.keras.activations.selu(x)
get_custom_objects()['Selu'] = Selu


# ======================================================================================
# Reproducibility Function:
# Fixes random seeds across Python, NumPy, TensorFlow, and the OS environment
# ======================================================================================
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# Multi-Layer Perceptron (MLP) model for Fashion-MNIST
# ============================================================
def build_fashion_model_MLP(activation, optimizer_name='adam', learning_rate=0.001):
    model = keras.Sequential()
    model.add(layers.Input(shape=(28, 28)))
    model.add(layers.Flatten())

    model.add(layers.Dense(units=398))
    model.add(layers.Activation(activation))
    model.add(BatchNormalization())
    model.add(layers.Dropout(rate=0.1))

    model.add(layers.Dense(units=128))
    model.add(layers.Activation(activation))
    model.add(BatchNormalization())
    model.add(layers.Dropout(rate=0.1))

    model.add(layers.Dense(units=64))
    model.add(layers.Activation(activation))
    model.add(BatchNormalization())
    model.add(layers.Dropout(rate=0.1))

    model.add(layers.Dense(units=10, activation='softmax'))

    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# ============================================================
# Convolutional Neural Network (CNN) model for MNIST
# ============================================================
def build_mnist_model_cnn3(activation, optimizer_name='adam', learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))

    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.Activation(activation))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.Activation(activation))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.Activation(activation))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation(activation))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(layers.Dense(10, activation='softmax'))

    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# ======= ResNet-18 architecture adapted for CIFAR-100 dataset ========
def resnet_block(x, filters, activation_fc='relu', stride=1):
    shortcut = x

    x = Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_fc)(x)

    x = Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation(activation_fc)(x)
    return x


def ResNet18_CIFAR(input_shape=(32, 32, 3), num_classes=100, activation_fc='relu'):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, 3, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation_fc)(x)

    x = resnet_block(x, 64, activation_fc=activation_fc)
    x = resnet_block(x, 64, activation_fc=activation_fc)

    x = resnet_block(x, 128, activation_fc=activation_fc, stride=2)
    x = resnet_block(x, 128, activation_fc=activation_fc)

    x = resnet_block(x, 256, activation_fc=activation_fc, stride=2)
    x = resnet_block(x, 256, activation_fc=activation_fc)

    x = resnet_block(x, 512, activation_fc=activation_fc, stride=2)
    x = resnet_block(x, 512, activation_fc=activation_fc)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation=activation_fc)(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

    return Model(inputs, outputs)


def build_model(input_shape=(32, 32, 3), activation_fc='relu', learning_rate=0.001, num_classes=100):
    model = ResNet18_CIFAR(input_shape=input_shape, num_classes=num_classes, activation_fc=activation_fc)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ======================================================================================
# Early Stopping Callback:
# Prevents overfitting by monitoring validation performance
# ======================================================================================
from tensorflow.keras.callbacks import EarlyStopping

early_stop_cifar100 = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


# ======= LSTM MODEL For IMDB Dataset =======
def build_lstm_model(activation, optimizer_name, learning_rate):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(32, activation=activation, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'adamw':
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer_name == 'radam':
        try:
            optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
        except:
            raise ValueError("RAdam requires tensorflow-addons installed.")
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ======= BiLSTM MODEL for Reuters =======
def BiLSTM_model(activation, optimizer_name, learning_rate):
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=32, input_length=200))
    model.add(Bidirectional(LSTM(32, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation(activation))
    model.add(Dropout(0.2))
    model.add(Dense(46, activation='softmax'))

    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'adamw':
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer_name == 'radam':
        try:
            optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
        except:
            raise ValueError("RAdam requires tensorflow-addons installed.")
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ====================================
# Model training and evaluation (MNIST / CNN)
# - Multiple random seeds
# - Different learning rates
# - Mean and standard deviation are reported
# ====================================
SEEDS = [164, 343, 865, 599, 251]
all_results = []

# SMishReLU is included as the proposed activation function, alongside MishReLU
# (baseline from the original paper) and standard activations.
activation_functions = [SMishReLU, MishRelU, ReLU, Mish, Elu, LeakyReLU, Selu]
learning_rates = [0.01, 0.001, 0.0001]
optimizers = ["nadam"]

for activation in activation_functions:
    act_name = activation.__name__

    for optimizer_name in optimizers:
        for learning_rate in learning_rates:

            print(f"\n>>> Training with {act_name}, {optimizer_name}, lr={learning_rate}")
            print("Seeds for this run:", SEEDS)

            accs = []
            precisions = []
            recalls = []
            f1s = []

            start = time.time()

            for seed in SEEDS:
                set_seed(seed)

                model = build_mnist_model_cnn3(
                    activation,
                    optimizer_name=optimizer_name,
                    learning_rate=learning_rate
                )

                history = model.fit(
                    X_train_mnist, y_train_mnist,
                    epochs=30,
                    validation_split=0.2,
                    batch_size=60,
                    verbose=0,
                )

                test_loss, test_acc = model.evaluate(X_test_mnist, y_test_mnist, verbose=0)
                accs.append(test_acc)

                y_pred_probs = model.predict(X_test_mnist, verbose=0)
                y_pred = np.argmax(y_pred_probs, axis=1)
                y_true = np.argmax(y_test_mnist, axis=1)

                precisions.append(precision_score(y_true, y_pred, average='macro'))
                recalls.append(recall_score(y_true, y_pred, average='macro'))
                f1s.append(f1_score(y_true, y_pred, average='macro'))

            end = time.time()
            total_minutes = (end - start) / 60

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)

            mean_prec = np.mean(precisions)
            std_prec = np.std(precisions)

            mean_rec = np.mean(recalls)
            std_rec = np.std(recalls)

            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)

            print(f"{act_name}, lr={learning_rate} | "
                  f"Acc: {mean_acc:.4f}\u00b1{std_acc:.4f} | "
                  f"Prec: {mean_prec:.4f}\u00b1{std_prec:.4f} | "
                  f"Rec: {mean_rec:.4f}\u00b1{std_rec:.4f} | "
                  f"F1: {mean_f1:.4f}\u00b1{std_f1:.4f} "
                  f"({len(SEEDS)} seeds) | Time: {total_minutes:.2f} min")

            all_results.append({
                "activation": act_name,
                "optimizer": optimizer_name,
                "lr": learning_rate,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_prec": mean_prec,
                "std_prec": std_prec,
                "mean_rec": mean_rec,
                "std_rec": std_rec,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "all_accs": accs,
                "all_precs": precisions,
                "all_recs": recalls,
                "all_f1s": f1s
            })


# ============================================================
# Aggregate results across runs and convert to DataFrame
# ============================================================
df_results = pd.DataFrame(all_results)
print("\n=== Summary Results ===")
print(df_results[[
    "activation", "lr",
    "mean_acc", "std_acc",
    "mean_prec", "std_prec",
    "mean_rec", "std_rec",
    "mean_f1", "std_f1"
]])


# ============================================================
# Statistical significance analysis
# Friedman test followed by post-hoc Wilcoxon test
# SMishReLU is included in the comparison set.
# ============================================================
activations = ["SMishReLU", "MishRelU", "ReLU", "Mish", "Elu", "LeakyReLU", "Selu"]
learning_rates = [0.01, 0.001, 0.0001]
metrics = ["all_accs", "all_precs", "all_recs", "all_f1s"]

for lr in learning_rates:
    print(f"\n=== Friedman Test for lr = {lr} ===")
    for metric in metrics:
        values = [df_results.query(f"activation=='{act}' and lr=={lr}")[metric].values[0] for act in activations]
        stat, p = friedmanchisquare(*values)
        print(f"{metric.replace('all_', '').capitalize()} -> Friedman statistic = {stat:.4f}, p-value = {p:.6f}")


# ========== Post-hoc Wilcoxon Test =============
activations = ["SMishReLU", "MishRelU", "ReLU", "Mish", "Elu", "LeakyReLU", "Selu"]
metrics = ["all_accs", "all_precs", "all_recs", "all_f1s"]
lr = 0.0001


def get_values(act, metric):
    return df_results.query(f"activation=='{act}' and lr=={lr}")[metric].values[0]


for metric in metrics:
    print("\n===================================")
    print("Post-hoc Wilcoxon Test | Metric:", metric.replace("all_", "").upper())
    print("===================================\n")

    data = {act: get_values(act, metric) for act in activations}
    avg_scores = {act: np.mean(vals) for act, vals in data.items()}

    print("Average Scores:")
    for act, score in avg_scores.items():
        print(f"{act}: {score:.4f}")

    best_act = max(avg_scores, key=avg_scores.get)
    print(f"\n==> BEST activation (based on mean): {best_act}\n")

    print("Wilcoxon Pairwise p-values:")
    for i in range(len(activations)):
        for j in range(i + 1, len(activations)):
            act1 = activations[i]
            act2 = activations[j]
            stat, p = wilcoxon(data[act1], data[act2])
            print(f"{act1} vs {act2}: p = {p:.6f}")
