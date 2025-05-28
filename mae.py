import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# GPU 메모리 설정
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

print("===== MAE 기반 결측치 처리 및 ANN 모델 =====")

# 1. 기본 데이터 전처리
df_processed = df_new.copy()

# 불리언 변수를 정수형으로 변환
for col in df_processed.columns:
    if df_processed[col].dtype == 'bool':
        df_processed[col] = df_processed[col].astype(int)

# 시간 관련 변수 처리 (날짜/시간 변수를 수치형으로 변환)
for col in df_processed.columns:
    if pd.api.types.is_datetime64_any_dtype(df_processed[col]):
        if not df_processed[col].isna().all():
            reference_date = df_processed[col].min()
            df_processed[col] = (df_processed[col] - reference_date).dt.total_seconds() / (24 * 3600)

# 목표 변수 분리
y = df_processed['transplanted'].values
X_df = df_processed.drop('transplanted', axis=1)

# 범주형 변수 인코딩
categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    mask = X_df[col].notna()
    if mask.sum() > 0:
        X_df.loc[mask, col] = le.fit_transform(X_df.loc[mask, col])
        label_encoders[col] = le

# 모든 열을 숫자형으로 변환
X_numeric = X_df.apply(pd.to_numeric, errors='coerce').values

print(f"데이터 형태: X={X_numeric.shape}, y={y.shape}")
print(f"전체 결측치 개수: {np.isnan(X_numeric).sum()}")
print(f"결측치 비율: {np.isnan(X_numeric).sum() / X_numeric.size * 100:.2f}%")

# 2. 데이터 분할 (결측치 유지)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_numeric, y, test_size=0.3, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

print(f"\n데이터 분할 완료:")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# 3. 개선된 Masked AutoEncoder 클래스
class ImprovedMaskedAutoEncoder:
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3, noise_factor=0.1):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.noise_factor = noise_factor
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self):
        # 입력과 마스크
        input_layer = layers.Input(shape=(self.input_dim,), name='input')
        mask_layer = layers.Input(shape=(self.input_dim,), name='mask')
        
        # 노이즈 추가 (더 강건한 학습을 위해)
        noise = layers.GaussianNoise(self.noise_factor)(input_layer)
        
        # 마스킹 적용
        masked_input = layers.Multiply(name='masked_input')([noise, mask_layer])
        
        # 인코더 (더 깊고 복잡하게)
        encoded = masked_input
        for i, hidden_dim in enumerate(self.hidden_dims):
            encoded = layers.Dense(hidden_dim, activation='relu', name=f'encoder_{i}')(encoded)
            encoded = layers.BatchNormalization(name=f'bn_encoder_{i}')(encoded)
            encoded = layers.Dropout(self.dropout_rate, name=f'dropout_encoder_{i}')(encoded)
        
        # 디코더
        decoded = encoded
        for i, hidden_dim in enumerate(reversed(self.hidden_dims[:-1])):
            decoded = layers.Dense(hidden_dim, activation='relu', name=f'decoder_{i}')(decoded)
            decoded = layers.BatchNormalization(name=f'bn_decoder_{i}')(decoded)
            decoded = layers.Dropout(self.dropout_rate, name=f'dropout_decoder_{i}')(decoded)
        
        # 출력층
        output = layers.Dense(self.input_dim, activation='linear', name='output')(decoded)
        
        # 모델 생성
        self.model = keras.Model([input_layer, mask_layer], output, name='MaskedAutoEncoder')
        
        # 컴파일 (더 나은 최적화를 위해)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss='huber',  # MSE보다 이상치에 강건
            metrics=['mae']
        )
        
    def create_mask(self, X):
        """결측치 마스크 생성 (관측된 값은 1, 결측치는 0)"""
        return (~np.isnan(X)).astype(np.float32)
    
    def prepare_training_data(self, X):
        """학습용 데이터 준비"""
        # 결측치를 0으로 대체 (임시)
        X_filled = np.nan_to_num(X, nan=0.0)
        
        # 표준화
        X_scaled = self.scaler.fit_transform(X_filled)
        
        # 마스크 생성
        mask = self.create_mask(X)
        
        return X_scaled.astype(np.float32), mask.astype(np.float32)
    
    def prepare_inference_data(self, X):
        """추론용 데이터 준비"""
        X_filled = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X_filled)
        mask = self.create_mask(X)
        
        return X_scaled.astype(np.float32), mask.astype(np.float32)
    
    def fit(self, X, epochs=100, batch_size=64, validation_split=0.15, verbose=1):
        """MAE 모델 학습"""
        if self.model is None:
            self.build_model()
        
        X_scaled, mask = self.prepare_training_data(X)
        
        # 콜백 설정
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print("MAE 모델 학습 시작...")
        history = self.model.fit(
            [X_scaled, mask], X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def impute(self, X):
        """결측치 보간"""
        X_scaled, mask = self.prepare_inference_data(X)
        
        # 모델 예측
        reconstructed = self.model.predict([X_scaled, mask], verbose=0)
        
        # 원본 스케일로 복원
        reconstructed_original = self.scaler.inverse_transform(reconstructed)
        
        # 결측치만 보간값으로 대체
        X_imputed = X.copy()
        missing_mask = np.isnan(X)
        X_imputed[missing_mask] = reconstructed_original[missing_mask]
        
        return X_imputed

# 4. MAE 모델 학습 및 결측치 보간
print("\n===== MAE 모델 학습 =====")
mae = ImprovedMaskedAutoEncoder(
    input_dim=X_train.shape[1],
    hidden_dims=[512, 256, 128, 64],  # 더 깊은 네트워크
    dropout_rate=0.3,
    noise_factor=0.05
)

# MAE 학습
mae_history = mae.fit(X_train, epochs=80, batch_size=32, verbose=1)

# 결측치 보간
print("\n결측치 보간 수행 중...")
X_train_imputed = mae.impute(X_train)
X_val_imputed = mae.impute(X_val)
X_test_imputed = mae.impute(X_test)

# 보간 결과 확인
print(f"보간 후 결측치 개수:")
print(f"X_train: {np.isnan(X_train_imputed).sum()}")
print(f"X_val: {np.isnan(X_val_imputed).sum()}")
print(f"X_test: {np.isnan(X_test_imputed).sum()}")

# 5. 최종 데이터 표준화
final_scaler = StandardScaler()
X_train_final = final_scaler.fit_transform(X_train_imputed)
X_val_final = final_scaler.transform(X_val_imputed)
X_test_final = final_scaler.transform(X_test_imputed)

# 6. 클래스 가중치 계산
pos_samples = np.sum(y_train)
neg_samples = len(y_train) - pos_samples
total_samples = len(y_train)

weight_for_0 = (1 / neg_samples) * (total_samples / 2.0)
weight_for_1 = (1 / pos_samples) * (total_samples / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print(f"\n클래스 분포:")
print(f"Class 0 (이식되지 않음): {neg_samples} ({neg_samples/total_samples*100:.2f}%)")
print(f"Class 1 (이식됨): {pos_samples} ({pos_samples/total_samples*100:.2f}%)")
print(f"클래스 가중치: {class_weight}")

# 7. 개선된 ANN 모델
def create_improved_ann_model(input_dim):
    """개선된 ANN 모델 생성"""
    model = keras.Sequential([
        # 첫 번째 블록
        keras.layers.Dense(256, activation='relu', input_dim=input_dim),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        
        # 두 번째 블록
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        # 세 번째 블록
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        # 네 번째 블록
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        
        # 출력층
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def apply_improved_ann_model(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
    """개선된 ANN 모델 학습 및 평가"""
    print("\n===== ANN 모델 학습 및 평가 =====")
    start_time = time.time()
    
    # 모델 생성
    model = create_improved_ann_model(X_train.shape[1])
    
    # 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # 콜백 설정
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # 모델 학습
    print("ANN 모델 학습 중...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    # 예측
    y_val_prob = model.predict(X_val, verbose=0).flatten()
    y_test_prob = model.predict(X_test, verbose=0).flatten()
    
    # 임계값 최적화 (F1 점수 기준)
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        y_val_pred_temp = (y_val_prob >= threshold).astype(int)
        f1_temp = f1_score(y_val, y_val_pred_temp, zero_division=0)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = threshold
    
    print(f"최적 임계값: {best_threshold:.3f} (F1: {best_f1:.4f})")
    
    # 최적 임계값으로 예측
    y_val_pred = (y_val_prob >= best_threshold).astype(int)
    y_test_pred = (y_test_prob >= best_threshold).astype(int)
    
    # 성능 평가
    def evaluate_performance(y_true, y_pred, y_prob, set_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        
        # 민감도 (Sensitivity) = 재현율 (Recall)
        sensitivity = recall
        
        # 특이도 (Specificity)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\n===== {set_name} Set 성능 =====")
        print(f"정확도 (Accuracy): {accuracy:.4f}")
        print(f"정밀도 (Precision): {precision:.4f}")
        print(f"재현율/민감도 (Recall/Sensitivity): {sensitivity:.4f}")
        print(f"특이도 (Specificity): {specificity:.4f}")
        print(f"F1 점수: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    # 성능 평가
    val_results = evaluate_performance(y_val, y_val_pred, y_val_prob, "Validation")
    test_results = evaluate_performance(y_test, y_test_pred, y_test_prob, "Test")
    
    # 실행 시간
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 시각화
    plt.figure(figsize=(20, 10))
    
    # 학습 과정
    plt.subplot(2, 4, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 4, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 혼동 행렬
    plt.subplot(2, 4, 3)
    sns.heatmap(val_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Validation Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.subplot(2, 4, 4)
    sns.heatmap(test_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # ROC 곡선
    from sklearn.metrics import roc_curve
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_prob)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    
    plt.subplot(2, 4, 5)
    plt.plot(fpr_val, tpr_val, label=f'Validation ROC (AUC = {val_results["auc"]:.3f})')
    plt.plot(fpr_test, tpr_test, label=f'Test ROC (AUC = {test_results["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # 성능 지표 비교
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    val_scores = [val_results[m] for m in metrics]
    test_scores = [test_results[m] for m in metrics]
    
    plt.subplot(2, 4, 6)
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, val_scores, width, label='Validation', alpha=0.8)
    plt.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 결과 반환
    results = {
        'model': model,
        'history': history.history,
        'validation': val_results,
        'test': test_results,
        'execution_time': execution_time,
        'best_threshold': best_threshold
    }
    
    return results

# 8. 모델 학습 및 평가
results = apply_improved_ann_model(
    X_train_final, y_train, 
    X_val_final, y_val, 
    X_test_final, y_test, 
    class_weight
)

# 9. 최종 결과 요약
print(f"\n" + "="*60)
print("최종 결과 요약 - MAE + 개선된 ANN")
print("="*60)
print(f"테스트 세트 성능 (최적 임계값: {results['best_threshold']:.3f}):")
print(f"- 정확도: {results['test']['accuracy']:.4f}")
print(f"- 정밀도: {results['test']['precision']:.4f}")
print(f"- 재현율/민감도: {results['test']['sensitivity']:.4f}")
print(f"- 특이도: {results['test']['specificity']:.4f}")
print(f"- F1 점수: {results['test']['f1']:.4f}")
print(f"- AUC: {results['test']['auc']:.4f}")
print(f"- 총 실행 시간: {results['execution_time']:.2f}초")
print("="*60)

# MAE 효과 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(mae_history.history['loss'], label='Training Loss')
plt.plot(mae_history.history['val_loss'], label='Validation Loss')
plt.title('MAE Training Process')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
# 보간 전후 데이터 분포 비교 (첫 번째 특성 기준)
plt.hist(X_train[~np.isnan(X_train[:, 0]), 0], bins=30, alpha=0.5, label='Original (non-missing)', density=True)
plt.hist(X_train_imputed[:, 0], bins=30, alpha=0.5, label='After MAE imputation', density=True)
plt.title('Data Distribution Before/After MAE')
plt.xlabel('Feature Value')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()

print("\n논문 작성을 위한 핵심 포인트:")
print("1. MAE를 통한 지능적 결측치 보간으로 데이터 품질 향상")
print("2. 클래스 불균형 문제 해결을 위한 가중치 적용")
print("3. 최적 임계값 탐색을 통한 F1 점수 최적화")
print("4. 배치 정규화와 드롭아웃을 통한 과적합 방지")
print("5. 조기 종료와 학습률 감소를 통한 안정적 학습")
