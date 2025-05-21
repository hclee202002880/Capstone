# Capstone

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

# 1. 데이터 불러오기 - Excel 파일로 수정
referrals = pd.read_excel(r"C:\Users\이희창\Downloads\opd.xlsx", engine='openpyxl')
# 또는 확장자가 없는 경우: 
# referrals = pd.read_excel(r"C:\Users\이희창\Downloads\opd", engine='openpyxl')

df = referrals
print(referrals['transplanted'].value_counts())
print(referrals['transplanted'].unique())

# PatientID, HospitalID 및 outcome으로 시작하는 변수들을 원본 데이터프레임(df)에서 제거
outcome_columns = [col for col in df.columns if col.startswith('outcome_')]
columns_to_drop = ['PatientID', 'HospitalID'] + outcome_columns

print("제거할 변수들:", columns_to_drop)

# 원본 데이터프레임에서 선택된 변수들 제거
df = df.drop(columns=columns_to_drop, axis=1)
print(f"변수 제거 후 원본 데이터프레임 크기: {df.shape}")





def get_missing_data(data):
  """
  Returns DataFrame with percent missing data from input data (DataFrame).

  Parameters
  -----
  data (DataFrame): input dataframe

  Returns
  -----
  missing_data (DataFrame): output dataframe with % missing values
  """

  #print(data.isnull().sum()) # uncomment this if you want to see list of counts

  # Get percentage of missing values in each column
  missing_data_prop={}
  for x,y in enumerate(list(data.isnull().sum())):
    missing_data_prop[data.columns[x]]=(float(y/data.shape[0])*100) #"{:.2f}".format

  missing_data=pd.DataFrame(missing_data_prop.items(), columns=['column', 'percent_missing'])
  return missing_data

missing_data=get_missing_data(df)
missing_data

df_new = df.copy()
def total_values(df,col,list_features,label):
  for i in list_features:
    #print(col,i)
    #Change each column value to the new label based on classification framework
    df[col].mask(df[col]==i, label, inplace=True)





infections=['Sepsis','Septic Shock','Infectious Disease - Bacterial','Infectious Disease - Viral',
            'Infectious Disease - Other, specify','Pneumonia','HIV','Hepatitis','AIDS/HIV']
total_values(df_new,'Cause_of_Death_OPO',infections,'Infectious Disease')

# Cardio
cardio=['CHF','CAR - CHF','AAA or thoracic AA', 'AAA - abdominal aortic aneurysm', 'CAR - cardiomegaly/cardiomyopathy/cardiovascular',
        'Pulmonary embolism','PE--Pulmonary Embolism ','Myocardial infarction',
        'CAR - MI', 'CAR - probable MI', 'CAR - arrhythmia',
        'Arrhythmia','Cardiac - Other, specify']
total_values(df_new,'Cause_of_Death_OPO',cardio,'Circulatory Disease')

# Respiratory
resp=['Anoxia','COPD','RES - COPD', 'Respiratory - Other','Respiratory - Other, specify',
      'RES - other', 'RES - pneumonia', 'RES - lung disease', 'RES - asthma',
      'RES - aspiration']
total_values(df_new,'Cause_of_Death_OPO',resp,'Respiratory Disease')

# Newborn/perinatal
newborn=['Fetal Demise','Prematurity','Sudden infant death syndrome',
         'PED - abuse/shaken baby']
total_values(df_new,'Cause_of_Death_OPO',newborn,'Newborn Disease')

# Cancers
cancers=['Leukemia / Lymphoma','Cancer', 'Cancer - Leukemia/Lymphoma','Cancer/Current or within five years']
total_values(df_new,'Cause_of_Death_OPO',cancers,'Cancer')

# Neurological
neuro=['CVA/Stroke - Cerebro Accident','ICB / ICH', 'Cerebrovascular / Stroke',
       'CNS Tumor','SAH','Meningitis','Seizure/Seizure Disorder', 'Aneurysm',
       ]
total_values(df_new,'Cause_of_Death_OPO',neuro,'Nervous Disease')

# Digestive
digestive=['GI - necrotic bowel','GI - bleed','GI - bowel perforation','GI - bowel obstruction']
total_values(df_new,'Cause_of_Death_OPO',digestive,'Digestive Disease')

# Liver
liver=['Liver Disease/Failure','ESLD']
total_values(df_new,'Cause_of_Death_OPO',liver,'Liver Disease')

# Kidney
kidney=['ESRD','Kidney/Renal  Disease']
total_values(df_new,'Cause_of_Death_OPO',kidney,'Kidney Disease')

# Eye
eye=['PED - other', 'PED - premature']
total_values(df_new,'Cause_of_Death_OPO',eye,'Eye Disease')

# Injuries, mostly external
injury=['GSW','TR - GSW','Drowning','Head Trauma','Trauma','Overdose',
        'Drug Overdose/Probable Drug Abuse','An - other', 'An - asphyixiation',
        'An - smoke inhalation','An -  hanging', 'An - drowning',
        'TR - MVA', 'TR - other', 'TR - other', 'TR - CHI - Closed Head Injury',
        'TR - burns', 'TR - stabbing', 'TR - electrocution','Poisoning',
        'Intracranial Hemorrhage','Exsanguination']
total_values(df_new,'Cause_of_Death_OPO',injury,'Injury_External Causes')

# Multisystem
multi=['Multi-system failure', 'MultiSystem Failure']
total_values(df_new,'Cause_of_Death_OPO',multi,'Multi-system failure')

# Other
other=['Other','Other, specify']
total_values(df_new,'Cause_of_Death_OPO',other,'Other')

#Cluster categories: cause of death UNOS

infections=['Sepsis','Infectious Disease - Bacterial','Infectious Disease - Viral','Infectious Disease - Other, specify','Pneumonia','HIV','Hepatitis']
total_values(df_new,'Cause_of_Death_UNOS',infections,'Infectious Disease')

cardio=['CHF','AAA or thoracic AA', 'Pulmonary embolism','Myocardial infarction','Arrhythmia','Cardiac - Other, specify']
total_values(df_new,'Cause_of_Death_UNOS',cardio,'Circulatory Disease')

resp=['Anoxia','COPD','Respiratory - Other','Respiratory - Other, specify']
total_values(df_new,'Cause_of_Death_UNOS',resp,'Respiratory Disease')

newborn=['Fetal Demise','Prematurity','Sudden infant death syndrome']
total_values(df_new,'Cause_of_Death_UNOS',newborn,'Newborn Disease')

cancers=['Leukemia / Lymphoma','Cancer']
total_values(df_new,'Cause_of_Death_UNOS',cancers,'Cancer')

neuro=['CVA/Stroke','ICB / ICH', 'Cerebrovascular / Stroke', 'CNS Tumor','SAH']
total_values(df_new,'Cause_of_Death_UNOS',neuro,'Nervous Disease')

injury=['GSW','Drowning','Head Trauma','Trauma','Overdose',
        'Exsanguination']
total_values(df_new,'Cause_of_Death_UNOS',injury,'Injury_External Causes')

other=['Other','Other, specify']
total_values(df_new,'Cause_of_Death_UNOS',other,'Other')

# Replace names to keep consistent with OPO category change
df_new['Cause_of_Death_UNOS'].replace('ESRD', 'Kidney Disease', inplace=True)
df_new['Cause_of_Death_UNOS'].replace('ESLD', 'Liver Disease', inplace=True)

# Cluster categories: mechanism of death

# Taking only natural causes
natural_causes=['Natural Causes','Death from Natural Causes']
total_values(df_new,'Mechanism_of_Death',natural_causes,'Natural Causes')

# Taking only injuries and external causes: blunt injury, drug intoxication, gunshot wound, asphyxiation, drowning, stab, electrical
injury_external=['Blunt Injury','Drug Intoxication','Gun Shot Wound','Asphyxiation','Drug / Intoxication',
                 'Drowning','Gunshot Wound','Stab','Electrical']
total_values(df_new,'Mechanism_of_Death',injury_external,'Injury_External Causes')

# Taking only nervous system related disorders: stroke, seizure
nervous_diseases=['ICH/Stroke','Intracranial Hemmorrhage / Stroke','Seizure']
total_values(df_new,'Mechanism_of_Death',nervous_diseases,'Nervous Disease')

# None of the above
nofa=['None of the Above','None of the above']
total_values(df_new,'Mechanism_of_Death',nofa,'Other')

# Cluster categories: Circumstances of Death

# Taking only natural causes
natural_causes=['Natural Causes','Death from Natural Causes']
total_values(df_new,'Circumstances_of_Death',natural_causes,'Natural Causes')

# Taking only motor vehicle accidents
mva=['Motor Vehicle Accident','MVA']
total_values(df_new,'Circumstances_of_Death',mva,'Motor Accident')

# Taking only non-motor vehicle accidents
non_mva=['Non-Motor Vehicle Accident','Accident, Non-MVA']
total_values(df_new,'Circumstances_of_Death',non_mva,'Non-motor Accident')

# Suicide - real or alleged
suicide=['Suicide','Alleged Suicide']
total_values(df_new,'Circumstances_of_Death',suicide,'Suicide')

# Homicide - real or alleged
homicide=['Homicide','Alleged Homicide']
total_values(df_new,'Circumstances_of_Death',homicide,'Homicide')

# Child Abuse - real or alleged
child_abuse=['Child Abuse','Alleged Child Abuse']
total_values(df_new,'Circumstances_of_Death',child_abuse,'Homicide')

# Other/none of the above
other=['Other','None of the Above']
total_values(df_new,'Circumstances_of_Death',other,'Other')



#Feature engineering: dealing with time
def get_duration_between_dates(then, now, interval = "default"):

    """
    Returns a duration as specified by variable interval.
    Used to calculate new feature of time authorized - time approached.

    Code source: https://stackoverflow.com/questions/1345827/how-do-i-find-the-time-difference-between-two-datetime-objects-in-python

    Parameters
    ----------
    then (DateTime): a date-time.
    now (DateTime): another date-time.
    interval (string): type of duration metric, e.g. minutes.

    Returns
    -------
    (float): A float with the duration in interval units.
    """

    duration = now - then # For build-in functions
    duration_in_s = duration.total_seconds()

    def years():
      return divmod(duration_in_s, 31536000) # Seconds in a year=31536000.

    def days(seconds = None):
      return divmod(seconds if seconds != None else duration_in_s, 86400) # Seconds in a day = 86400

    def hours(seconds = None):
      return divmod(seconds if seconds != None else duration_in_s, 3600) # Seconds in an hour = 3600

    def minutes(seconds = None):
      return divmod(seconds if seconds != None else duration_in_s, 60) # Seconds in a minute = 60

    def seconds(seconds = None):
      if seconds != None:
        return divmod(seconds, 1)
      return duration_in_s

    def totalDuration():
        y = years()
        d = days(y[1]) # Use remainder to calculate next variable
        h = hours(d[1])
        m = minutes(h[1])
        s = seconds(m[1])

        return "Time between dates: {} years, {} days, {} hours, {} minutes and {} seconds".format(int(y[0]), int(d[0]), int(h[0]), int(m[0]), int(s[0]))

    return {
        'years': float(years()[0]),
        'days': float(days()[0]),
        'hours': float(hours()[0]),
        'minutes': float(minutes()[0]),
        'seconds': float(seconds()),
        'default': totalDuration()
    }


def create_time_column(df,col1,col2,new_col_name):
  """
  Create new column to describe the number of hours between administrative milestones,
  e.g. between referral (death) and approach.

  Parameters
  ----------
  df (DataFrame): input data.
  col1 (string): name of column representing one timepoint.
  col2 (string): name of column representing another timepoint.
  new_col_name (string): new column name representing a time category between time points.

  Returns
  -------
  df (DataFrame): modified df with new column.

  """
  def convert_datetime(str1,str2):
    # Helper function to convert to datetime
    return [pd.to_datetime(str1), pd.to_datetime(str2)]

  time_category = []
  for row in zip(df[col1], df[col2]):
    if pd.isnull(row[0])==False and pd.isnull(row[1])==False:
      date_row=convert_datetime(row[0],row[1])
      time_elapsed=abs(get_duration_between_dates(date_row[0],date_row[1])['hours'])

      if time_elapsed <= 24:
        time_category.append('Within 24 hours')

      if time_elapsed > 24:
        time_category.append('Over 24 hours')

    else:
      time_category.append('Milestone not reached')

  df[new_col_name]=time_category

  return df


# Define timepoint variables
time_vars = ['time_asystole','time_brain_death','time_referred', 'time_approached', 'time_authorized', 'time_procured']

# Get category of intervals between them
asystole_to_referred = 'time_asystole_to_referred'
df_new = create_time_column(df_new,time_vars[0],time_vars[2], asystole_to_referred)

brain_death_to_referred = 'time_brain_death_to_referred'
df_new = create_time_column(df_new,time_vars[1],time_vars[2], brain_death_to_referred)

referred_to_approached = 'time_referred_to_approached'
df_new = create_time_column(df_new,time_vars[2],time_vars[3], referred_to_approached)

approached_to_authorized = 'time_approached_to_authorized'
df_new = create_time_column(df_new,time_vars[3],time_vars[4], approached_to_authorized)

authorized_to_procured = 'time_authorized_to_procured'
df_new = create_time_column(df_new,time_vars[4],time_vars[5], authorized_to_procured)


def get_missing_data(data):
  """
  Returns DataFrame with percent missing data from input data (DataFrame).

  Parameters
  -----
  data (DataFrame): input dataframe

  Returns
  -----
  missing_data (DataFrame): output dataframe with % missing values
  """

  #print(data.isnull().sum()) # uncomment this if you want to see list of counts

  # Get percentage of missing values in each column
  missing_data_prop={}
  for x,y in enumerate(list(data.isnull().sum())):
    missing_data_prop[data.columns[x]]=(float(y/data.shape[0])*100) #"{:.2f}".format

  missing_data=pd.DataFrame(missing_data_prop.items(), columns=['column', 'percent_missing'])
  return missing_data

missing_data=get_missing_data(df_new)
missing_data

cols_large_missing=list(missing_data[missing_data['percent_missing']>50]['column'])
print(f'{len(cols_large_missing)} columns to drop due to over 50% missing')
cols_large_missing #over 50% missing



df_new2=df_new.copy()
# Drop time variables and keep (some of) the time interval variables
cols_large_missing.remove('Cause_of_Death_OPO') # Keep this as it is still domain relevant
cols_large_missing.remove('time_brain_death')
cols_large_missing.remove('time_approached')
cols_large_missing.remove('time_authorized')
df_new = df_new.drop(cols_large_missing,axis=1) # drop Procured_Year as it is almost perfectly collinear with Referral_Year (0.98)
   

# Make this copy before we remove collinear variables >0.8
df_new_with_collinear=df_new.copy()
cols_collinear = ['brain_death','time_referred','time_asystole','authorized','procured','time_approached_to_authorized','time_authorized_to_procured']
df_new = df_new.drop(cols_collinear,axis=1)

print(len(df_new.columns))
df_new.columns


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 원본 데이터에서 의심 변수 제거
df_clean = df_new.copy()
#df_clean = df_clean.drop(columns=leakage_vars)


# 데이터 유형별로 컬럼 분류하는 함수
def categorize_columns(df):
    """데이터프레임의 컬럼을 유형별로 분류"""
    categorical_cols = []
    numerical_cols = []
    datetime_cols = []
    binary_cols = []
    
    for col in df.columns:
        if col == 'transplanted':
            continue
        
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 2:
                binary_cols.append(col)
            else:
                numerical_cols.append(col)
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            categorical_cols.append(col)
    
    return categorical_cols, numerical_cols, datetime_cols, binary_cols

# 데이터프레임 복사
df_processed = df_new.copy()

# 불리언 변수를 정수형으로 변환
for col in df_processed.columns:
    if df_processed[col].dtype == 'bool':
        df_processed[col] = df_processed[col].astype(int)

# 시간 관련 변수 처리 (날짜/시간 변수를 수치형으로 변환)
for col in df_processed.columns:
    if pd.api.types.is_datetime64_any_dtype(df_processed[col]):
        # 기준 날짜 선택 (예: 데이터셋의 최소 날짜)
        if not df_processed[col].isna().all():  # 모든 값이 NaN이 아닌 경우만
            reference_date = df_processed[col].min()
            # 날짜를 일수로 변환
            df_processed[col] = (df_processed[col] - reference_date).dt.total_seconds() / (24 * 3600)
            # NaN 값을 0으로 대체
            df_processed[col] = df_processed[col].fillna(0)

# 목표 변수 분리
X_df = df_processed.drop('transplanted', axis=1)
y = df_processed['transplanted']

# 변수 유형 분류
categorical_cols, numerical_cols, datetime_cols, binary_cols = categorize_columns(df_processed)

print(f"범주형 변수: {len(categorical_cols)}개")
print(f"수치형 변수: {len(numerical_cols)}개")
print(f"날짜/시간 변수(수치형으로 변환됨): {len(datetime_cols)}개")
print(f"이진 변수: {len(binary_cols)}개")

# 날짜/시간 변수는 이미 수치형으로 변환되었으므로 수치형 변수에 추가
numerical_cols.extend(datetime_cols)

# 결측치 처리 및 정규화를 위한 전처리 파이프라인 생성
transformers = []

if numerical_cols:
    transformers.append(('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numerical_cols))

if categorical_cols:
    transformers.append(('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_cols))

if binary_cols:
    transformers.append(('bin', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ]), binary_cols))

# 모든 변수가 맞게 분류되었는지 확인
all_cols = categorical_cols + numerical_cols + binary_cols
missing_cols = [col for col in X_df.columns if col not in all_cols]
if missing_cols:
    print(f"분류되지 않은 컬럼: {missing_cols}")
    print("이 컬럼들은 그대로 유지됩니다.")

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder='passthrough'  # 분류되지 않은 열도 유지
)

# 전처리 적용
print("전처리 파이프라인 적용 중...")
X_preprocessed = preprocessor.fit_transform(X_df)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.3, random_state=42
)

# 훈련 및 검증 세트 분할
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"전처리 완료!")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# 클래스 불균형 처리를 위한 가중치 계산
pos_samples = np.sum(y)  # True(1) 샘플 수
neg_samples = len(y) - pos_samples  # False(0) 샘플 수
total_samples = len(y)

# 가중치 계산
weight_for_0 = (1 / neg_samples) * (total_samples / 2.0)
weight_for_1 = (1 / pos_samples) * (total_samples / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print(f"\n클래스 분포:")
print(f"True (이식됨) 샘플 수: {pos_samples} ({pos_samples/total_samples*100:.2f}%)")
print(f"False (이식되지 않음) 샘플 수: {neg_samples} ({neg_samples/total_samples*100:.2f}%)")
print(f"\n클래스 가중치:")
print(f"클래스 0(이식되지 않음)의 가중치: {weight_for_0:.4f}")
print(f"클래스 1(이식됨)의 가중치: {weight_for_1:.4f}")





import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

def apply_ann_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    전처리된 데이터에 ANN 모델을 적용하고 성능을 평가합니다.
    """
    print("ANN 모델 학습 및 평가 시작...")
    start_time = time.time()
    
    # 입력 차원 계산
    input_dim = X_train.shape[1]
    
    # 모델 구성
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Early Stopping 설정
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # 모델 학습
    print("모델 학습 중...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 검증 세트 예측
    y_val_prob = model.predict(X_val).flatten()
    y_val_pred = (y_val_prob >= 0.5).astype(int)
    
    # 테스트 세트 예측
    y_test_prob = model.predict(X_test).flatten()
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    
    # 성능 평가
    # 검증 세트 성능
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_prob)
    
    # 테스트 세트 성능
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_prob)
    
    # 실행 시간 계산
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 결과 요약
    results = {
        'model_summary': model.summary(),
        'history': history.history,
        'validation': {
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'auc': val_auc,
            'confusion_matrix': confusion_matrix(y_val, y_val_pred),
            'classification_report': classification_report(y_val, y_val_pred, output_dict=True)
        },
        'test': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'auc': test_auc,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
        },
        'execution_time': execution_time
    }
    
    # 결과 출력
    print("\n===== ANN 모델 성능 =====")
    print(f"실행 시간: {execution_time:.2f}초")
    
    print("\n검증 세트 성능:")
    print(f"정확도: {val_accuracy:.4f}")
    print(f"정밀도: {val_precision:.4f}")
    print(f"재현율: {val_recall:.4f}")
    print(f"F1 점수: {val_f1:.4f}")
    print(f"AUC: {val_auc:.4f}")
    
    print("\n테스트 세트 성능:")
    print(f"정확도: {test_accuracy:.4f}")
    print(f"정밀도: {test_precision:.4f}")
    print(f"재현율: {test_recall:.4f}")
    print(f"F1 점수: {test_f1:.4f}")
    print(f"AUC: {test_auc:.4f}")
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(results['validation']['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('검증 세트 혼동 행렬')
    plt.xlabel('예측')
    plt.ylabel('실제')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(results['test']['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('테스트 세트 혼동 행렬')
    plt.xlabel('예측')
    plt.ylabel('실제')
    
    plt.tight_layout()
    plt.show()
    
    # 학습 과정 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='훈련 손실')
    plt.plot(history.history['val_loss'], label='검증 손실')
    plt.title('학습 과정 - 손실')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='훈련 정확도')
    plt.plot(history.history['val_accuracy'], label='검증 정확도')
    plt.title('학습 과정 - 정확도')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, results

# 하이퍼파라미터 튜닝용 함수 
def tune_ann_model(X_train, y_train, X_val, y_val):
    """
    다양한 하이퍼파라미터 조합으로 ANN 모델을 튜닝합니다.
    """
    print("ANN 모델 하이퍼파라미터 튜닝 시작...")
    input_dim = X_train.shape[1]
    best_val_auc = 0
    best_params = {}
    
    # 하이퍼파라미터 조합
    learning_rates = [0.01, 0.001, 0.0005]
    first_layer_sizes = [32, 64, 128]
    dropout_rates = [0.2, 0.3, 0.4]
    
    for lr in learning_rates:
        for first_layer in first_layer_sizes:
            for dropout in dropout_rates:
                print(f"\n테스트 중: lr={lr}, first_layer={first_layer}, dropout={dropout}")
                
                # 모델 구성
                model = Sequential([
                    Dense(first_layer, activation='relu', input_dim=input_dim),
                    Dropout(dropout),
                    Dense(first_layer // 2, activation='relu'),
                    Dropout(dropout),
                    Dense(first_layer // 4, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                
                # 모델 컴파일
                model.compile(
                    optimizer=Adam(learning_rate=lr),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Early Stopping 설정
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=0
                )
                
                # 모델 학습
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # 검증 세트 성능 평가
                y_val_prob = model.predict(X_val).flatten()
                val_auc = roc_auc_score(y_val, y_val_prob)
                
                print(f"검증 AUC: {val_auc:.4f}")
                
                # 최적 파라미터 업데이트
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_params = {
                        'learning_rate': lr,
                        'first_layer_size': first_layer,
                        'dropout_rate': dropout
                    }
    
    print("\n===== 최적 하이퍼파라미터 =====")
    print(f"최적 파라미터: {best_params}")
    print(f"최적 검증 AUC: {best_val_auc:.4f}")
    
    return best_params




# ANN 모델을 위한 하이퍼파라미터 튜닝 (선택적)
best_params = tune_ann_model(X_train, y_train, X_val, y_val)

# 최적 하이퍼파라미터로 모델 학습 및 평가
# 또는 기본 파라미터로 바로 모델 학습
best_ann_model, ann_results = apply_ann_model(X_train, y_train, X_val, y_val, X_test, y_test)

# 결과 확인
print(f"\nANN 모델 테스트 세트 정확도: {ann_results['test']['accuracy']:.4f}")
print(f"ANN 모델 테스트 세트 AUC: {ann_results['test']['auc']:.4f}")

# KNN과 성능 비교
print("\n===== KNN vs ANN 성능 비교 =====")
print(f"KNN 테스트 정확도: {knn_results['test']['accuracy']:.4f}, AUC: {knn_results['test']['auc']:.4f}")
print(f"ANN 테스트 정확도: {ann_results['test']['accuracy']:.4f}, AUC: {ann_results['test']['auc']:.4f}")
      
