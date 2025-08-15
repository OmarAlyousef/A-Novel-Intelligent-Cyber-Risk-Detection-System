import inspect
inspect.getargspec = inspect.getfullargspec

import os, json, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import spacy

from solcx import install_solc, compile_standard
from web3 import Web3
from web3.exceptions import ContractLogicError

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==== [Cell Separator] =============================================

def deploy_contract():
    install_solc('0.8.7')
    with open('SimpleStorage.sol','r',encoding='utf-8') as f:
        source = f.read()
    compiled = compile_standard({
        'language':'Solidity',
        'sources':{'SimpleStorage.sol':{'content':source}},
        'settings':{'outputSelection':{'*':{'*':['abi','evm.bytecode.object']}}}
    }, solc_version='0.8.7')
    abi = compiled['contracts']['SimpleStorage.sol']['SimpleStorage']['abi']
    bytecode = compiled['contracts']['SimpleStorage.sol']['SimpleStorage']['evm']['bytecode']['object']

    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545", request_kwargs={'timeout':120}))
    if not w3.is_connected():
        raise RuntimeError("Cannot connect to Ganache")
    acct = w3.eth.accounts[0]

    factory = w3.eth.contract(abi=abi, bytecode=bytecode)
    try:
        tx = factory.constructor().transact({
            'from':acct,'gas':3_000_000,'gasPrice':w3.to_wei('1','gwei')
        })
        receipt = w3.eth.wait_for_transaction_receipt(tx)
        address = receipt.contractAddress
    except ContractLogicError as e:
        print("Deployment reverted:", e); import sys; sys.exit(1)

    with open('SimpleStorage_abi.json','w',encoding='utf-8') as f:
        json.dump(abi, f, indent=2)
    with open('contract_address.txt','w',encoding='utf-8') as f:
        f.write(address)

    contract = w3.eth.contract(address=address, abi=abi)
    return w3, contract, acct

# ==== [Cell Separator] =============================================

if not (os.path.exists('SimpleStorage_abi.json') and os.path.exists('contract_address.txt')):
    w3, contract, acct = deploy_contract()
else:
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545", request_kwargs={'timeout':120}))
    if not w3.is_connected(): raise RuntimeError("Cannot connect to Ganache")
    acct = w3.eth.accounts[0]
    with open('SimpleStorage_abi.json','r') as f: abi = json.load(f)
    with open('contract_address.txt','r') as f: addr = f.read().strip()
    contract = w3.eth.contract(address=addr, abi=abi)

DATA_FOLDER = r"C:\Users\omaralyousef\Desktop\Test\archive"
OUTPUT_DIR  = r"C:\Users\omaralyousef\Desktop\Test\ICRDS-blockchain\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Saving outputs to:", OUTPUT_DIR)

# ==== [Cell Separator] =============================================

Saving outputs to: C:\Users\omaralyousef\Desktop\Test\ICRDS-blockchain\outputs

# ==== [Cell Separator] =============================================

TARGET_N = 50000
dfs = []
files = sorted(f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.csv'))
for fn in tqdm(files, desc='Loading data'):
    df0 = pd.read_csv(os.path.join(DATA_FOLDER, fn), sep='|', low_memory=False)
    df0.columns = df0.columns.str.strip()
    if 'label' not in df0.columns:
        df0['label'] = os.path.splitext(fn)[0]
    dfs.append(df0)
df = pd.concat(dfs, ignore_index=True)
if len(df) > TARGET_N:
    df = df.sample(TARGET_N, random_state=42).reset_index(drop=True)

# ==== [Cell Separator] =============================================

Loading data: 100%|████████████████████████████████████████████████████████████████████| 12/12 [01:38<00:00,  8.22s/it]

# ==== [Cell Separator] =============================================

df['label'] = (df['label'].astype(str)
               .str.strip().str.lower()
               .apply(lambda x: 'other' if ('attack' in x or 'c&c' in x) else x))
df = df[df['label']!='other']
meta_df = df[['ts','uid','service','history','label','orig_bytes','resp_bytes']].copy()
df.drop(columns=['ts','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p'],
        errors='ignore', inplace=True)
df.rename(columns={'label':'Label'}, inplace=True)
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
df = df.select_dtypes(include=[np.number]).fillna(0)
X, y = df.drop('Label',axis=1), df['Label']

# ==== [Cell Separator] =============================================

X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
counts = y_tr.value_counts()
rare = counts[counts3000].index.tolist()
X_res, y_res = X_tr.copy(), y_tr.copy()
for cls in rare:
    idx = y_tr[y_tr==cls].index
    if len(idx)>=6:
        sm = SMOTE(random_state=42,k_neighbors=3)
        X_sm,y_sm = sm.fit_resample(X_tr.loc[idx],y_tr.loc[idx])
        X_res = pd.concat([X_res,pd.DataFrame(X_sm,columns=X.columns)],ignore_index=True)
        y_res = pd.concat([y_res,pd.Series(y_sm)],ignore_index=True)
X_tr, y_tr = X_res, y_res

cw = {cls: round(max(y_tr.value_counts())/cnt,2) for cls,cnt in y_tr.value_counts().items()}
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

X_b, X_m, y_b, y_m = train_test_split(X_tr,y_tr,test_size=0.2,stratify=y_tr,random_state=42)
cat = CatBoostClassifier(verbose=0,class_weights=cw,random_seed=42)
cat.fit(X_b,y_b)
meta = LogisticRegression(max_iter=200)
p_m = cat.predict_proba(X_m)
meta.fit(p_m,y_m)
p_te = cat.predict_proba(X_te)
ens_proba = meta.predict_proba(p_te)
ens_pred  = meta.predict(p_te)
conf      = np.max(ens_proba,axis=1)
ens_thr   = ens_pred.copy(); ens_thr[conf0.6] = -1
mask      = ens_thr!=-1

# ==== [Cell Separator] =============================================

rep = classification_report(y_te[mask], ens_thr[mask], output_dict=True, target_names=le.classes_)
df_rep = pd.DataFrame(rep).T
for c in ['precision','recall','f1-score']:
    df_rep[c] = (df_rep[c]*100).round(2)
df_rep['support'] = df_rep['support'].astype(int)
csv_path = os.path.join(OUTPUT_DIR,'classification_report_percent.csv')
df_rep.to_csv(csv_path)
print("Saved:", csv_path)

plt.figure(figsize=(8,4))
sns.heatmap(df_rep.iloc[:-1,:-1], annot=True, fmt=".2f", cmap="Blues")
plt.title("Classification Report (%)")
plt.tight_layout()
png1 = os.path.join(OUTPUT_DIR,'classification_report_percent.png')
plt.savefig(png1); plt.show(); plt.close()
print("Saved:", png1)

# ==== [Cell Separator] =============================================

Saved: C:\Users\omaralyousef\Desktop\Test\ICRDS-blockchain\outputs\classification_report_percent.csv

# ==== [Cell Separator] =============================================

Saved: C:\Users\omaralyousef\Desktop\Test\ICRDS-blockchain\outputs\classification_report_percent.png

# ==== [Cell Separator] =============================================

cm = confusion_matrix(y_te[mask], ens_thr[mask])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.tight_layout()
png2 = os.path.join(OUTPUT_DIR,'confusion_matrix.png')
plt.savefig(png2); plt.show(); plt.close()
print("Saved:", png2)

# ==== [Cell Separator] =============================================

Saved: C:\Users\omaralyousef\Desktop\Test\ICRDS-blockchain\outputs\confusion_matrix.png

# ==== [Cell Separator] =============================================

plt.figure(figsize=(6,5))
for i,cls in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve((y_te==i).astype(int), ens_proba[:,i])
    plt.plot(fpr,tpr,label=f"{cls} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1],'--',color='gray')
plt.title("ROC Curves")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
png3 = os.path.join(OUTPUT_DIR,'roc_curve.png')
plt.savefig(png3); plt.show(); plt.close()
print("Saved:", png3)

# ==== [Cell Separator] =============================================

Saved: C:\Users\omaralyousef\Desktop\Test\ICRDS-blockchain\outputs\roc_curve.png

# ==== [Cell Separator] =============================================

cnts = pd.Series(ens_pred[mask]).value_counts()
plt.figure(figsize=(5,5))
cnts.plot.pie(autopct='%1.1f%%', labels=le.classes_[cnts.index], legend=False)
plt.title("Class Distribution")
plt.tight_layout()
png4 = os.path.join(OUTPUT_DIR,'class_distribution_pie.png')
plt.savefig(png4); plt.show(); plt.close()
print("Saved:", png4)

# ==== [Cell Separator] =============================================

Saved: C:\Users\omaralyousef\Desktop\Test\ICRDS-blockchain\outputs\class_distribution_pie.png

# ==== [Cell Separator] =============================================

hashes = []
for i,idx in enumerate(X_te.index):
    txt = (f"UID:{meta_df.loc[idx,'uid']}|Service:{meta_df.loc[idx,'service']}|"
           f"Score:{conf[i]:.4f}|Entropy:{-np.sum(ens_proba[i]*np.log(ens_proba[i]+1e-12)):.4f}")
    hashes.append(w3.keccak(text=txt))

batch_size = 50
for i in range(0,len(hashes),batch_size):
    batch = hashes[i:i+batch_size]
    tx = contract.functions.addRecords(batch).transact({
        'from':acct,'gas':3_000_000,'gasPrice':w3.to_wei('1','gwei')
    })
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    print("On-chain Tx:", receipt.transactionHash.hex())

# ==== [Cell Separator] =============================================

On-chain Tx: ccaf5b63f0951e824b68a91a21d3d94105a3c6d3d098986bec77104f9aa90846
On-chain Tx: ea8806b03de6a08517777d8d761bf64620b05575dd57addfb033c22a0d473c05
On-chain Tx: a7696bf399927e4aa92eb10496392da72666bd009eed11ffaefc7006af8470a3
On-chain Tx: 6e49362e10299db4e8ad74601af5c39319c9626d401584680780854b29abde4b
On-chain Tx: c645af11c540c2321182b37ab0f563f7fd01bde59653d42f540fdec83640e888
On-chain Tx: 0acfdf3efb489efc4ce95a0e03e89dc0879dda1eef1e6cbd3acbb7219c8db9d6
On-chain Tx: e0ac538a1e71cabdf44730988465123876aa3e82c4e19efc86d6bd3b0fc67d25
On-chain Tx: f75821df3c1565c8cee3b845744e8b0651119f62267836d1be9778b91dc6dda0
On-chain Tx: df0d1e5e1058e379a942e9da198764efdc1f5ccf9fca3134679d1672ed2a8c13
On-chain Tx: f29ebe632a28a0dcbf49f54e5eeff2b8ecf9fe6e4523baca376eb5860fa5087b
On-chain Tx: 3ea0d2a3b69a9857848676e8a6cff12062b931c46059ee94ff4f72aeca69de6a
On-chain Tx: 7de2ef7ac9b4704b16b9bf0e06669a7fa3949789e69e6290c1213078b557f4a9
On-chain Tx: 03cbf4d99f98db0047e22136d4c0bbe7b9da45997195b14af6597a0fcf2e80a7
On-chain Tx: 42e176a87f5887f5f6e78b6b7d7edbebf71eee4a4b10bf3b9a2c6a8ffca7f904
On-chain Tx: 02443a2a784d8082b365d039d25361b8bfa98b0db0917a0ce8835994d332af50
On-chain Tx: 41957afdf4464c907b81b40d8c2304c6a1e160900a7959fd88763b8874b6f198
On-chain Tx: ace136c880342fd004c7199cddff435c8bcd5b72b7ea33fac84749a0ad311c1e
On-chain Tx: b77febd1d99c7033c6c90fde24c59009cf95beccb3970659edf77ec19dae0232
On-chain Tx: 16e520bcc7f79505405312e5774bcf17e2ba2ce4a05ddb491377f5f8ca1a3e32
On-chain Tx: c7db620531f3ccea8ecba0039583e7b7b9a79e293c29758e15c77090e1cd4b31
On-chain Tx: 89def24bbf25a8c5027b084fd57703758e53d826fae845872c2242458a0bff71
On-chain Tx: 1a0137437e6cb896bdfd7385c898fc4628527957e2401c6eae430a7b00d04845
On-chain Tx: 56428fb35d397bce896a4f4c4fa85cd6e6c4542e3b150959d7e13a0ef80d3f86
On-chain Tx: 65610f462a9f709586176b69db3c2862a55406259e632c54d5b6de66485cdfa1
On-chain Tx: d0e3f01bb9da32bf6f7e0fd2a6d8ea29af30eec5a8d60594e767fe212030ed47
On-chain Tx: b09caf741195f536ebaffe763ce2f3e2c19a8ccfc13dc1710e4a9958f0b62baa
On-chain Tx: db66a96895f5af52441f35897e57ca65d173c03942edd3342b432fa6289963d2

# ==== [Cell Separator] =============================================


KeyboardInterrupt

# ==== [Cell Separator] =============================================

recs = []
filt = contract.events.RecordAdded.create_filter(from_block=0)
for ev in filt.get_all_entries():
    recs.append({'hash':ev.args.hash.hex(),'timestamp':ev.args.timestamp})
csv4 = os.path.join(OUTPUT_DIR,'onchain_records.csv')
pd.DataFrame(recs).to_csv(csv4,index=False); print("Saved:",csv4)
json4 = os.path.join(OUTPUT_DIR,'onchain_records.json')
with open(json4,'w',encoding='utf-8') as f: json.dump(recs,f,indent=2)
print("Saved:",json4)

# ==== [Cell Separator] =============================================

texts, ner = [], []
for _,row in meta_df.iterrows():
    texts.append(f"UID:{row['uid']},Service:{row['service']},History:{row['history']},"
                 f"Label:{row['label']},OrigBytes:{row['orig_bytes']},RespBytes:{row['resp_bytes']}")
    if len(texts)>=500: break

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

for i,txt in enumerate(texts):
    doc = nlp(txt)
    ner.append({'index':i,'text':txt,'entities':[(e.text,e.label_) for e in doc.ents]})

json5 = os.path.join(OUTPUT_DIR,'ner_results.json')
with open(json5,'w',encoding='utf-8') as f: json.dump(ner,f,ensure_ascii=False,indent=2)
print("Saved:",json5)
