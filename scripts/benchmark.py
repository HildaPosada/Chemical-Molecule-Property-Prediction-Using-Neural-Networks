"""Re-evaluate the committed checkpoint and baselines on the saved split.
Run from repository root: python scripts/benchmark.py
The saved split is retrospective; it is not a prospective external validation.
"""
import sys,json,sqlite3,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from rdkit import Chem,RDLogger
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from src.utils import load_config
from src.data import MoleculePreprocessor
from src.models import create_model
RDLogger.DisableLog('rdApp.*')
def main():
 import os
 os.chdir(ROOT)
 config=load_config('config/config_codespaces.yaml');config['training']['device']='cpu'
 pre=MoleculePreprocessor(config);pre.load_scaler('data/processed/scaler.pkl')
 conn=sqlite3.connect('data/processed/molecules.db')
 def features(table):
  df=pd.read_sql_query('SELECT * FROM '+table,conn);out=[];valid=[]
  for i,r in df.iterrows():
   if pre.validate_smiles(r.smiles):
    x=pre.extract_features(r.smiles)
    if x is not None:out.append(x);valid.append(i)
  return df.loc[valid].reset_index(drop=True),np.asarray(out)
 train,Xtrain=features('train');test,Xtest=features('test');ytrain=train.p_np.to_numpy();ytest=test.p_np.to_numpy()
 checkpoint=torch.load('models/checkpoints/best_model.pth',map_location='cpu',weights_only=False);state=checkpoint['model_state_dict']
 config['model']['hidden_sizes']=[state[k].shape[0] for k in sorted(state) if k.startswith('layers.') and k.endswith('.weight')]
 model=create_model(config,Xtest.shape[1]);model.load_state_dict(state);model.eval()
 with torch.no_grad():scores=torch.softmax(model(torch.tensor(pre.scaler.transform(Xtest),dtype=torch.float32)),dim=1)[:,1].numpy()
 baseline=LogisticRegression(max_iter=2000,random_state=42).fit(Xtrain,ytrain)
 dummy=DummyClassifier(strategy='prior').fit(Xtrain,ytrain)
 results={}
 for name,p in [('committed_neural_network',scores),('logistic_regression',baseline.predict_proba(Xtest)[:,1]),('class_prior',dummy.predict_proba(Xtest)[:,1])]:
  pred=(p>=.5).astype(int);results[name]={'accuracy':float(accuracy_score(ytest,pred)),'precision':float(precision_score(ytest,pred,zero_division=0)),'recall':float(recall_score(ytest,pred)),'f1':float(f1_score(ytest,pred)),'roc_auc':float(roc_auc_score(ytest,p)),'confusion_matrix':confusion_matrix(ytest,pred).tolist()}
 canonical=lambda s:Chem.MolToSmiles(Chem.MolFromSmiles(s))
 overlap=set(map(canonical,train.smiles))&set(map(canonical,test.smiles))
 output=ROOT/'experiments/benchmark';output.mkdir(parents=True,exist_ok=True)
 report={'dataset':'Committed BBBP SQLite split','valid_train_rows':len(train),'valid_test_rows':len(test),'canonical_structure_overlap_train_test':len(overlap),'checkpoint_sha256':hashlib.sha256(Path('models/checkpoints/best_model.pth').read_bytes()).hexdigest(),'threshold':.5,'results':results,'limitations':['Retrospective saved split; training history is not independently reconstructed.','Canonical structure overlap can inflate random-split performance.','Model scores are not calibrated probabilities or clinical evidence.']}
 (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 test['bbb_score']=scores;test['predicted_label']=(scores>=.5).astype(int);test['correct']=test.predicted_label==test.p_np
 test.to_csv(output/'predictions.csv',index=False);test[~test.correct].to_csv(output/'errors.csv',index=False)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
