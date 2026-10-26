"""Reuse the frozen classifier math with an explicit sender-condition admission."""
from common import *
import inspect,numpy as np
SOURCE_SHA='3f40eda1ba9d79ba4412a57560f9f5420b92e7e426b9152c7907f5f7eba4e7e3'
WEIGHT_SHA='983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'
def load_classifier(reg,out,ledger):
 desc=reg['classifier'];source=checked(desc['source']);assert sha(source)==SOURCE_SHA
 weight=checked(desc['weight']);assert sha(weight)==WEIGHT_SHA
 def require(ok,message):
  if not ok:raise ValueError(message)
 def admission(stage,policy_path,policy_sha256):
  assert stage=='sender_predicted_class' and policy_path==reg['revision']['path'] and policy_sha256==reg['revision']['sha256'];checked(reg['revision'])
 ns=dict(np=np,Path=Path,inspect=inspect,require=require,sha=sha,admission=admission,VERSION='frozen_convnext_tiny_sender_predicted_class_v1',WEIGHTS_SHA256=WEIGHT_SHA)
 definitions(source,{'ConvNeXtValidation'},ns)
 classifier=ledger.call('classifier_load','once',lambda:ns['ConvNeXtValidation'](str(weight),WEIGHT_SHA,device='cuda:0',stage='sender_predicted_class',policy_path=reg['revision']['path'],policy_sha256=reg['revision']['sha256']))
 assert len(classifier.names)==1000
 save(out/'classifier_identity.json',dict(original_predict_and_constructor_AST_unchanged=True,source=desc['source'],weight=desc['weight'],identity=classifier.identity,category_order_sha256=digest(classifier.names),label_source='Sender image top-1 prediction; never dataset ground truth',paid_class_bits=10))
 return classifier
