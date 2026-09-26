import copy
import unittest
from collections import OrderedDict
import torch
from tools.publish_c_pure_milestone import (
    ARM, validate_pure_rows, validate_completion, validate_parent_payloads, same_tree,
)

class PurePublicationTests(unittest.TestCase):
    def row(self):
        return dict(method=ARM,source_index='0',image_id='source',snr_db='1',seed='4101',
                    mse='.01',lpips_alex='.2',normalized_latent='.5',utility='.035',
                    U_image='.03',header_ok='1',body_crc_ok='not_applicable')

    def validate(self,rows):
        validate_pure_rows(rows,[{'image_id':'source'}],[1],[4101])

    def test_pure_has_no_crc_and_unmasked_latent(self):
        self.validate([self.row()])
        for change in ({'body_crc_ok':'1'},{'body_crc_ok':'0'},{'header_ok':'0'},
                       {'utility':'.03'},{'method':'H8-V'}):
            with self.assertRaises(ValueError):
                self.validate([dict(self.row(),**change)])

    def test_complete_source_noise_grid(self):
        for rows in ([],[self.row(),self.row()],[dict(self.row(),image_id='wrong')],
                     [dict(self.row(),seed='4102')]):
            with self.assertRaises(ValueError):
                self.validate(rows)

    def done(self):
        return dict(status='REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE',
                    registration_sha256='r',state=dict(step=20000,last_full=20000,updates={ARM:20000}),
                    selected={ARM:dict(step=17500,total_updates=17500,parent_updates=10000,
                                      arm_key=ARM,registration_sha256='r')})

    def test_completion_requires_20k_and_inherited_10k(self):
        validate_completion(self.done(),'r')
        for change in ('incomplete','fresh','wrong_arm','wrong_reg'):
            d=self.done()
            if change=='incomplete':d['state']['last_full']=17500
            if change=='fresh':d['selected'][ARM]['parent_updates']=0
            if change=='wrong_arm':d['selected']['H8-P']=d['selected'][ARM]
            if change=='wrong_reg':d['selected'][ARM]['registration_sha256']='other'
            with self.assertRaises(ValueError):validate_completion(d,'r')

    def payloads(self):
        state=torch.tensor([1.,2.])
        optimizer=dict(state={0:dict(step=torch.tensor(10000.),exp_avg=state.clone())},param_groups=[dict(lr=.0002)])
        p=dict(registration_sha256='p',state=dict(step=10000,updates={'pure_continuous':10000}),
               models=OrderedDict([('pure_continuous.weight',state.clone()),('other.weight',state.clone())]),
               optimizers={'pure_continuous':optimizer},order={'position':16},rng=torch.tensor([2],dtype=torch.uint8))
        i=dict(registration_sha256='r',state=dict(step=10000,updates={ARM:10000}),
               models=OrderedDict([('P4084.weight',state.clone())]),optimizers={ARM:copy.deepcopy(optimizer)},
               order={'position':16},rng=torch.tensor([2],dtype=torch.uint8))
        return p,i

    def test_parent_mapping_preserves_model_optimizer_order_rng(self):
        p,i=self.payloads()
        validate_parent_payloads(p,i,'p','r')
        for field in ('models','optimizer','order','rng'):
            bad=copy.deepcopy(i)
            if field=='models':bad['models']['P4084.weight'][0]+=1
            if field=='optimizer':bad['optimizers'][ARM]['state'][0]['exp_avg'][0]+=1
            if field=='order':bad['order']['position']=0
            if field=='rng':bad['rng'][0]=3
            with self.assertRaises(ValueError):validate_parent_payloads(p,bad,'p','r')

    def test_tree_rejects_dtype_shape_and_missing_keys(self):
        self.assertTrue(same_tree({'x':torch.tensor([1])},OrderedDict(x=torch.tensor([1]))))
        self.assertFalse(same_tree({'x':torch.tensor([1])},{'x':torch.tensor([1.])}))
        self.assertFalse(same_tree({'x':torch.tensor([1])},{'x':torch.tensor([[1]])}))
        self.assertFalse(same_tree({'x':1},{}))
