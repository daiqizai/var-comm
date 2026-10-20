"""Boundary and continuation tests; no model, source or packet calls."""
import sys
import unittest
from unittest.mock import patch
import adapter as a


class Boundary(unittest.TestCase):
    def setUp(self):
        self.w = a.adapted_worker()
        self.source = a.RT/'assets/private_source_fixture.npz'
        self.r = dict(model_routes={'path':str(a.R/'results/model_routes_fixture.json')},
            P_checkpoint={'path':str(a.T/'assets/fake_P.pt')},
            Swin_checkpoint={'path':str(a.T/'assets/fake_S.pt')},
            static={k:{'path':str(a.T/'assets'/k)} for k in ['completion','table']},
            arpc_public={'weights_root':str(a.T/'assets/weights'), 'environment':str(a.T/'assets/environment')},
            sources=[{'archive':{'path':str(self.source)}}])
        routes = dict(files=[], author_files=[], author_root=str(a.E/'fake_author'),
                      dist={'path':str(a.E/'fake_dist.py')})
        hooks = []
        with patch.dict(self.w.firewall.__globals__, read=lambda _:routes):
            with patch.object(sys, 'addaudithook', hooks.append):
                self.blocked = self.w.firewall(self.r, 'NEST', a.stage_dir('NEST','rx'))
        self.audit = hooks[0]

    def test_existing_library_allowed(self):
        self.audit('open', (str(a.LIBRARY/'idna/__init__.py'),'r',0))

    def test_source_denied(self):
        with self.assertRaises(PermissionError):
            self.audit('open',(str(self.source),'r',0))

    def test_tx_private_denied(self):
        with self.assertRaises(PermissionError):
            self.audit('open',(str(a.T/'NEST/tx/cases/private.json'),'r',0))

    def test_other_results_denied(self):
        with self.assertRaises(PermissionError):
            self.audit('open',(str(a.R/'results/private/source.json'),'r',0))

    def test_actual_rx_allowed(self):
        self.audit('open',(str(a.T/'NEST/rx_inputs/actual.json'),'r',0))

    def test_library_parent_still_denied(self):
        with self.assertRaises(PermissionError):
            self.audit('open',(str(a.LIBRARY.parent/'secret.npy'),'r',0))

    def test_exact_remaining_scope(self):
        self.assertEqual(len(a.stages()),11)
        self.assertNotIn(('NEST','tx'),a.stages())
        self.assertNotIn(('NEST','phy'),a.stages())
        self.assertEqual(a.stage_dir('NEST','rx'),a.T/'NEST/rx_library_recovery_v1')
        self.assertEqual(a.stage_dir('ARPC','rx'),a.T/'ARPC/rx')


if __name__ == '__main__':
    unittest.main()
