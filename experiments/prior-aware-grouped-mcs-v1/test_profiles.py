"""Resource logic only. SyntheticPlanner is not an LDPC implementation."""
import unittest
import profiles as p


class SyntheticPlanner:
    identity={'implementation':'SYNTHETIC_UNIT_TEST_ONLY','version':1}
    qualified=True
    def plan(self,k,n,num_bits_per_symbol):
        if k/n>.95 or k/n<.2:raise p.UnsupportedConfiguration('synthetic rate domain')
        lifting=(k+21)//22
        ldpc=22*lifting
        return dict(k=k,n=n,k_ldpc=ldpc,k_filler=ldpc-k,n_cb=66*lifting,bg=1,z=lifting,
            code_blocks=1,mother_bits=68*lifting,puncturing_bits=max(0,68*lifting-n),
            shortening_bits=ldpc-k,repetition_bits=max(0,n-66*lifting),
            layout_id=f'synthetic:{k}:{n}:{num_bits_per_symbol}',decoder_config={'test_only':True},
            bit_mapping='test-only',scrambling='test-only',power_protocol='fixed_constellation_average_Es2')


class ProfileTests(unittest.TestCase):
    def test_true_raw_lengths_crc_and_no_convolutional_tail(self):
        self.assertEqual([p.payloads(m)[0] for m in range(4,10)],[360,660,1092,1860,3060,5088])
        self.assertEqual(p.payloads(7,25,4),(360,1800))
        row=p.Planner(SyntheticPlanner()).candidate(1024,8,0,None,(p.MCS('16QAM','5/6'),),'nominal')
        self.assertIsNotNone(row);g=row['groups'][0]
        self.assertEqual(g['information_bits'],3076);self.assertEqual(g['crc_bits'],16);self.assertEqual(g['tail_bits'],0)
        self.assertEqual(row['header_symbols'],68);self.assertEqual(row['header_tail_bits'],6)
        self.assertIsNone(p.Planner(SyntheticPlanner()).candidate(1024,8,0,None,(p.MCS('16QAM','3/4'),),'nominal'))

    def test_state_boundary_normalization_and_empty_group(self):
        self.assertEqual(p.normalize_state(7,100),(8,0));self.assertEqual(p.state_id(7,100),'m8_K0')
        self.assertEqual(p.payloads(7,100,7),p.payloads(8,0,7))
        with self.assertRaises(ValueError):p.payloads(7,0,7)
        with self.assertRaises(ValueError):p.normalize_state(8,170)

    def test_filled_budget_proportional_floor_then_group_order(self):
        self.assertEqual(p.fill_symbols((100,201),305),(102,203))
        row=p.Planner(SyntheticPlanner()).candidate(1024,7,0,4,(p.MCS('QPSK','1/2'),p.MCS('16QAM','3/4')),'full_budget')
        self.assertIsNotNone(row);self.assertEqual(sum(g['symbols'] for g in row['groups']),956)
        self.assertEqual(row['idle_symbols'],0)
        self.assertEqual(sum(g['symbols'] for g in row['groups'])+row['header_symbols'],1024)

    def test_unsupported_fill_retains_nominal_candidate(self):
        planner=p.Planner(SyntheticPlanner());mcs=(p.MCS('16QAM','1/2'),)
        self.assertIsNotNone(planner.candidate(2048,4,0,None,mcs,'nominal'))
        self.assertIsNone(planner.candidate(2048,4,0,None,mcs,'full_budget'))

    def test_maximum_is_configuration_local_and_includes_full_next_scale(self):
        planner=p.Planner(SyntheticPlanner())
        self.assertEqual(planner.max_K(1024,8,None,(p.MCS('16QAM','5/6'),)),9)
        self.assertIsNone(planner.max_K(1024,8,None,(p.MCS('QPSK','5/6'),)))
        self.assertEqual(planner.max_K(1024,4,None,(p.MCS('16QAM','5/6'),)),25)

    def test_family_and_matched_controls(self):
        def g(mod,rate):return dict(modulation=mod,nominal_rate=rate)
        self.assertEqual(p.family_memberships([g('QPSK','1/2')]),['B0','B3','B4'])
        self.assertEqual(p.family_memberships([g('QPSK','1/2'),g('QPSK','1/2')]),['B1','B2','B3','B4'])
        self.assertEqual(p.family_memberships([g('QPSK','1/2'),g('QPSK','3/4')]),['B2','B3','B4'])
        self.assertEqual(p.family_memberships([g('QPSK','1/2'),g('16QAM','1/2')]),['B3','B4'])

    def test_backend_failures_are_not_silently_declared_infeasible(self):
        class Broken(SyntheticPlanner):
            def plan(self,**kwargs):raise RuntimeError('backend crash')
        with self.assertRaises(RuntimeError):p.Planner(Broken()).candidate(1024,4,0,None,(p.MCS('QPSK','1/2'),),'nominal')
        class HiddenCRC(SyntheticPlanner):
            def plan(self,**kwargs):
                v=super().plan(**kwargs);v['k']+=24;return v
        with self.assertRaises(ValueError):p.Planner(HiddenCRC()).candidate(1024,4,0,None,(p.MCS('QPSK','1/2'),),'nominal')

    def test_candidate_ids_are_not_paid_header_ids_and_final_codebook_deduplicates(self):
        planner=p.Planner(SyntheticPlanner())
        a=planner.candidate(1024,7,0,None,(p.MCS('16QAM','1/2'),),'full_budget')
        b=planner.candidate(1024,7,0,None,(p.MCS('16QAM','3/4'),),'full_budget')
        self.assertEqual(a['wire_key'],b['wire_key']);self.assertNotEqual(a['stable_id'],b['stable_id'])
        self.assertNotIn('profile_id',a)
        codebook=p.freeze_codebook([a,b]);self.assertEqual(len(codebook['entries']),1)
        self.assertEqual(codebook['candidate_to_profile'][a['stable_id']],codebook['candidate_to_profile'][b['stable_id']])
        with self.assertRaises(ValueError):p.freeze_codebook([dict(a,encoder_qualified=False)])
        with self.assertRaises(ValueError):p.freeze_codebook([dict(a,wire_key=str(i),stable_id=str(i)) for i in range(4097)])


if __name__=='__main__':unittest.main()
