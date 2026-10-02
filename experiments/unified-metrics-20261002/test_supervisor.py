"""Current parent identity remains a gate after a supervisor handoff."""
import json
from pathlib import Path
import tempfile
import unittest

import supervisor as s


class ExitGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.parent=Path(self.tmp.name)
        (self.parent/'supervisor_launch.json').write_text(json.dumps(dict(pid=100,start_ticks='500')))

    def test_replacement_controller_is_waited_by_identity(self):
        processes={100:dict(state='S',start_ticks='500',command='python /repo/experiments/metric-speed-20261002/controller.py')}
        self.assertEqual(s.original_processes(self.parent,processes),[100])

    def test_exited_and_recycled_owner_do_not_block(self):
        for processes in ({},{100:dict(state='Z',start_ticks='500',command='old')},
                          {100:dict(state='S',start_ticks='700',command='unrelated')}):
            with self.subTest(processes=processes):self.assertEqual(s.original_processes(self.parent,processes),[])

    def test_original_worker_and_unreadable_owner_fail_closed(self):
        processes={200:dict(state='S',start_ticks='800',command='python /repo/experiments/scale-causal-partial-residual-20261002/m2_runner.py')}
        self.assertEqual(s.original_processes(self.parent,processes),[200])
        with self.assertRaises(RuntimeError):s.original_processes(self.parent,{100:dict(unreadable=True)})


if __name__=='__main__':unittest.main()
