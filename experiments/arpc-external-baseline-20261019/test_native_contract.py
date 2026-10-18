"""Static boundary checks without importing Torch or any model."""
import ast,pathlib,unittest

ROOT=pathlib.Path(__file__).resolve().parent
UP=ROOT/'upstream'
class Contract(unittest.TestCase):
    def test_upstream_decoder_truth_parameter_unused(self):
        tree=ast.parse((UP/'compression/reciever.py').read_text(encoding='utf8'))
        f=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='decoding'][-1]
        self.assertFalse(any(isinstance(n,ast.Name) and n.id=='gt_ls_Bl' and isinstance(n.ctx,ast.Load) for n in ast.walk(f)))
    def test_exactly_one_scale_loop_is_bounded(self):
        tree=ast.parse((UP/'compression/reciever.py').read_text(encoding='utf8'))
        f=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='decoding'][-1]
        self.assertEqual(sum(isinstance(n,ast.For) and ast.unparse(n.iter)=='enumerate(vae_scale_schedule)' for n in ast.walk(f)),1)
    def test_completion_only_uses_received_prefix(self):
        tree=ast.parse((UP/'compression/reciever.py').read_text(encoding='utf8'))
        f=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='decompress_cfg'][0]
        uses=[n for n in ast.walk(f) if isinstance(n,ast.If) and ast.unparse(n.test)=='si <= gt_leak']
        self.assertEqual(len(uses),1)
        self.assertEqual(ast.unparse(uses[0].body[0]),'idx_Bld = gt_ls_Bl[si]')
    def test_source_traversal_remains_native(self):
        tree=ast.parse((UP/'compression/sender.py').read_text(encoding='utf8'))
        f=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='encoding'][0]
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='get_prob' for n in ast.walk(f)),1)
        self.assertIn('len(bits) < args.vae_type',ast.unparse(f))
    def test_rx_source_argument_is_rejected(self):
        s=(ROOT/'a1_native.py').read_text(encoding='utf8')
        self.assertIn("if role=='rx':assert a.sources is None",s)
        self.assertIn('install_rx_boundary',s)
    def test_no_scientific_import_in_audit(self):
        tree=ast.parse((ROOT/'a1_audit.py').read_text(encoding='utf8'))
        names=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):names.extend(x.name for x in n.names)
            if isinstance(n,ast.ImportFrom):names.append(n.module)
        self.assertFalse(set(names)&{'torch','transformers','compression.sender','compression.reciever'})
if __name__=='__main__':unittest.main()
