import ast
import io
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import rebuttal_iclr27 as r
import rebuttal_iclr27_r2_o12 as o


class IndependentTargetParserTest(unittest.TestCase):
    def test_fraction_decimal_equivalence(self):
        self.assertTrue(o.math_correct(r'\frac{1}{2}',r'Therefore $\boxed{0.5}$.'))

    def test_nested_box(self):
        self.assertTrue(o.math_correct(r'\frac{1}{2}',r'Final: $\boxed{\frac{1}{2}}$.'))

    def test_last_box_has_priority(self):
        self.assertTrue(o.math_correct('2',r'Working: $\boxed{1}$. Final: $\boxed{2}$.'))
        self.assertFalse(o.math_correct('1',r'Working: $\boxed{1}$. Final: $\boxed{2}$.'))

    def test_working_value_is_not_final_answer(self):
        self.assertFalse(o.math_correct('0.5',r'Compute $0.5$ midway; final answer is $\boxed{3}$.'))

    def test_empty_and_nonmath_fail(self):
        self.assertFalse(o.math_correct('2',''))
        self.assertFalse(o.math_correct('2','I cannot solve this problem.'))

    def test_wrong_answer_fails(self):
        self.assertFalse(o.math_correct('2',r'$\boxed{3}$'))

    def test_symbolic_equivalence(self):
        self.assertTrue(o.math_correct(r'\sqrt{8}',r'$\boxed{2\sqrt{2}}$'))

    def test_fenced_python_entrypoint(self):
        text='Here is the solution:\n```python\ndef add(x, y):\n    return x + y\n```\nDone.'
        clean=o.code_solution(text,'add');tree=ast.parse(clean)
        self.assertEqual([n.name for n in tree.body if isinstance(n,ast.FunctionDef)],['add'])

    def test_python_dependencies_survive(self):
        text='import math\ndef helper(x):\n    return math.sqrt(x)\ndef solve(x):\n    return helper(x)\n'
        tree=ast.parse(o.code_solution(text,'solve'))
        self.assertEqual({n.name for n in tree.body if isinstance(n,ast.FunctionDef)},{'helper','solve'})
        self.assertTrue(any(isinstance(n,ast.Import) for n in tree.body))

    def test_empty_python(self):
        self.assertEqual(o.code_solution('','solve'),'')


if __name__=='__main__':
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTargetParserTest)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue())
    r.write(o.P/'parser_unit_tests.json',dict(status='pass' if result.wasSuccessful() else 'fail',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),test_source_sha256=r.sha(Path(__file__)),
        parser_source_sha256=r.sha(r.ROOT/'scripts/rebuttal_iclr27_r2_o12.py'),log=stream.getvalue()))
    sys.exit(0 if result.wasSuccessful() else 1)
