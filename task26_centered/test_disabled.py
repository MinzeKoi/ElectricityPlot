import subprocess
import sys
import unittest
from pathlib import Path


class DisabledEntryTest(unittest.TestCase):
    def test_every_execution_phase_refused(self):
        entry = Path(__file__).with_name('run.py')
        for phase in ('train', 'source', 'freeze', 'evaluate', 'predict'):
            run = subprocess.run([sys.executable, str(entry), phase], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('execution is disabled', run.stderr)

if __name__ == '__main__':
    unittest.main()
