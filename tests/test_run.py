from psdf_suction.psdf import PSDF
import numpy as np

def test_run():
    psdf = PSDF((100,100,100), 0.002)
    depth = np.random.random((100,100)).astype(np.float32)
    psdf.fuse(depth, np.eye(3), np.eye(4))
    

if __name__ == '__main__':
    test_run()