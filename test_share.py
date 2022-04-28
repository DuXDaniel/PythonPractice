import numpy as np
import sys
from multiprocessing.managers import SharedMemoryManager

def main(argv):
    smm = SharedMemoryManager()
    smm.start()

    try:
        a_sh = np.arange(12)
        a_list = list()
        a_list.append(a_sh)
        print('testing begins')
        print(a_sh)
        print(np.array(a_sh).reshape((3,4)))
        alternate(a_list)
    finally:
        smm.shutdown()

def alternate(a_list):
    print(a_list[0])

if __name__ == '__main__':
    main(sys.argv)
