import sys
import time
import threading

class XPSObj(object):
    checkLoop = 0

    def XPS_Open (self):
        f = open('connectStatFile.txt', 'w')
        f.write('1')
        f.close()

    def XPS_Close(self):
        f = open('connectStatFile.txt', 'w')
        f.write('0')
        f.close()

    def processMovementFile(self):
        f = open('movementCommFile.txt','r')
        posMov = float(f.readline())
        compStat = int(f.readline())
        f.close()

        return posMov, compStat

    def indicateCompletedMovement(self, posMov):
        f = open('movementCommFile.txt','w')
        f.write(str(posMov) + "\n")
        f.write('1')
        f.close()

    def checkDisconnectOrder(self):
        f = open('connectStatFile.txt','r')
        checker = int(f.readline())
        f.close()
        return checker

    def signalPosition(self): # async
        while(self.checkLoop != 2): # have a more rigorous method to kill the thread
            f = open('positionFile.txt','w')
            f.write(str(self.checkLoop)) #is this the correct function?
            f.close()
            time.sleep(0.1) # pause every 100 ms, don't need the crazy granularity

    def orderLoop(self): #async
        while(self.checkLoop != 2): # have a more rigorous method to kill the thread
            posMov, compStat = self.processMovementFile()
            if (compStat == 0):
                print('Moving to: ' + str(posMov))
                time.sleep(5)
                self.indicateCompletedMovement(posMov)
            time.sleep(0.1) # pause every 100 ms, don't need the crazy granularity

    def initOrderLoop(self):
        self._orderLoop_thread = threading.Thread(target=self.orderLoop, args=())
        self._orderLoop_thread.daemon = True
        self._orderLoop_thread.start()

    def initSignalPos(self):
        self._signalPosition_thread = threading.Thread(target=self.signalPosition, args=())
        self._signalPosition_thread.daemon = True
        self._signalPosition_thread.start()

def main(argv):
    controlXPS = XPSObj()
    controlXPS.XPS_Open()
    controlXPS.initOrderLoop()
    controlXPS.initSignalPos()

    checkLoop = controlXPS.checkDisconnectOrder()

    while (checkLoop != 2):
        checkLoop = controlXPS.checkDisconnectOrder()
        if (checkLoop == 2):
            controlXPS.checkLoop = 2
            controlXPS.XPS_Close()
        time.sleep(0.5)

if __name__ == '__main__':
    main(sys.argv)