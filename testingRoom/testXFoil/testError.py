import os
import errno
import time
from timeout import timeout

@timeout(1, os.strerror(errno.ETIMEDOUT))
def Longfunc():
    timeout = time.time() + 60 * 5  # 5 minutes from now
    while True:
        test = 0
        if test == 5 or time.time() > timeout:
            break
        test = test - 1


Longfunc()

#with timeout(seconds=1):
#    time.sleep(4)