import OpenVSPGame

a=1
'''
CSP = OpenVSPGame.start_gametest()
test = CSP.step(a)
print test
'''
CSP = OpenVSPGame.start_gametest()
test = CSP.reset()
read = CSP.readAeroData()
print test, read