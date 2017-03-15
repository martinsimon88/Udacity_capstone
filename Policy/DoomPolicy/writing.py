import re

# creation
with open('pers.txt','wb') as g:
    g.write('Dan \n Warrior \n 500 \r\n 1 \r 0 \n Jim  \n  dragonfly\r300\r2\n10\r\nSomo\ncosmonaut\n490\r\n3\r65')

with open('pers.txt','rb') as h:
    print 'exact content of pers.txt before treatment:\n',repr(h.read())
with open('pers.txt','rU') as h:
    print '\nrU-display of pers.txt before treatment:\n',h.read()


# treatment
def ripli(file_name,who,what):
    with open(file_name,'rb+') as f:
        ch = f.read()
        x,y = re.search('^\s*'+who+'\s*[\r\n]+([^\r\n]+)',ch,re.MULTILINE).span(1)
        f.seek(x)
        f.write(what+ch[y:])
ripli('pers.txt','Jim','Wizard')


# after treatment
with open('pers.txt','rb') as h:
    print 'exact content of pers.txt after treatment:\n',repr(h.read())
with open('pers.txt','rU') as h:
    print '\nrU-display of pers.txt after treatment:\n',h.read()