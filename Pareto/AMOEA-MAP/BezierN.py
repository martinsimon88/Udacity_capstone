class BezierN():
    order = 1
    binom = []
    tts   = []
    ts    = []

    def __init__(self, order_):
        self.order=order_
        self.binom = [0]* self.order
        self.ts    = [0]* self.order
        self.tts   = [0]* self.order
        self.binom[0]=1

        for i in range (1,self.order+1):
            temp=self.binom[0]
            for ii in range (1,i):
                temp2= self.binom[ii]
                self.binom[ii]=temp+self.binom[ii]
                temp = temp2
                print( str(self.binom[ii])+" ");
                #println();
        for i in range(0,self.order):
            print self.binom[i],


    def interpolate(self,coefs, n ):
        output = [0]*(n+1)
       # print
       # print coefs
       # print self.ts
       # print self.tts
       # print self.binom
       # print output
        step = 1.0/float(n)
        t = 0
        output[0]=coefs[0]
        for i in range (1,n+1):
            t+=step
            tt=1.0-t
            ttemp=1.0
            tttemp=1.0
            for j in range (0,self.order):  # generate powers of t
                self.ts[j] = ttemp;
                self.tts[self.order-j-1] = tttemp
                ttemp*=t
                tttemp*=tt
            output[i]=0
            for j in range (0,self.order):  
                output[i]+=coefs[j]*self.tts[j]*self.ts[j]*self.binom[j]
        return output

#MyBezier = BezierN(5)
#bezierout=MyBezier.interpolate([0,3,2,1,0],100)
#print bezierout




