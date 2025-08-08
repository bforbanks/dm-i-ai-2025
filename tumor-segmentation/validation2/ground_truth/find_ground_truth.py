width = 400
height = 426
total_pixels = width*height
P = 2120
N = total_pixels-P


def TP_FP_TN_FN(score,Pp,P,N):
    TP = ((Pp+P)*score*200)/2
    FN = P-TP
    FP = Pp-TP
    TN = N-FP
    return TP,FP,TN,FN

TP,FP,TN,FN = TP_FP_TN_FN(0.00019613483766663407,1347,P,N)
print(TP,FP,TN,FN)

for i in range(1,2000):
    if abs(TP_FP_TN_FN(0.00019613483766663407,i,P,N)[0]-round(TP_FP_TN_FN(0.00019613483766663407,i,P,N)[0]))<0.0001:
        print(i)
        # break