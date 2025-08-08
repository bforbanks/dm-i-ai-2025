import numpy as np
#1 and 2 together is 0.00010481654109988844
print(0.000028293951212051693+0.00007652258988783675)
print(0.000028293951212051693+0.00007652258988783675-0.00010481654109988844)
total_pixels = 26049600
no1_pixels = 170400
weighing = no1_pixels/total_pixels
weighing = 1/129

number4 = 0.00007646479882128902/weighing
number5 = 0.00007646479882128902/weighing

number1 = 0.00007652258988783675/weighing
number2 = 0.00007652303553021612/weighing
number3 = 0.00007652348117778606/weighing

numb400 = 0.0000767012620101101/weighing

number1o = 1/number1
number2o = 1/number2
number3o = 1/number3
numb400o = 1/numb400
diff = number1o-number2o
diff2 = number2o-number3o
diff3 = number1o-number3o
diff4 = number1o-numb400o
print(diff,diff2,diff3)
print(1/(2*diff))
print(1/(2*diff2))
print(1/(diff3))
print("new",400/(2*diff4))
print((2*number1-number1*number4)/(2*number1-2*number4))
# for i in range(10):
#     e = i+0.06
#     # if abs((2*number1-number1*number4+e*number4-e*number1)/(2*number1-2*number4)-round((2*number1-number1*number4+e*number4-e*number1)/(2*number1-2*number4))) < 1e-1:
#     print("wuup",(2*number1-number1*number4+e*number4-e*number1)/(2*number1-2*number4))

# hi,lo=1400,840
# previousdiff=1000
# for e in np.linspace(0,1,100):
#     for TP in range(00,2500):
#         FP = 170400-TP
#         for TN in [0]:# range(0,170400-TP-FP):
#             FN = 170400-TP-FP-TN
#             diff = abs(2*TP+e/(2*TP+FP+FN+e)-number1)
#             if diff < previousdiff:
#                 previousdiff = diff
#                 print(TP,FP,TN,FN,e,diff)
#                 print(2*TP/(2*TP+FP+FN))


# # epsilon = 
# number1 = 0.00007652258988783675*129
# number2 = 0.00007652303553021612*129
# number3 = 0.00007652348117778606*129
# number1o = 1/number1
# number2o = 1/number2
# number3o = 1/number3
# diff = number1o-number2o
# diff2 = number2o-number3o
# diff3 = number1o-number3o
# print(diff,diff2,diff3)
# print(1/(2*diff))
# print(1/(2*diff2))
# print(1/(diff3))

# for i in range(1,170400*2+1):
#     if diff*i % 1 < 1e-10:
#         print(i, diff*i)








