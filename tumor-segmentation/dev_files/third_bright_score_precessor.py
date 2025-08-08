number1 = 0.00012288430326918618*200 # 1 image all white
number2 = 0.00012288501556350316*200 # one pixel black
number3 = 0.00017258360054970222
number4 = 0.00022570244127130356*200


number1o = 1/number1
number2o = 1/number2
diff = number1o-number2o
print(diff)
print(1/(2*diff))