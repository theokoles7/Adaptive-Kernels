x = [6,7,8,9,10,11,12,13,14,15]

for i in x:
  if (i) %2 == 0 and (i) % 3 != 0:
    print("if " + str(i))
  elif i % 3 == 0:
    print("elif ", str(i))
  else: print("else",str())