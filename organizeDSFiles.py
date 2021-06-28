import os
for d in os.listdir('dataset'):
    i = 0
    for f in os.listdir('dataset'+ '/'+d) :
        os.rename('dataset'+ '/'+d+'/'+f,'dataset'+'/'+d+str(i)+'.jpg')
        i += 1
