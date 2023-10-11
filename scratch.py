import pandas as pd

file = pd.read_csv('experiments/results.csv')

print(file)

file.loc[
    (file['MODEL']=='Normal CNN') & 
    (file['DATASET']=='Cifar 10') & 
    (file['DISTRIBUTION']=='Cauchy') & 
    (file['KERNEL TYPE']==2), 
    ['BEST ACCURACY', '@ EPOCH']] = 98, 2
# file.loc[['Normal CNN', 'Cifar10', 'Cauchy', 2], ['BEST ACCURACY']] = 98

print(file.loc[(file['MODEL']=='Normal CNN') & (file['DATASET']=='Cifar 10') & (file['DISTRIBUTION']=='Cauchy') & (file['KERNEL TYPE']==2)])

file.to_csv('experiments/results.csv')
# print(row_index)