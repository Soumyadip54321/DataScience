import numpy as np
import pandas as pd
l = [1, 2, 3]
two_d = [[1, 2, 3], [4, 5, 6]]
array1 = np.array(l)
array2 = np.array(two_d)
#print(array2-1.1)

series1 = pd.Series(array1, index=['Soumyadip', 'Diganta', 'Rita'])
print(series1)
