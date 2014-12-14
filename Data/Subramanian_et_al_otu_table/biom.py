import numpy as np
from biom import load_table

table_test = load_table('only_Healthy_Singletons.biom')
print(table_test)