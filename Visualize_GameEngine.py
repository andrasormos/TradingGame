import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(threshold=np.nan, linewidth=300)


from GameEngine_v001 import PlayGame

GE = PlayGame()
#GE.startGame()
df_segment = GE.getChartData()

print (np.shape(df_segment))

#print(df_segment)
print("\n")
#df_segment.reset_index()



# PLOT
if 1 == 2:
    # graphed on matrix
    plt.imshow(df_segment, cmap='hot')
    plt.show()


    #ax2 = fig.add_subplot(211)
    #plt.imshow(blank_matrix, cmap='hot')

    #plt.show())





