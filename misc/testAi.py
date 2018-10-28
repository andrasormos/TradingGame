from game_engines.GE_v061 import PlayGame
import matplotlib.pyplot as plt

GE = PlayGame()
GE.startGame()
logNr = "061E"
GE.defineLogNr(logNr)




a=0
s, r, _ = GE.nextStep(a)
d = GE.getCurrentData()
sp = d[0]
print(r,",",sp,"=","(",a,")")

plt.imshow(s, cmap='hot')
plt.show()


a=1
s, r, _ = GE.nextStep(a)
d = GE.getCurrentData()
sp = d[0]
print(r,",",sp,"=","(",a,")")
plt.imshow(s, cmap='hot')
plt.show()

a=2
s, r, _ = GE.nextStep(a)
d = GE.getCurrentData()
sp = d[0]
print(r,",",sp,"=","(",a,")")
plt.imshow(s, cmap='hot')
plt.show()