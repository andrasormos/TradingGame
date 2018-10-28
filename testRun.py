
from ai.ai_versions.Trader_AI_v003_decider import Predictor
ChrisMarshall = Predictor()


action = ChrisMarshall.predictNextHourNow()

print(action)