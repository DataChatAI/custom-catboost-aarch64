import numpy 
import platform
import catboost
from catboost import CatBoostRegressor

print("CatBoost sanity check")
print("==================================================")
print(f"CatBoost version: {catboost.version.VERSION}")
print(f"Architecture: {platform.machine()}")
dataset = numpy.array([[1,4,5,6],[4,5,6,7],[30,40,50,60],[20,15,85,60]])
train_labels = [1.2,3.4,9.5,24.5]
model = CatBoostRegressor(iterations=20, learning_rate=1, depth=6, loss_function='RMSE')
fit_model = model.fit(dataset, train_labels)

print(fit_model.get_params())
