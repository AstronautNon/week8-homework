#库导入区
import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

