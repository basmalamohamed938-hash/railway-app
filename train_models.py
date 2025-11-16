import pandas as pd
from multi_model import MultiTaskRailwayModel

# Load dataset
df = pd.read_csv("D:/Depi/railway app/railway-app/Railway Cleaned.csv")

# Create the multi-task model
multi_model = MultiTaskRailwayModel()

# Register targets
multi_model.add_target('Refund Request', 'classification')
multi_model.add_target('Price', 'regression')
multi_model.add_target('Journey Status', 'classification')

# Train models
multi_model.fit(df)

# Save models and encoders
multi_model.save(path='models/')
