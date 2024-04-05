import pandas as pd


df = pd.read_excel("normalizasyonyapilmamisveriler.xlsx") 


normalized_df = (df - df.min()) / (df.max() - df.min())


normalized_df.to_excel("normalized_veriler.xlsx", index=False)  



