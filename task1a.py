import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process



df1 = pd.read_csv("buy_small.csv")
df2 = pd.read_csv("abt_small.csv", encoding='ISO-8859-1')

df1 = df1.drop_duplicates(subset='idBuy', keep='first')
df2 = df2.drop_duplicates(subset='idABT', keep='first')


abtID = []
buyID = []


buy_small_name = df1["name"]
abt_small_name = df2["name"]

abt_small_id = df2["idABT"]


zip_iterator = zip(abt_small_name, abt_small_id)

for name, idnum in zip_iterator:
    highest = process.extractOne(name,buy_small_name,  scorer=fuzz.token_set_ratio)
    if highest[1]>=95:
        buy_small_name.drop([highest[2]],inplace=True)

    if highest[1]>=72:
        buyID.append(df1.iloc[highest[2],0])
        abtID.append(idnum)
        
 
        
df = pd.DataFrame({'idAbt':abtID, 'idBuy':buyID,})
df.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)
  
df.to_csv('task1a.csv', index=False)


print(type(buy_small_name))


