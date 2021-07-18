import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np
dct_buy = {}
dct_abt ={}

df1 = pd.read_csv("buy.csv")
df2 = pd.read_csv("abt.csv", encoding='ISO-8859-1')


names_buy = df1["name"]
brand = df1["manufacturer"]
names_abt = df2["name"]
brand = brand.str.lower()
brand = brand.tolist()
brand = list(dict.fromkeys(brand))
names_buy = names_buy.str.lower()
names_abt = names_abt.str.lower()

brand = pd.Series(brand)
brand = brand.replace(np.nan, '', regex=True)
brand = brand.tolist()

brand = [name.split(" ")[0] for name in brand]
brand = pd.Series(brand)
brand = brand.replace('', 'nan', regex=True)
brand = brand.tolist()


for company in brand:
    truth_buy = (names_buy.str.contains(company, regex=False))
    truth_values_buy = np.where(truth_buy)[0] 
    truth_values_buy = truth_values_buy.tolist()
    new_list = []
    
    for value in truth_values_buy:
        new_list.append(df1.iloc[value,0])
    
    dct_buy[company]=new_list
    
    new_lst2 = []
    
    truth_abt = (names_abt.str.contains(company, regex=False))
    truth_values_abt = np.where(truth_abt)[0]
    truth_values_abt = truth_values_abt.tolist()
    for value in truth_values_abt:
        new_lst2.append(df2.iloc[value,0])
    dct_abt[company]=new_lst2
    
    

abt_list = []
abt_brand =[]

buy_list = []
buy_brand = []
for key, value in dct_abt.items():
    if len(value) != 0:
        abt_brand += ([key] * int(len(value)))
        abt_list.append(value)


for key, value in dct_buy.items():
    if len(value) != 0:
        buy_brand += ([key] * int(len(value)))
        buy_list.append(value)
        
def flatten_out_nested_list(input_list):
    if input_list is None:
        return None
    if not isinstance(input_list, (list, tuple)):
        return None
    flattened_list = []
    for entry in input_list:
        entry_list = None
        if not isinstance(entry, list):
            try:
                entry_list = ast.literal_eval(entry)
            except:
                pass
        if not entry_list:
            entry_list = entry
        if isinstance(entry_list, list):
            flattened_entry = flatten_out_nested_list(entry_list)
            if flattened_entry:
                flattened_list.extend(flattened_entry)
        else:
            flattened_list.append(entry)
    return flattened_list

abt_list = flatten_out_nested_list(abt_list)
buy_list = flatten_out_nested_list(buy_list)

df_abt = pd.DataFrame({'block_key':abt_brand, 'product_id':abt_list,})
df_abt.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)
  
df_abt.to_csv('abt_blocks.csv', index=False)

df_buy = pd.DataFrame({'block_key':buy_brand, 'product_id':buy_list,})
df_buy.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)
  
df_buy.to_csv('buy_blocks.csv', index=False)