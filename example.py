import docarray
import finetuner
from finetuner.hubble import check_data_exists

finetuner.login()

da = docarray.DocumentArray.empty(3)
resp = da.push('sabas-da-3')

print(resp)

resp = check_data_exists(finetuner.ft._client, 'sabas-da-3')

print(resp)