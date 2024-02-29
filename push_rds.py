import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv('./makale3_data.csv')

## PUSH TO THE AMAZON RDS ##
## amazon rds informations
host='egeamai1.ch6k0gkmsjko.eu-north-1.rds.amazonaws.com'
port=int(5432)
user='postgres'
passw='AtakanCuneyt123'
database='postgres'

# Oluşturulan engine nesnesinden bir bağlantı oluşturun
url= 'postgresql://' + user + ":" + passw + "@" + host + ":" + str(port) + "/" + database
engine = create_engine(url, echo=False, connect_args={'connect_timeout': 30})
connection = engine.connect()

# Convert the DataFrame to a PostgreSQL table using the connection
df.to_sql('makale3_data', connection, index=False, if_exists='replace')

# Bağlantıyı kapat
connection.close()
