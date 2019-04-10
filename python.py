# Notes:

# show version
pd.__version__
# show file location
pd.__file__

# install fastai
conda create --name fastai python=3.7
conda update --prefix /Users/xbno/anaconda3 anaconda
conda install fastai -c fastai -c pytorch -c conda-forge

# ************************************************************************************************************* #
# in jupyter notebooks/lab
# ************************************************************************************************************* #

# add conda env as kernel to lab or notebook
conda install ipykernel
python -m ipykernel install --user --name mykernel

# on server use:
import matplotlib
matplotlib.use('agg')
%matplotlib inline

# remove ... in dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

# span full width (notebook only)
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:98% !important; }</style>"))

# decimal points
pd.options.display.float_format = '{:.2f}'.format

# ************************************************************************************************************* #
# matplotlib + seaborn
# ************************************************************************************************************* #

# log scale
sns.distplot(..., hist_kws={'log':True})

# ************************************************************************************************************* #
# pandas + numpy + scipy
# ************************************************************************************************************* #

# read_csv, to_csv
df = pd.read_csv('/home/ec2-user/corecustomers_ctrla_utf8.csv',sep=r'\\\\001',nrows=500000,usecols=['summary_memberstatus','membershipstatus'])
df = pd.read_csv(f'{args.path}/raw_calc.csv',sep=r'\\\\001',dtype=np.dtype('object')) # dtype = object reads values either a string or nan
df.to_csv(f'/mnt/gc/data/raw_calc.csv',sep='~',quotechar='"',index=False, quoting=csv.QUOTE_MINIMAL)

# import data integrity checks, coerce means convert all non dt or numeric data to nat nan
pd.to_numeric(df['tester'], errors='coerce')
pd.to_datetime(df['tester'], errors='coerce')

# replace values with nan
df = df.replace(['', ' ', 'null','None','Null','none','NONE','?','na'], np.nan)

# drop/find duplicates
rec_df = rec_df.drop_duplicates(subset=['rec','score'],keep='first') # remove duplicates
rec_df = rec_df[rec_df.duplicated(subset=['rec','score'],keep=False)] # keep only duplicates
rec_df = rec_df[~rec_df.duplicated(subset=['rec','score'],keep=False)] # keep only non-duplicates

# simple way to find duplicate occurances of data in a log version df
match_ct = match['sm_id'].value_counts()
updates = match[match['sm_id'].isin(match_ct[match_ct>1].index)].sort_values(by='sm_id')

# build up a version of a df from all nans to filled in values
df_a.combine_first(df_b)

# for multiple columns
for bool_col in [k for k,v in remap.items() if v in [k for k,v in uploaded_types.items() if v['type'] == 'boolean']]:
    test_case_core[bool_col] = pd.to_numeric(test_case_core[bool_col],errors='coerce').astype(bool)

# flatten multi-index
date_grouper.columns = ['_'.join([str(i) for i in t]) if t[1] != '' else t[0] for t in date_grouper.columns]

# find word in column
ia[ia['item_description'].str.contains('BANA')]

# concat columns
rule = ['firstname','lastname','emailname']
rule_applies['glued'] = ''
rule_applies['glued'] = rule_applies['glued'].str.cat([rule_applies[v] for v in rule]).str.lower()

# get sinlge value (will fail if result is series)
a.loc[a['firstvisitdate']==a['firstvisitdate'].min(),'firstvisitrefno'].item()

# grab all columns that are datetime except columns named 'birthdate'
a.loc[:,[c for c,t in a.dtypes.to_dict().items() if np.issubdtype(t, np.datetime64) and 'birthdate' not in c]]#.min().min()

# dictionary from dataframe with unique index as keys, and column as values
df.set_index('item_id').to_dict()['family_group_name']

# subset df by random id
combined_df[combined_df['sm_id_winner'].isin(np.random.choice(combined_df['sm_id_winner'].unique(),10000))]

# apply programatic lambda:
def format_phone_numbers(df,phone_dict):
    df['phone_numbers'] = df.apply(lambda row: [{"phone_number":row[col], "phone_type":label} for col,label in phone_dict.items()], axis=1)
    df.drop([k for k in phone_dict.keys()],axis=1,inplace=True)
    return df
combined_df = format_phone_numbers(combined_df,phone_dict)

# apply to non-null values
def format_number_to_int(x):
    if pd.isnull(x):
        return x
    elif type(x) == str:
        return x
    else:
        return '{:.0f}'.format(float(x))

# find only numbers
def only_nums(x):
    return re.findall('(\d*)',x)[0]
combined_df['workphone'] = combined_df['workphone'].apply(only_nums)

q = df.sum(axis=1).quantile(0.99)
print(q, 'is the cuttoff')
df_no_outliers = df[df.sum(axis=1).map(lambda x: x < q)]
print(df.shape[0] - df_no_outliers.shape[0],'many customers removed')

# split into equal parts
dfs = [df for df in np.array_split(df,3)]

# ttest pvalue
from scipy import stats
a = player_df[player_df['segment'] == '25_50_control']['huggies_product_total']
b = player_df[player_df['segment'] == '25_50_email_5000pts']['huggies_product_total']
stats.ttest_ind(a,b)

# ************************************************************************************************************* #
# sys, os, and file manipulation
# ************************************************************************************************************* #

# append paths
import sys
sys.path.append('../src/scripts/')

# env variables
import os
home = os.environ['HOME']

# create dir
if not os.path.exists(local_creds_path):
    os.makedirs(f'{home}/.smsync/')
    os.system(f'aws s3 cp {s3_creds_path} {local_creds_path}')

# catch cmdline stdout
job_ids = [i for i in os.popen(f'''smsyncer job list wellbiz_{args.vpc} -format='{{{{ .ID }}}}' -limit {len(files_to_upload)}''').read().split('\n') if i != '']

# ************************************************************************************************************* #
# python std libs
# ************************************************************************************************************* #

# args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sample', type=str, default='0')
parser.add_argument('--gzip', type=str, default='t')
args = parser.parse_args()

# fake args for use in notebook to test scripts
class Args():
    def __init__():
        pass

args = Args
args.path = '/mnt/gc/data'
args.source = 'full_inc_2'
args.map = 'full_inc_mapping_2'
args.calc = 'full_inc_calc_2'
args.vpc = 'stg'
args.sample = '0'

# logging
def get_logger(name,logfile):
    log_format = '%(asctime)s - %(name)8s - %(levelname)5s - %(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        format=log_format,
                        filename=logfile,
                        filemode='a') # appends
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(name).addHandler(console)
    return logging.getLogger(name)

logger = get_logger('inc_GR.py','/data/20190318.log')
logger.info('msg')
logger.info(f'{i}: {(datetime.now() - loop_start_time).seconds} sec - new record matches: {len(tf[f"rule_{i}_all"][tf[f"rule_{i}_all"]])}')

# formatting
columns = '(origin_id serial primary key, '+' text, '.join([c for c in df.columns if 'origin_id' not in c]) + ' text);'
create_cmd = f'''-c "create table if not exists {table} {columns}"'''

'.26' = '{:.2f}'.format(.235872)
'.26' = f'{.235872}'
'01' = '{:02d}'.format(1)
'100203' = '{:.0f}'.format(100203.021)

# add a,b,c suffix to filename
letter = chr(ord('a')+i)
file_to_upload = f'{args.path}/std_apd_email_{args.sample}{letter}.json'

# dates
datetime.datetime.strptime('20180531', '%Y%m%d')

# from string to date
'20180608'.strftime('%Y%m%d')

# regex
'datetime.datetime(2018, 7, 1)' = re.search("'start_date': (.+\(.+\))",''''start_date': datetime.datetime(2018, 7, 1),''').group(1)

# dict
a = {'a':.1,'b':.2,'c':.3,'e':.5,'f':2,'g':2}
b = {'a':1,'b':2,'c':3,'d':4,'h':1,'g':1}

# new dictionaries based on keys of both
{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 0.5, 'f': 2, 'g': 1, 'h': 1} = dict(a.items()^b.items())
{'e': 0.5, 'f': 2} = {i:a[i] for i in set(a.keys())-set(b.keys())}

# add dicts
remap = {}
remap.update(remap_appended)

# chop list into multiple lists of maximum length n
def chunks(l, n=2500):
    for i in range(0, len(l), n):
        yield l[i:i+n]

# read/write json to dict
with open('/Users/gcounihan/Downloads/user_item_recommendations.json') as f:
    data = json.load(f)
with open('/vol/pipeline/v2/lightmf_analysis.txt', 'w') as fp:
    json.dump(user_item_recommended_nonpurch, fp)

# read/write json gzip
import gzip
import json
with gzip.GzipFile(jsonfilename, 'w') as fout:
    fout.write(json.dumps(data).encode('utf-8'))
with gzip.GzipFile(jsonfilename, 'r') as fin:
    data = json.loads(fin.read().decode('utf-8'))

# read/write pickles
with open('filename.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)


# ************************************************************************************************************* #
# data to the cloud, boto3, sqlalchemy, s3fs
# ************************************************************************************************************* #

# slow for large queries
import boto3
client = boto3.client(service_name='athena',region_name='us-east-1')
client.start_query_execution(QueryString="select * from test.ads_log_s3 where dy = '02' and mon = '02' and yr = '2017'",ResultConfiguration={'OutputLocation':'s3://bucket/path/'})

# boto3 s3
def load_csv_from_s3(args,b=f'teams-{args.vpc}-com',p=f'whatever/more.tsv'):
    s3 = boto3.resource('s3')
    return pd.read_csv(StringIO(s3.Object(b,p).get()['Body'].read().decode('latin-1')),sep='\t')
def load_json_from_s3(args,b=f'teams-{args.vpc}-com',p=f'admteam/common/{args.vpc}_config.json'):
    s3 = boto3.resource('s3')
    return json.loads(s3.Object(b,p).get()['Body'].read().decode('utf-8'))

def save_to_s3(args,b=f'teams-{args.vpc}-com',p=f'whatever'):
    s3 = boto3.resource('s3')
    s3.Bucket(f'teams-{args.vpc}-sessionm-com').upload_file(f'{args.path}/{}',f'{args.s3_path}/raw_2019-02-12_2019-02-14.csv')
    # s3.Bucket(b).upload_file(f'/mnt/gc/data/raw_2019-02-12_2019-02-14.csv',f'{p}/raw_2019-02-12_2019-02-14.csv')
def save_to_s3(args,file):
    s3 = boto3.resource('s3')
    s3.Bucket(f'teams-{args.vpc}-sessionm-com').upload_file(f'{args.path}/{file}',f'{args.s3_path}/{file}')

# s3fs
import s3fs
s3 = s3fs.S3FileSystem()
raw_s3_file = f"s3://teams-{args.vpc}-sessionm-com/{args.s3_raw_path}/raw.csv"
core = pd.read_csv(raw_s3_tmp_file, sep="~", header=0, na_values=na_values, compression='gzip', dtype=np.dtype('object'))
with s3.open(raw_s3_file, 'w') as f:
    df.to_csv(f, sep='~', index=False, compression='gzip')
with s3.open(raw_s3_file, 'r') as f:
    e = pd.read_csv(f, sep='~')

with s3.open('s3://teams-stg-sessionm-com/admteam/integration/wellbiz/data/std_apd_email_50000a.json', 'r') as f:
    d = json.load(f)
with s3.open('s3://teams-stg-sessionm-com/admteam/integration/wellbiz/data/std_apd_email_50000a.json', 'w') as f:
    json.dump(d, f)

# pandas + postgres query
import psycopg2 as pg
import pandas.io.sql as psql

# query
def run_postgres_query(config,query):
    conn = pg.connect(dbname=config['database'], user=config['user'], password=config['password'], host=config['host'])
    return psql.read_sql(query,conn)

# slow for bulk (500k+ rows) inserts
from sqlalchemy import create_engine

def insert_postgres(args,job_config,df):
    engine = create_engine(f'postgresql+psycopg2://{job_config["db"]["user"]}:{job_config["db"]["password"]}@{job_config["db"]["host"]}/{job_config["db"]["dbname"]}')
    df.to_sql('table',engine,index=False,if_exists=args.if_exists,chunksize=int(args.chunksize))

# use psql via cmdline to have quicker performance for bulk uploads
psql_cmd_down = f'''PGPASSWORD='{config["redshift"]["password"]}' psql -A -F '\\\\001' -h{config["redshift"]["host"]} -p{config["redshift"]["port"]} -U{config["redshift"]["user"]} -d{config["redshift"]["database"]} -f {args.query} -o {args.path}/raw_{today}.csv -v v_start={args.start} -v v_end={args.end}'''
os.system(psql_cmd_down)


# ************************************************************************************************************* #
# pytorch
# ************************************************************************************************************* #

# embeddings
import torch
u_emb = torch.nn.Embedding(5,5)
i_emb = torch.nn.Embedding(5,5)

u_emb(torch.LongTensor([0,1]))
i_emb(torch.LongTensor([0,1]))
ex_model = torch.nn.Sequential(torch.nn.Linear(10,1))

users = [0,1]
items = [0,1]
cat_emb_lookups = torch.cat([u_emb(torch.LongTensor(users)),i_emb(torch.LongTensor(items))],1)
ex_model(cat_emb_lookups)

# conv1d
conv = torch.nn.Conv1d(5,1,1,padding=2,bias=False)
filt = torch.nn.Parameter(torch.Tensor([[[0],[1],[0],[0],[0]]]))
conv.weight = filt
conv(torch.Tensor([0,1,0,0,0]).view(1,5,1))
