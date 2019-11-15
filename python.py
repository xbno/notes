# Notes:

# show version
pd.__version__
# show file location
pd.__file__

# ************************************************************************************************************* #
# in jupyter notebooks/lab
# ************************************************************************************************************* #

# install fastai
conda create --name fastai python=3.7
conda update --prefix /Users/xbno/anaconda3 anaconda
conda install fastai -c fastai -c pytorch -c conda-forge
conda install ipykernel
python -m ipykernel install --user --name fastai

# add conda env as kernel to lab or notebook
conda install ipykernel
python -m ipykernel install --user --name mykernel

# on server use:
import matplotlib
matplotlib.use('agg')
%matplotlib inline

# autoreload imports
%load_ext autoreload
%autoreload 2

# remove ... in dataframes
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
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

# convert from unix timestamp to readable
pd.to_datetime(df['userTags_expires_at'],unit='s')

# timezones
pd.Timestamp('09/13/19 10:46').tz_localize('America/New_York').tz_convert('Asia/Singapore')

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

# referencing a string dictionary or json in df
edf['json'] = edf['json'].apply(lambda x: json.loads(x))
edf['external_id'] = edf['json'].apply(lambda x: x['external_id'])

# find only numbers
def only_nums(x):
    return re.findall('(\d*)',x)[0]
combined_df['workphone'] = combined_df['workphone'].apply(only_nums)

# extract strings regex from df
error_codes = {'age':'age .* is out of range',
        'zip_usa':'unrecognized zip .* for country USA'}

error_values = {'age':'age (.*) is out of range',
        'zip_usa':'unrecognized zip (.*) for country USA'}

for code,error in error_codes.items():
    edf.loc[edf['error'].str.contains(error,regex=True),'error_code'] = code
for code,error_value in error_values.items():
    edf[f'{code}_error'] = edf['error'].str.extract(error_value)


# quintile
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

# logging simple one liner at top
logging.basicConfig(filename='logs.log', level=logging.DEBUG, format="%(asctime)s:%(levelname)s: %(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

# formatting
columns = '(origin_id serial primary key, '+' text, '.join([c for c in df.columns if 'origin_id' not in c]) + ' text);'
create_cmd = f'''-c "create table if not exists {table} {columns}"'''

'.26' = '{:.2f}'.format(.235872)
'.26' = f'{.235872}'
'01' = '{:02d}'.format(1)
'100203' = '{:.0f}'.format(100203.021)

# partial formatting f string
f"{base_path}/output/smsync/{date_parts}/{{recommendation_model}}---{uparams['segmentation_model']}.json"

# add a,b,c suffix to filename
letter = chr(ord('a')+i)
file_to_upload = f'{args.path}/std_apd_email_{args.sample}{letter}.json'

# datetimes
from datetime import datetime, timedelta

# from datetime to string
today = datetime.now().strftime('%Y-%m-%d')

# from string to datetime
datetime.strptime('20180531', '%Y%m%d')

# add day
datetime.strftime(datetime.now()+timedelta(days=1),'%Y-%m-%d')

# show all timezones, pytz tz
import pytz
pytz.all_timezones_set

from dateutils import tz
def utc2sing(dt,from_zone=tz.gettz('UTC'),to_zone=tz.gettz('Asia/Singapore')):
    return dt.tz_localize(from_zone).tz_convert(to_zone)

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

# read files
from glob import glob

df = pd.DataFrame()
for fp in glob('./*_fill_cnt.csv'):
    df = pd.concat([df,pd.read_csv(fp)])

# ************************************************************************************************************* #
# pytest with decorator and class init
# ************************************************************************************************************* #

"""python -m pytest -v"""

import pytest
import collections
import pandas as pd
import inc_GR_m # located within the same folder as the tests script

@pytest.mark.parametrize("raw_s3_files,todays,init_map_s3_file,expected_map_s3_file,creation_date_col", [
    # kelly: 3 record on 2019-04-02 should match, however creation_date is the same for all records, so
    # new record should not become master_sm_id when matching
    (['s3://teams-ent-sessionm-com/admteam/integration/wellbiz/test_data/tmp/date=2019-04-02/kelly.csv.gz',
     's3://teams-ent-sessionm-com/admteam/integration/wellbiz/test_data/tmp/date=2019-04-21/kelly.csv.gz'],
     ['2019-04-02','2019-04-21'],
     's3://teams-ent-sessionm-com/admteam/integration/wellbiz/test_data/map/date=2019-04-02/kelly_map.csv',
     's3://teams-ent-sessionm-com/admteam/integration/wellbiz/test_data/map/date=2019-04-22/kelly_map.csv',
     'creation_date_coalesce'),
])
def test_inc_GR_over_time(raw_s3_files, init_map_s3_file, todays, expected_map_s3_file, creation_date_col):
    "Test incremental golden record matching over time, comparing map_df built over time vs expected_map_df"

    i = 0
    for raw_s3_file,today in zip(raw_s3_files,todays):
        if i == 0:
            df, map_df = inc_GR_m.load_df_and_map(None,raw_s3_file,init_map_s3_file)
            df = select_creation_date_col(df,creation_date_col)
        else:
            df, throw_away_map_df = inc_GR_m.load_df_and_map(None,raw_s3_file,init_map_s3_file)
            df = select_creation_date_col(df,creation_date_col)

        bf,new_sm_ids = inc_GR_m.inc_GR(df,map_df,today)

        if len(new_sm_ids) > 0:
            map_df = inc_GR_m.extend_map(map_df,bf,new_sm_ids,today)

        i += 1

    # set/sort so they'll be comparable
    map_df.reset_index(drop=True,inplace=True)
    expected_map_df = pd.read_csv(expected_map_s3_file)
    map_df = map_df.sort_values(by=['updated_at','sm_id'])
    expected_map_df = expected_map_df.sort_values(by=['updated_at','sm_id'])

    print('map_df',map_df)
    print('expected_map_df',expected_map_df)
    assert (map_df.reset_index(drop=True) == expected_map_df.reset_index(drop=True)).all().all()

# ************************************************************************************************************* #
# data locally
# ************************************************************************************************************* #

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

# if running from different profiles, set prior to running python or within
export AWS_PROFILE=ent

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

# boto3 GzipFile (on lambdas)
import numpy as np
import boto3
import gzip
import pandas as pd
from io import BytesIO, TextIOWrapper

# write, save as a gz_buffer in memory
s3_loc = 'teams-ent-sessionm-com'
fname = 'admteam/integration/wellbiz/std_apd_noemail_25000.json.gz'

ex_json = {'a':[1,2,3]}

gz_buffer = BytesIO()
with gzip.GzipFile(mode='w', fileobj=gz_buffer) as fout:
    fout.write(json.dumps(ex_json).endocde('utf-8'))

s3_resource = boto3.resource('s3')
s3_resource.Object(s3_loc, fname).put(Body=gz_buffer.getvalue())

# read
s3 = boto3.client('s3')
obj = s3.get_object(Bucket=s3_loc, Key=fname)

with gzip.GzipFile(mode='r', fileobj=obj['Body']) as fin:
    data = json.loads(fin.read().decode('utf-8'))

# s3fs
# if assuming instance has proper permissions, make sure to assign an IAM role to it to inherit,or pass profile locally
import s3fs
s3 = s3fs.S3FileSystem() # or s3 = s3fs.S3FileSystem(profile_name=profile)
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

# fastai 1.0.51
il = ItemList.from_df(df[['item_id','label']],cols=['item_id'],label_cls=MultiCategoryList)
sd = il.split_by_rand_pct()
data = sd.label_from_df('label',label_delim=' ')


            'mazu_std_s3_path': f's3://{self.kwargs["mazu_s3_bucket"]}/{self.kwargs["client"]}/warehouse',
            'mazu_dd_s3_path': f's3://{self.kwargs["mazu_s3_bucket"]}/{self.kwargs["client"]}/data_dump'
