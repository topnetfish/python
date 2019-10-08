#_*_ coding:utf-8 _*_

import requests

from requests_toolbelt.multipart import encoder

from requests_toolbelt.utils import formdata

import  json

import  time

query={

    'capKey': 'tts.cloud.synth',

    'audioFormat': 'mp3',

    'voiceContext': '你好，今天天气很好',

    'boundary': '----WebKitFormBoundary7MA4YWxkTrZu0gW'

}

start = time.clock()

 

response = requests.post('http://dggtsp298-or.huawei.com/mdata/dev/api/finance/voice/tts/postToTTSByStr',

                         data=formdata.urlencode(query),

                         headers={'Content-Type':'application/x-www-form-urlencoded'},)

texts = response.text

 

elapsed = (time.clock()-start)

print ("Time used:",elapsed)

# print(response.json, texts)

 

tags = [item[1] for item in [item.split(":")for item in texts.split(',')]]

tag =tags[2][1:-2]

# print(tag)

header = {'filePath':tag}

r = requests.get('http://dggtsp298-or.huawei.com/mdata/dev/api/finance/voice/downloadTTSFile',headers = header)

# print(r.url)

# print (r.text)

# print(r.status_code)

# print (r.json)

# #print(r.headers['content-type'])

# print(r.encoding)

# print(r.content)

 

 

 

with open('ceshi.mp3','wb') as f:

    f.write(r.content)

 

elapsed = (time.clock()-start)

print ("Time used:",elapsed)