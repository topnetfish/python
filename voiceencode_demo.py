# -*- coding: utf-8 -*-
#voice encode demo
import requests

from requests_toolbelt.multipart import encoder

 

multipart_encoder = encoder.MultipartEncoder(

    fields={

        'capKey': 'asr.cloud.freetalk',

        'audioFormat': 'pcm16k16bit',

        'voiceFile': ('格式工厂1分25秒.wav', open('格式工厂1分25秒.wav', 'rb'), 'audio/wave')

    },

    boundary='----WebKitFormBoundary7MA4YWxkTrZu0gW'

)

response =

requests.post('http://dggtsp298-or.huawei.com/mdata/api/finance/voice/asr/postToASRByOutLine',

data=multipart_encoder,

headers={'Content-Type':multipart_encoder.content_type})

print(response.json, response.text)



