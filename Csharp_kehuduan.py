# -*- coding: utf-8 -*-
"""
Created on Sun May 26 00:53:39 2019

@author: peng.zhou
"""
import requests
import base64
def strToImage(str,filename):
    image_str= str.encode('ascii')
    image_byte = base64.b64decode(image_str)
    image_json = open(filename, 'wb')
    image_json.write(image_byte)  #将图片存到当前文件的fileimage文件中
    image_json.close()
a=input("please input a count:\n")
a=int(a)
print("the {} now is inputted".format(a))
url_1="http://10.139.30.77:5000/1"
url_4="http://10.139.30.77:5000/4"
url_2="http://10.139.30.77:5000/2"
if  a==1:
    r = requests.post(url_1)
    response_data = r.json()
    print("response_data is:\n",response_data)
    file_address = "./fileimage/zhou.jpg"
    strToImage(response_data['img'],file_address) 
    print("results is:\n",r)
    print ("succesful")

elif a==2:
    r = requests.post(url_2)
    result = r.text
    print (result)
elif a==4:
    r = requests.post(url_4)
    result = r.text
    print (result)
    