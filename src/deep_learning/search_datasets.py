import requests

url = "https://huggingface.co/api/datasets?search=waf"
response = requests.get(url)
for ds in response.json()[:10]:
    print(ds['id'])

url2 = "https://huggingface.co/api/datasets?search=payloads"
response2 = requests.get(url2)
for ds in response2.json()[:10]:
    print(ds['id'])

