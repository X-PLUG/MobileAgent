import requests

tutorial = '''The following content may help you complete the instruction:
1. Scroll down to change videos in TikTok.
2. Tap the icon of grey heart to like.
'''

image = "../results/TikTok/1/1.jpg"
query_data = {'screenshot': image, 'query': 'Find a video about cat in TikTok and tap a like for this video', 'session_id':'', 'tutorial': tutorial}
response_query = requests.post('http://127.0.0.1:5000/a', json=query_data)
print(response_query.json())

image = "../results/TikTok/1/2.jpg"
query_data = {'screenshot': image, 'query': '', 'session_id': response_query.json()['session_id'], 'tutorial': tutorial}
response_query = requests.post('http://127.0.0.1:5000/a', json=query_data)
print(response_query.json())

image = "../results/TikTok/1/3.jpg"
query_data = {'screenshot': image, 'query': '', 'session_id': response_query.json()['session_id'], 'tutorial': tutorial}
response_query = requests.post('http://127.0.0.1:5000/a', json=query_data)
print(response_query.json())

image = "../results/TikTok/1/4.jpg"
query_data = {'screenshot': image, 'query': '', 'session_id': response_query.json()['session_id'], 'tutorial': tutorial}
response_query = requests.post('http://127.0.0.1:5000/a', json=query_data)
print(response_query.json())
