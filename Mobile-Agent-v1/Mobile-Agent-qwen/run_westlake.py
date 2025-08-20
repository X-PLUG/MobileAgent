import requests

tutorial = '''The following content may help you complete the instruction:
1. Use \"Maps\" app for navigation.
2. Tap \"Search here\" to activate the search box before typing.
3. Type \"Hangzhou West Lake\" in search box to search the target location.
4. Tap \"West Lake\" of the search results to get the location.
5. Tap \"Start\" to start the navigation.
'''

image = "../results/Google Maps/1/1.jpg"
query_data = {'screenshot': image, 'query': 'Navigate to Hangzhou West Lake', 'session_id':'', 'tutorial': tutorial}
response_query = requests.post('http://127.0.0.1:5000/a', json=query_data)
print(response_query.json())

image = "../results/Google Maps/1/2.jpg"
query_data = {'screenshot': image, 'query': '', 'session_id': response_query.json()['session_id'], 'tutorial': tutorial}
response_query = requests.post('http://127.0.0.1:5000/a', json=query_data)
print(response_query.json())

image = "../results/Google Maps/1/3.jpg"
query_data = {'screenshot': image, 'query': '', 'session_id': response_query.json()['session_id'], 'tutorial': tutorial}
response_query = requests.post('http://127.0.0.1:5000/a', json=query_data)
print(response_query.json())

image = "../results/Google Maps/1/4.jpg"
query_data = {'screenshot': image, 'query': '', 'session_id': response_query.json()['session_id'], 'tutorial': tutorial}
response_query = requests.post('http://127.0.0.1:5000/a', json=query_data)
print(response_query.json())

image = "../results/Google Maps/1/5.jpg"
query_data = {'screenshot': image, 'query': '', 'session_id': response_query.json()['session_id'], 'tutorial': tutorial}
response_query = requests.post('http://127.0.0.1:5000/a', json=query_data)
print(response_query.json())

image = "../results/Google Maps/1/6.jpg"
query_data = {'screenshot': image, 'query': '', 'session_id': response_query.json()['session_id'], 'tutorial': tutorial}
response_query = requests.post('http://127.0.0.1:5000/a', json=query_data)
print(response_query.json())
