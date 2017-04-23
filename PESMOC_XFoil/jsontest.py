import json


json_data = open('strings.json').read()

data = json.loads(json_data)
pprint(data)