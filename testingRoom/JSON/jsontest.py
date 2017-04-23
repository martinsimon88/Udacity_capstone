# -*- coding: utf-8 -*-
import json
import optparse
# Make it work for Python 2+3 and with Unicode
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

# Define data
data = {'a list': [1, 42, 3.141, 1337, 'help', u'€'],
        'a string': 'bla',
        'another dict': {'foo': 'bar',
                         'key': 'value',
                         'the answer': 42}}

# Write JSON file
with io.open('data.json', 'w', encoding='utf8') as outfile:
    str_ = json.dumps(data,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
    outfile.write(to_unicode(str_))

# Read JSON file
with open('data2.json') as data_file:
    data_loaded = json.load(data_file)

print(data == data_loaded)

##########
'''
data = {}
data['people'] = []
data['people'].append({
    'name': 'Scott',
    'website': 'stackabuse.com',
    'from': 'Nebraska'
})
data['people'].append({
    'name': 'Larry',
    'website': 'google.com',
    'from': 'Michigan'
})
data['people'].append({
    'name': 'Tim',
    'website': 'apple.com',
    'from': 'Alabama'
})

with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)

'''



##########

with open('data3.json') as json_file:
    data = json.load(json_file)
    for p in data['people']:
        if p.has_key('min'):
            z = ['min']
            print z




'''
with open('config.json') as json_file:
    data = json.load(json_file)
    for v in data['variables']:
        for val in v['x']:
            print('Name: ' + v['type'])
            print('Website: ' + v['y'])
            print('')

'''