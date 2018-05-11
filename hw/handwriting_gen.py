import subprocess
import csv

model = 'pretrained/custom/models/model-40'
sample = []
with open('/Users/grace.han/Documents/Work/auspost/nets/hw/lexicon_suburbs.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row["word"]
        text = text[0].upper() + text[1:]
        sample.append(text)

for text in sample:
    for style in range(5):
        subprocess.call(["python", "generate.py", "--noinfo", "--style", str(style), "--bias=3", "--model", model, "--text", text])
