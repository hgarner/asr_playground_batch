import json
import os
import argparse
import re

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-dir', dest = 'batch_dir', action = 'store')
  args = parser.parse_args()
  
  if os.path.exists(args.batch_dir):
    files_to_process = []

    for files in os.walk(args.batch_dir):
      for filename in files[2]:
        if re.match(r'.+?transcription\.json$', filename):
          files_to_process.append(os.path.join(files[0], filename))

    for filepath in files_to_process: 
      print(f'Working on file {filepath}')
      with open(filepath) as f:
        transcription = json.load(f)
      
      flat_transcription = list(map(lambda t: f'{t["speaker"]}: {t["text"]}', transcription))

      output = '\n'.join(flat_transcription)

      with open(f'{filepath}_flat.txt', 'w') as f:
        f.write(output)

