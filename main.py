import whisper
import wave
import time
from pathlib import Path
import json
import argparse
import os
import re

from pyannote.audio import Pipeline

def load_json(path):
    return json.load(open(path))

def split_audio_at_timestamp(start_in_sec, end_in_sec, audio_file, output_file):
    # file to extract the snippet from
    with wave.open(audio_file, "rb") as infile:
        # get file data
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()
        # set position in wave to start of segment
        infile.setpos(int(start_in_sec * framerate))
        # extract data
        data = infile.readframes(int((end_in_sec - start_in_sec) * framerate))

    # write the extracted data to a new file
    with wave.open(output_file, 'w') as outfile:
        outfile.setnchannels(nchannels)
        outfile.setsampwidth(sampwidth)
        outfile.setframerate(framerate)
        outfile.setnframes(int(len(data) / sampwidth))
        outfile.writeframes(data)

def diarise(filepath):
  pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
  diarization = pipeline(filepath)
  return diarization

def get_turns(diarization):
  turns = []
  for turn, _, speaker in diarization.itertracks(yield_label=True):
      turns.append({
          "start": turn.start,
          "end": turn.end,
          "speaker": speaker
      })
  return turns

def save_transcription(transcription_data, output_dir, src_filename):
  metadata_dir = os.path.join(output_dir, 'metadata')
  if not os.path.exists(metadata_dir):
    os.mkdir(metadata_dir)

  output_filename = f'{src_filename}_transcription.json'
  
  with open(os.path.join(metadata_dir, output_filename), 'w') as f:
    json.dump(transcription_data, f)

  print(f'Outputfile: {output_filename}')
  
def transcribe(split_audio_metadata, output_dir):
  print('Transcribing Audio')
  # load asr
  model = whisper.load_model('large')
  model_transcribe_start_time = time.time()

  transcriptions = []
  for turn in turns:
      result = model.transcribe(turn['file'])
      model_transcribe_end_time = time.time()
      transcriptions.append({
          'file': turn['file'],
          'text': result['text'],
          'start': turn['start'],
          'end': turn['end'],
          'speaker': turn['speaker'],
      })
      print(f'Transcribed {len(transcriptions)}/{len(turns)}')

  return transcriptions


def split_audio(filepath, turns, output_dir):
  filename = os.path.split(filepath)[1]
  print(f'Splitting file {filename}...')
  split_count = 0
  for turn in turns:
    split_count = split_count + 1
    output_filepath = os.path.join(output_dir, f'{filename}_{turn["start"]}-{turn["end"]}-{turn["speaker"]}.wav')
    split_audio_at_timestamp(turn['start'], turn['end'], filepath, output_filepath)
    turn['file'] = output_filepath

  print(f"Split {split_count}/{len(turns)}")

  return turns

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-dir', dest = 'batch_dir', action = 'store')
  parser.add_argument('--file-pattern', dest = 'file_pattern', action = 'store')
  args = parser.parse_args()
  
  if os.path.exists(args.batch_dir):
    files_to_process = []

    for files in os.walk(args.batch_dir):
      for filename in files[2]:
        if re.match(args.file_pattern, filename):
          files_to_process.append(os.path.join(files[0], filename))

    # running sequentially, could use multiprocessing
    for filepath in files_to_process: 
      print(f'Working on file {filepath}')
      base_dir, filename = os.path.split(filepath)
      # create a temp dir to drop split audio files into
      tmp_dir = os.path.join(base_dir, filename + 'tmp')
      os.mkdir(tmp_dir)

      try:
        # create the output dir
        output_dir = os.path.join(base_dir, filename + '_output')
        os.mkdir(output_dir)

        # split the audio file by speaker
        diarization = diarise(filepath)

        # get a list containing dicts each with start time, end time and speaker
        turns = get_turns(diarization)

        # split the audio file into speaker parts, save in 
        # output_dir and get the metadata back (turns with file added to each dict)
        split_audio_metadata = split_audio(filepath, turns, output_dir)

        # transcribe the files listed in the metadata
        transcription_data = transcribe(split_audio_metadata, output_dir)

        # save
        save_transcription(transcription_data, output_dir, filename)
      except Exception as e:
        raise e
      finally:
        os.rmdir(tmp_dir)

  else:
    raise ValueError(f'args.batch_dir {args.batch_dir} does not exist')
    sys.exit(1)

