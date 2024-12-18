import pyaudio
import webrtcvad
import numpy as np
import noisereduce as nr
import whisper
import threading
import queue
from scipy.signal import butter, lfilter 
import google.generativeai as genai
import os
from dotenv import load_dotenv
from boardprocessing import BoardView
import time
import pyautogui

load_dotenv()

# audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 480
SILENCE_THRESHOLD = 10

vad = webrtcvad.Vad(0)  # low sensitivity so only clear speech works
model = whisper.load_model("small.en")

# this is a global, there's only one board across threads. I make sure to lock and unlock when I need to use it.
board = BoardView()
lock = threading.Lock()

def main():
    board_thread = threading.Thread(target=board_process_thread)
    board_thread.start()
    process_audio()

# thread for processing the screen to detect and keep track of the chessboard onscreen
def board_process_thread():
    # update board info every 5 seconds or so
    while True:
        with lock:
            board.process_board()
        time.sleep(3)

def process_audio():
    p = pyaudio.PyAudio()
    # this raw data queue is going to contain raw data from stream.read and will be processed inthe the transcribe_thread
    raw_data_queue = queue.Queue()
    transcribe_thread = threading.Thread(target=transcribe_process, args=(raw_data_queue,))
    transcribe_thread.start()
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
    while True: 
        try:
            # reading the from the stream happens in the main thread constantly, so the pyaudio buffer doesn't overflow
            audio_data = stream.read(CHUNK)
            raw_data_queue.put(audio_data)
        except IOError as e:
            print(f"Buffer overflow error: {e}")
            # restart the stream
            stream.close()
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        except KeyboardInterrupt:
            print("end")
            return

# band pass filter to try and get only human frequencies
def band_pass_filter(data, low_cut, high_cut):
    nyquist = 0.5 * RATE
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = butter(1, [low, high], btype="band")
    return lfilter(b, a, data)

def transcribe_process(raw_data_queue):
    # buffer that holds audio data, when the user has spoken a phrase, the buffer data will be processed by Whisper model
    audio_buffer = b""
    # silence counter is so that we have a couple frames of silence so that we don't cut off a phrase early
    silence_counter = 0
    while True:
        audio_data = raw_data_queue.get()
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        reduced_audio = nr.reduce_noise(y=audio_np, sr=RATE)
        low_cut = 500.0  # Lower cutoff frequency in Hz
        high_cut = 2000.0  # Upper cutoff frequency in Hz
        filtered_audio = band_pass_filter(reduced_audio, low_cut, high_cut)

        # applying noise reduction, it expects a np array for the data
        reduced_audio_data = filtered_audio.astype(np.int16).tobytes()

        # if it is user speaking, we add the audio data it to a buffer
        if vad.is_speech(reduced_audio_data, RATE):
            audio_buffer += audio_data            
            silence_counter = 0     
        else:
            silence_counter += 1
        
        # been silent for atleast 3 collection of frames and there's stuff in the buffer
        if silence_counter >= SILENCE_THRESHOLD:
            if len(audio_buffer) > 20480:
                data_np = np.frombuffer(audio_buffer, dtype=np.int16)
                result = model.transcribe(data_np.astype(np.float32) / 32768.0, language="en", fp16=False, verbose=False)
                print(result['text'])
                # if there is some text at all try to send it
                if (len(result["text"])) >= 2 and contains_pieceword(result["text"]):
                    with lock:
                        print(board.fen)
                        print(board.orientation)
                        # give the text to gemini and let it figure it out.
                        genai.configure(api_key=os.getenv("GEMINI_API"))
                        genaiModel = genai.GenerativeModel("gemini-1.5-flash")
                        response = genaiModel.generate_content(f"""This is the FEN of a chess game {board.fen}. 
                                                                Additionally you could have the state of the board from the white perspective as {board.board}
                                                                Here is the users intended next move {result["text"]}.
                                                                User is currently playing as {board.orientation} color. 
                                                                Be careful as there may be multiple pieces of the same name, for example 2 white bishops at different locations.
                                                                Be sure to calculate if the move is possible and pick the piece that the user most likely wants to move.
                                                                Either respond just "N/A" if the next move cannot be confidently determined,
                                                                or "rank,file - rank,file" where rank and file are the moves rank and file. 
                                                                Keep file as letters and rank in numbers in your output. The first rank,file pair is for the initial location of the piece. 
                                                                The second rank,file pair is the end location after the move.
                                                                Additionally, please keep the comma in between the rank and file.
                                                                For situtations where the user wants to castle, put the king as the start  and the rook on the castling side as the destination.
                                                                Please sure to be accurate in your answer as that is paramount. Also double check. Thank you""")
                        # time to parse response.text to actually make the moves
                        print(response.text)
                        try:
                            split = response.text.strip().split("-")
                            if len(split) == 2:
                                first = split[0]
                                second = split[1]
                                file_to_num = {"a" : 0, "b" : 1, "c" : 2, "d":3, "e":4, "f": 5, "g": 6, "h":7}
                                start_file = file_to_num[first.split(",")[1].strip().lower()]
                                start_rank = int(first.split(",")[0].strip())
                                end_file = file_to_num[second.split(",")[1].strip().lower()]
                                end_rank = int(second.split(",")[0].strip())
                                start_coord = board.board_coords[8 - start_rank][start_file]
                                end_coord = board.board_coords[8 - end_rank][end_file]
                                print(board.board_coords)
                                print(start_rank, start_file, end_rank, end_file)
                                print(start_coord, end_coord)
                                # click the start square, then the end
                                pyautogui.click(start_coord[0], start_coord[1])
                                # click and draw
                                pyautogui.moveTo(start_coord[0], start_coord[1])
                                pyautogui.mouseDown(button='left')
                                pyautogui.moveTo(end_coord[0], end_coord[1], 0.1)
                                pyautogui.mouseUp(button="left")
                        except Exception as e:
                            print(e)
            audio_buffer = b""            

def contains_pieceword(string):
    pieces = {"king", "queen", "bishop", "rook", "pawn", "knight", "castle", "castling", "night"}
    string = string.lower()
    words = set(string.split())
    return bool(words & pieces)
main()


